# -*- coding: utf-8 -*-

# =========================
# 显存管理（2.4.x 稳定）
# =========================
import os

# ---- 强制移除/覆盖 expandable_segments:True ----
_default_alloc_conf = "max_split_size_mb:128,garbage_collection_threshold:0.8"
_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", _default_alloc_conf)
# 去掉任何包含 expandable_segments 的片段
_conf = ",".join([p for p in _conf.split(",") if not p.strip().startswith("expandable_segments") and p.strip() != ""])
if not _conf:
    _conf = _default_alloc_conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _conf
# ----------------------------------------------

import sys, math, gc, random, time, inspect, re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Sampler

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

from transformers import AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# 统一用 torch.amp 新接口
from torch import amp as torch_amp


# -----------------------
# 超参区（可按需通过环境变量覆盖）
# -----------------------

# === 路径（默认基于仓库目录） ============================================
REPO_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("PYFLAKY_HOME", REPO_DIR))

MODEL_NAME = os.getenv("MODEL_NAME", str(BASE_DIR / "models" / "graphcodebert-base"))
DATASET_PATH = Path(os.getenv("DATASET_PATH", str(BASE_DIR / "dataset" / "python_dataset_generated.xlsx")))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S")
MODEL_WEIGHTS_PATH = OUTPUT_DIR / f"graphcodebert_chunkpool_{STAMP}.bin"
RESULTS_FILE = OUTPUT_DIR / f"results_{STAMP}.csv"

dataset_path = DATASET_PATH
model_weights_path = MODEL_WEIGHTS_PATH
results_file = RESULTS_FILE

# === Transformers 离线 & 并发 =============================================
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# === 数据集类型：python | java  （决定列名、提示与规则）===
DATASET_KIND = os.getenv("DATASET_KIND", "python").strip().lower()
assert DATASET_KIND in {"python", "java"}, "DATASET_KIND 必须是 python 或 java"

# === 训练超参（4090 24GB 稳妥默认） =======================================
MAX_LEN    = int(os.getenv("MAX_LEN", 512))
WINDOW_OVERLAP_TOKENS = int(os.getenv("WINDOW_OVERLAP_TOKENS", 256))

LR        = float(os.getenv("LR", 2e-5))
EPOCHS    = int(os.getenv("EPOCHS", 5))
BATCH_TRAIN = int(os.getenv("BATCH_TRAIN", 4))
BATCH_EVAL  = int(os.getenv("BATCH_EVAL", 8))
SEED      = int(os.getenv("SEED", 42))
N_SPLITS  = int(os.getenv("N_SPLITS", 10))
AGG_METHOD = os.getenv("AGG_METHOD", "mean")   # "mean" | "max"
MERGE_OD_NOD = (os.getenv("MERGE_OD_NOD", "1") != "0")

# === 不平衡处理（默认只用类权重；需要过采样再改这里） =========================
OVERSAMPLE_METHOD = os.getenv("OVERSAMPLE_METHOD", "smote_like")   # "smote_like" | "ros" | "none"
TARGET_POS_RATIO = float(os.getenv("TARGET_POS_RATIO", 0.5)) # 若 smote/ros，用 0.5 更稳
SVD_DIM = int(os.getenv("SVD_DIM", 256))
K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", 5))

# === 显存相关（微批 + 累积） ===============================================
GRAD_ACC_STEPS = int(os.getenv("GRAD_ACC_STEPS", 8))
MAX_CHUNKS_PER_SAMPLE = int(os.getenv("MAX_CHUNKS_PER_SAMPLE", 8))
MICRO_CHUNK = int(os.getenv("MICRO_CHUNK", 8))

# === 任务提示（固定 Prompt + 可选 Auto-Hint）===  （1=开，0=关）
USE_TASK_PROMPT = (os.getenv("USE_TASK_PROMPT", "1") != "0")
PROMPT_LANG     = os.getenv("PROMPT_LANG", "en")        # "en" 或 "zh"
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 260))

USE_AUTO_HINT   = (os.getenv("USE_AUTO_HINT", "1") != "0")
MAX_AUTO_HINT_CHARS = int(os.getenv("MAX_AUTO_HINT_CHARS", 120))

# === 新增：调度与提前停止 ===
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", 0.06))   # 5~10% 都可
SCHEDULER = os.getenv("SCHEDULER", "cosine")            # "cosine" 或 "linear"
PATIENCE = int(os.getenv("PATIENCE", 3))                # 连续 N 个 epoch 无提升则停
MIN_DELTA = float(os.getenv("MIN_DELTA", 1e-3))         # 认为“有提升”的最小幅度

# === 新增：两层 MLP 头部超参 ===
HEAD_HIDDEN = int(os.getenv("HEAD_HIDDEN", 512))
HEAD_DROPOUT = float(os.getenv("HEAD_DROPOUT", 0.1))

# === 新增：总开关（最少改动的“开关式”） ===
USE_SCHEDULER = (os.getenv("USE_SCHEDULER", "1") != "0")   # 设为 0 关闭学习率调度
USE_EARLY_STOP = (os.getenv("USE_EARLY_STOP", "1") != "0") # 设为 0 关闭提前停止
# （可选）梯度检查点开关（默认保持开启行为）
USE_CHECKPOINT = (os.getenv("USE_CHECKPOINT", "1") != "0")


# -----------------------
# 稳定性 & 设备
# -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(require_cuda=True):
    info = []
    try:
        info.append(f"python: {sys.executable}")
        info.append(f"torch: {torch.__version__} (cuda runtime: {torch.version.cuda})")
        info.append(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        info.append(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            info.append(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        info.append(f"[diag error] {e}")
    print("[GPU DIAG] " + " | ".join(info))

    if require_cuda and (not torch.cuda.is_available()):
        raise RuntimeError(
            "未检测到可用 GPU：请确认安装 GPU 版 torch、解释器选择正确，或未隐藏 GPU。"
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = pick_device(require_cuda=True)
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 设备：{device}")
set_seed(SEED)

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def load_dataset_any(path: Path) -> pd.DataFrame:
    print(f"[{now()}] 读取数据集：{path}")
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    suf = path.suffix.lower()
    if suf in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif suf in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(path, sheet_name=0)
        except ImportError as e:
            raise ImportError("需要 openpyxl：`python -m pip install openpyxl`") from e
    else:
        raise ValueError(f"不支持的文件类型: {suf}（仅支持 .csv/.xlsx）")
    print(f"[{now()}] 数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
    return df


# -----------------------
# 列名选择 & 标签映射
# -----------------------
def pick_cols(df: pd.DataFrame, kind: str) -> Tuple[str, str]:
    lower2orig = {c.lower(): c for c in df.columns}
    if kind == "java":
        label_candidates = ["flaky"]  # Java 固定用 flaky
    else:
        label_candidates = ["flaky", "is_flaky", "label"]
    label_key = next((k for k in label_candidates if k in lower2orig), None)
    if label_key is None:
        raise ValueError(f"未找到标签列（{kind} 期望之一：{label_candidates}，不区分大小写）")
    label_col = lower2orig[label_key]

    code_priority = ["final_code", "extractedcode", "code"] if kind == "java" \
                    else ["extractedcode", "final_code", "code"]
    code_key = next((k for k in code_priority if k in lower2orig), None)
    if code_key is None:
        raise ValueError(f"未找到代码列（{kind} 期望之一：{code_priority}，不区分大小写）")
    code_col = lower2orig[code_key]
    return code_col, label_col

def map_label_merge(x: str) -> int:
    s = str(x).strip().lower()
    if s in {"od", "nod", "flaky", "is_flaky", "yes", "1", "true"}: return 1
    if s in {"not flaky", "non-flaky", "not_flaky", "no", "0", "false"}: return 0
    return 0

def map_label_separate(x: str) -> int:
    s = str(x).strip().lower()
    if s == "od": return 1
    if s in {"nod", "not flaky", "non-flaky", "not_flaky", "no", "0", "false"}: return 0
    return 0


# -----------------------
# 任务提示（按数据集类型切换）
# -----------------------
PROMPTS = {
    "python": {
        "en": (
            "Task: Decide if the following *Python* unit test is FLAKY (1=flaky, 0=not-flaky). "
            "Common flaky patterns: timing/sleep/timeouts; async/await races; order dependence; "
            "unseeded randomness; network/HTTP or external services; filesystem/tmp dirs; "
            "system clock/timezone; shared global/module state; env/config diffs; subprocess; UI/webdriver."
        ),
        "zh": (
            "任务：判断下面的 *Python* 单元测试是否为 Flaky（1=Flaky，0=非Flaky）。"
            "常见 Flaky 模式：时序/休眠/超时、async/await 竞态、测试顺序依赖、未固定随机数、"
            "网络/HTTP 或外部服务、文件系统/临时目录、系统时钟/时区、模块/全局共享状态、"
            "环境变量/配置差异、子进程、UI/WebDriver 等。"
        )
    },
    "java": {
        "en": (
            "Task: Decide if the following *Java* unit test is FLAKY (1=flaky, 0=not-flaky). "
            "Common flaky patterns: timing/Thread.sleep/timeouts/Awaitility; async/concurrency races "
            "(Future/CompletableFuture/executors); order dependence (JUnit/TestNG); unseeded randomness; "
            "network/HTTP/external services; filesystem; system clock; shared static/global state; "
            "env/config differences; mocks flakiness; UI/WebDriver."
        ),
        "zh": (
            "任务：判断下面的 *Java* 单元测试是否为 Flaky（1=Flaky，0=非Flaky）。"
            "常见 Flaky 模式：时序/Thread.sleep/超时/Awaitility，异步/并发竞态（Future/CompletableFuture/线程池），"
            "JUnit/TestNG 的顺序依赖、未固定随机数、网络/HTTP/外部服务、文件系统、系统时钟、"
            "静态/全局共享状态、环境/配置差异、Mock 不稳定、UI/WebDriver 等。"
        )
    }
}

def _task_prompt_text():
    lang = "zh" if PROMPT_LANG.lower().startswith("zh") else "en"
    base = PROMPTS[DATASET_KIND][lang]
    return base[:MAX_PROMPT_CHARS]

_HINT_PATTERNS_PY = [
    (r"\btime\.sleep\(", "timing/sleep"),
    (r"\basyncio\.sleep\(", "timing/async-sleep"),
    (r"\bpytest\.mark\.timeout\b", "timeout"),
    (r"\bpytest\.mark\.asyncio\b|\basyncio\b|\btrio\b|\banyio\b", "async/asyncio"),
    (r"\bthreading\.(Thread|Lock|Event|Condition|Semaphore)\b|\bconcurrent\.futures\b", "threads/concurrency"),
    (r"\bmultiprocessing\.(Process|Pool)\b", "multiprocessing"),
    (r"\brandom\.(random|randint|choice|shuffle|sample)\b|\bnumpy\.random\b|\buuid\.uuid4\b", "randomness"),
    (r"\brequests\.\w+\(|\bhttpx\.\w+\(|\baiohttp\.\w+\(|\burllib\.request\.\w+\(|\bsocket\.", "network-io"),
    (r"\bboto3\b|\bredis\b|\bpika\b|\bkafka\b|\belasticsearch\b|\bpsycopg2\b|\bsqlalchemy\.create_engine\b", "external service/db"),
    (r"\bopen\(|\bpathlib\.Path\b|\bos\.path\b|\btempfile\b|\bshutil\.", "fs-io"),
    (r"\btime\.(time|monotonic)\(|\bdatetime\.(now|utcnow|today)\(", "clock"),
    (r"\bfreezegun\.", "time-freeze"),
    (r"\bos\.environ\b|\bos\.getenv\(", "env/config"),
    (r"\bmonkeypatch\.(setenv|setitem|setattr)\b", "monkeypatching"),
    (r"\bpytest\.mark\.order\b|\bpytest-order\b", "order-dependence"),
    (r"\bsubprocess\.(run|Popen|check_call|check_output)\b", "subprocess/external"),
    (r"\bselenium\.webdriver\b|\bplaywright\.", "ui/webdriver"),
]

_HINT_PATTERNS_JAVA = [
    (r"\bThread\.sleep\b|\bTimeUnit\.\w+\.sleep\b|\bAwaitility\b", "timing/sleep"),
    (r"\bFuture\b|\bCompletableFuture\b|\bExecutor(Service)?\b|\bForkJoinPool\b", "async/concurrency"),
    (r"@Order|@FixMethodOrder|dependsOnMethods", "order-dependence"),
    (r"\bRandom\b|\bMath\.random\b", "randomness"),
    (r"\bHttp(Client|URLConnection)\b|\bSocket\b|\bOkHttp\b|\bRestTemplate\b|\bWebClient\b", "network-io"),
    (r"\bFile(Input|Output)Stream\b|\bPaths?\b|\bFiles\b", "fs-io"),
    (r"\bSystem\.currentTimeMillis\b|\bClock\b|\bInstant\.now\b", "clock"),
    (r"\bvolatile\b|\bsynchronized\b|\bwait\(|\bnotify\(", "locks/wait-notify"),
    (r"\bSystem\.getenv\b|\bSystem\.getProperty\b|(?:^|\s)-D\w+", "env/config"),
    (r"\bMockito\.(when|verify|spy|mock)\b|\bPowerMockito\b", "mocks"),
    (r"\bSelenium\b|\bWebDriver\b", "ui/webdriver"),
]

def _patterns_for_kind():
    return _HINT_PATTERNS_PY if DATASET_KIND == "python" else _HINT_PATTERNS_JAVA

def _auto_hint(code: str) -> str:
    if not USE_AUTO_HINT or not code:
        return ""
    tags = []
    for pat, name in _patterns_for_kind():
        try:
            if re.search(pat, code, flags=re.I):
                tags.append(name)
        except re.error:
            pass
        if len(tags) >= 5:
            break
    if not tags:
        return ""
    text = " HINT: " + ", ".join(tags) + "."
    return text[:MAX_AUTO_HINT_CHARS]

def build_input_with_prompt(code_text: str) -> str:
    code_text = code_text or ""
    if not USE_TASK_PROMPT and not USE_AUTO_HINT:
        return code_text
    prompt = _task_prompt_text() if USE_TASK_PROMPT else ""
    ahint  = _auto_hint(code_text) if USE_AUTO_HINT else ""
    if prompt or ahint:
        return f"{prompt}{ahint}\n\n{code_text}"
    return code_text


# -----------------------
# 分块编码（每样本 → 多块）
# -----------------------
def encode_chunks_per_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_len: int = 512,
    overlap_tokens: int = 256,
    cap_chunks: int = MAX_CHUNKS_PER_SAMPLE,
) -> List[Dict[str, torch.Tensor]]:
    out = []
    chunk_size = max_len - 2
    step = max(1, chunk_size - overlap_tokens)

    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        chunk_inputs = []
        if len(ids) == 0:
            piece_ids = tokenizer.build_inputs_with_special_tokens([])
            attn = [1] * len(piece_ids)
            if len(piece_ids) > max_len:
                piece_ids = piece_ids[:max_len]; attn = attn[:max_len]
            else:
                pad_len = max_len - len(piece_ids)
                piece_ids += [tokenizer.pad_token_id] * pad_len
                attn += [0] * pad_len
            chunk_inputs.append((torch.tensor(piece_ids, dtype=torch.long),
                                 torch.tensor(attn, dtype=torch.long)))
        else:
            start = 0
            while start < len(ids):
                piece = ids[start:start+chunk_size]; start += step
                piece_ids = tokenizer.build_inputs_with_special_tokens(piece)
                attn = [1] * len(piece_ids)
                if len(piece_ids) > max_len:
                    piece_ids = piece_ids[:max_len]; attn = attn[:max_len]
                else:
                    pad_len = max_len - len(piece_ids)
                    piece_ids += [tokenizer.pad_token_id] * pad_len
                    attn += [0] * pad_len
                chunk_inputs.append((torch.tensor(piece_ids, dtype=torch.long),
                                     torch.tensor(attn, dtype=torch.long)))
                if len(chunk_inputs) >= cap_chunks:
                    break

        X_ids  = torch.stack([ci[0] for ci in chunk_inputs], dim=0)   # [C,L]
        X_mask = torch.stack([ci[1] for ci in chunk_inputs], dim=0)   # [C,L]
        out.append({"input_ids": X_ids, "attention_mask": X_mask})
    return out


# -----------------------
# 数据集
# -----------------------
class OwnerChunkDataset(Dataset):
    def __init__(self, chunks_per_sample: List[Dict[str, torch.Tensor]], labels: List[int]):
        assert len(chunks_per_sample) == len(labels)
        self.data = chunks_per_sample
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_ids"], item["attention_mask"], self.labels[idx]

def collate_owner_batch(batch):
    ids_list, mask_list, labels = zip(*batch)
    B = len(ids_list)
    L = ids_list[0].size(-1)
    Cmax = max(x.size(0) for x in ids_list)

    pad_ids  = torch.zeros(B, Cmax, L, dtype=torch.long)
    pad_mask = torch.zeros(B, Cmax, L, dtype=torch.long)
    chunk_mask = torch.zeros(B, Cmax, dtype=torch.bool)

    for i in range(B):
        C = ids_list[i].size(0)
        pad_ids[i, :C]  = ids_list[i]
        pad_mask[i, :C] = mask_list[i]
        chunk_mask[i, :C] = True

    labels = torch.stack(labels, dim=0)
    return pad_ids, pad_mask, chunk_mask, labels


# -----------------------
# 过采样（可选）
# -----------------------
def smote_like_oversample(texts: np.ndarray, labels: np.ndarray,
                          target_pos_ratio: float = 1.0,
                          svd_dim: int = 256,
                          k_neighbors: int = 5,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    texts = np.asarray(texts).ravel()
    labels = np.asarray(labels, dtype=int).ravel()

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    n_pos_target = int(target_pos_ratio * n_neg)
    if n_pos >= n_pos_target:
        return texts, labels

    n_to_add = n_pos_target - n_pos
    if n_pos < 2:
        add_idx = rng.choice(pos_idx, size=n_to_add, replace=True)
        return np.concatenate([texts, texts[add_idx]]), np.concatenate([labels, np.ones(n_to_add, dtype=int)])

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(texts[pos_idx])
    svd = TruncatedSVD(n_components=min(svd_dim, max(2, X_tfidf.shape[1]-1)))
    X_dense = svd.fit_transform(X_tfidf)

    K = min(k_neighbors, max(1, X_dense.shape[0]-1))
    nn = NearestNeighbors(n_neighbors=K+1, metric="euclidean").fit(X_dense)
    neigh = nn.kneighbors(X_dense, return_distance=False)[:, 1:]

    anchors = rng.integers(0, len(pos_idx), size=n_to_add)
    cols    = rng.integers(0, neigh.shape[1], size=n_to_add)
    pick_in_anchor_neigh = neigh[anchors, cols]
    dup_pos_idx = pos_idx[pick_in_anchor_neigh]

    texts_new  = np.concatenate([texts, texts[dup_pos_idx]], axis=0)
    labels_new = np.concatenate([labels, np.ones(n_to_add, dtype=int)], axis=0)
    return texts_new, labels_new


# -----------------------
# 分桶 BatchSampler：每个 epoch 打乱
# -----------------------
class BucketedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size: int, seed: int = 42):
        self.batch_size = batch_size
        self.lengths = np.array(lengths)
        self.idx_sorted = np.argsort(self.lengths)  # 短→长
        self.base_seed = seed

    def __iter__(self):
        batches = [self.idx_sorted[i:i+self.batch_size] for i in range(0, len(self.idx_sorted), self.batch_size)]
        rng = random.Random(self.base_seed + int(time.time()) % 100000)
        rng.shuffle(batches)
        batches = [list(b) for b in batches]
        for b in batches:
            rng.shuffle(b)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)


# -----------------------
# AMP 兼容封装
# -----------------------
def _make_grad_scaler(enabled: bool):
    try:
        sig = inspect.signature(torch_amp.GradScaler)
        if 'device_type' in sig.parameters:
            return torch_amp.GradScaler(device_type="cuda", enabled=enabled)
        else:
            return torch_amp.GradScaler(enabled=enabled)
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(enabled=enabled)

class AutocastCUDA:
    def __init__(self, dtype):
        self.dtype = dtype
        self.ctx = None
    def __enter__(self):
        try:
            self.ctx = torch_amp.autocast(device_type="cuda", dtype=self.dtype)
        except TypeError:
            from torch.cuda.amp import autocast as cuda_autocast
            self.ctx = cuda_autocast(dtype=self.dtype)
        return self.ctx.__enter__()
    def __exit__(self, exc_type, exc, tb):
        return self.ctx.__exit__(exc_type, exc, tb)


# -----------------------
# 模型（微批上卡 + 池化 + 两层 MLP 头）
# -----------------------
class ChunkedClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2, agg: str = "mean",
                 micro_chunk: int = 8, use_checkpoint: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        if use_checkpoint and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden = self.encoder.config.hidden_size
        # 两层 MLP 头：768→512→2（GELU+Dropout）
        self.head = nn.Sequential(
            nn.Linear(hidden, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, num_labels),
        )
        assert agg in {"mean", "max"}
        self.agg = agg
        self.micro_chunk = micro_chunk

    def forward(self, input_ids, attention_mask, chunk_mask):
        """
        输入张量均在 CPU；仅微批切片搬上 GPU。
        """
        B, C, L = input_ids.size()
        device = next(self.parameters()).device
        H = self.encoder.config.hidden_size

        # CPU reshape
        x_ids = input_ids.view(B*C, L)
        x_att = attention_mask.view(B*C, L)
        flat_valid = chunk_mask.view(-1)

        # 输出累计（GPU）
        pooled_sum = torch.zeros(B, H, dtype=torch.float32, device=device)
        pooled_cnt = torch.zeros(B, 1, dtype=torch.float32, device=device)
        if self.agg == "max":
            pooled_max = torch.full((B, H), torch.finfo(torch.float32).min, dtype=torch.float32, device=device)

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        start = 0
        while start < B*C:
            end = min(start + self.micro_chunk, B*C)
            idx_range = torch.arange(start, end, device=device)
            sample_idx = torch.div(idx_range, C, rounding_mode='floor')

            valid_here = flat_valid[start:end]
            if valid_here.sum().item() == 0:
                start = end
                continue

            ids_mb  = x_ids[start:end].to(device, non_blocking=True)
            att_mb  = x_att[start:end].to(device, non_blocking=True)

            with AutocastCUDA(amp_dtype):
                out = self.encoder(input_ids=ids_mb, attention_mask=att_mb, return_dict=True)
                cls = out.last_hidden_state[:, 0, :]   # [m,H]

            mask_idx = torch.nonzero(valid_here.to(device), as_tuple=False).squeeze(-1)
            if mask_idx.numel() == 0:
                start = end
                continue

            cls_valid = cls.index_select(0, mask_idx).to(torch.float32)
            samp_valid = sample_idx.index_select(0, mask_idx)

            pooled_sum.index_add_(0, samp_valid, cls_valid)
            add_cnt = torch.ones((mask_idx.numel(), 1), dtype=torch.float32, device=device)
            pooled_cnt.index_add_(0, samp_valid, add_cnt)

            if self.agg == "max":
                for s in samp_valid.unique():
                    sel = (samp_valid == s).nonzero(as_tuple=False).squeeze(-1)
                    cur = cls_valid.index_select(0, sel)
                    pooled_max[s] = torch.maximum(pooled_max[s], cur.max(dim=0).values)

            del ids_mb, att_mb, out, cls, cls_valid, samp_valid, mask_idx
            start = end

        pooled = pooled_sum / pooled_cnt.clamp(min=1e-6) if self.agg == "mean" else pooled_max
        logits = self.head(pooled)  # 改为两层 MLP 头
        return logits


# -----------------------
# 指标
# -----------------------
def compute_scores(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return acc, f1, prec, rec


# -----------------------
# 主程序
# -----------------------
def main():
    print(f"[{now()}] 设备：{device}")
    print(f"[{now()}] DATASET_KIND = {DATASET_KIND}")
    df = load_dataset_any(dataset_path)

    code_col, label_col = pick_cols(df, DATASET_KIND)
    print(f"[{now()}] 使用代码列：{code_col} | 标签列：{label_col}")

    # 文本（提示 + 自动提示）
    codes = df[code_col].fillna("").astype(str)
    X = codes.map(build_input_with_prompt).values

    # 标签
    df["label"] = df[label_col].apply(map_label_merge if MERGE_OD_NOD else map_label_separate).astype(int)
    y = df["label"].values
    uniq = np.unique(y)
    print(f"[{now()}] 全量标签分布：", {int(v): int((y==v).sum()) for v in uniq})
    base_pos_rate = float((y==1).mean())
    print(f"[{now()}] 全量正例占比：{base_pos_rate:.4%}")

    if len(uniq) < 2:
        raise ValueError("数据仅单一类别，无法二分类。")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.model_max_length = int(1e9)
    print(f"[{now()}] 已加载 tokenizer，model_max_length={tokenizer.model_max_length}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    TN = FP = FN = TP = 0
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        fold_t0 = time.time()
        print("\n" + "=" * 90)
        print(f"[{now()}] ===== Fold {fold_idx}/{N_SPLITS} =====")

        X_train_full, X_test = X[train_index], X[test_index]
        y_train_full, y_test = y[train_index], y[test_index]
        print(f"[{now()}] 原始划分：train={len(X_train_full)}  test={len(X_test)}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=SEED
        )
        print(f"[{now()}] 划分后：train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

        if len(np.unique(y_train)) < 2:
            print(f"[{now()}] [WARN] 本折训练集只有一个类别，跳过该折。")
            continue

        # 过采样（可选）
        if OVERSAMPLE_METHOD == "smote_like":
            X_train_os, y_train_os = smote_like_oversample(
                X_train, y_train,
                target_pos_ratio=TARGET_POS_RATIO,
                svd_dim=SVD_DIM,
                k_neighbors=K_NEIGHBORS,
                random_state=SEED
            )
            print(f"[{now()}] SMOTE-like 后：train={len(X_train_os)} | pos={int((y_train_os==1).sum())} neg={int((y_train_os==0).sum())}")
            class_weights = np.array([1.0, 1.0], dtype=np.float32)  # 避免与过采样叠加
        elif OVERSAMPLE_METHOD == "ros":
            ros = RandomOverSampler(sampling_strategy=min(1.0, TARGET_POS_RATIO), random_state=SEED)
            X_train_os, y_train_os = ros.fit_resample(X_train.reshape(-1,1), y_train.reshape(-1,1))
            X_train_os = X_train_os.ravel(); y_train_os = y_train_os.ravel()
            print(f"[{now()}] ROS 后：train={len(X_train_os)} | pos={int((y_train_os==1).sum())} neg={int((y_train_os==0).sum())}")
            class_weights = np.array([1.0, 1.0], dtype=np.float32)
        else:
            X_train_os, y_train_os = X_train, y_train
            print(f"[{now()}] 不做过采样：train={len(X_train_os)} | pos={int((y_train_os==1).sum())} neg={int((y_train_os==0).sum())}")
            try:
                class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
            except Exception:
                class_weights = np.array([1.0, 1.0], dtype=np.float32)

        # 分块 & 截顶
        train_chunks = encode_chunks_per_texts(list(X_train_os), tokenizer, MAX_LEN, WINDOW_OVERLAP_TOKENS, MAX_CHUNKS_PER_SAMPLE)
        val_chunks   = encode_chunks_per_texts(list(X_val),      tokenizer, MAX_LEN, WINDOW_OVERLAP_TOKENS, MAX_CHUNKS_PER_SAMPLE)
        test_chunks  = encode_chunks_per_texts(list(X_test),     tokenizer, MAX_LEN, WINDOW_OVERLAP_TOKENS, MAX_CHUNKS_PER_SAMPLE)
        print(f"[{now()}] 块计数示例：train[0] chunks = {train_chunks[0]['input_ids'].size(0)}")

        # 数据集
        train_ds = OwnerChunkDataset(train_chunks, list(y_train_os))
        val_ds   = OwnerChunkDataset(val_chunks,   list(y_val))
        test_ds  = OwnerChunkDataset(test_chunks,  list(y_test))

        # 分桶采样器（每个 epoch 打乱）
        train_lengths = [d["input_ids"].size(0) for d in train_chunks]
        train_batch_sampler = BucketedBatchSampler(train_lengths, BATCH_TRAIN, seed=SEED)

        # DataLoader（CPU 常驻）
        g = torch.Generator(); g.manual_seed(SEED)
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler,
                                  generator=g, num_workers=0, collate_fn=collate_owner_batch)
        val_loader   = DataLoader(val_ds, sampler=SequentialSampler(val_ds),
                                  batch_size=BATCH_EVAL, generator=g, num_workers=0,
                                  collate_fn=collate_owner_batch)
        test_loader  = DataLoader(test_ds, sampler=SequentialSampler(test_ds),
                                  batch_size=BATCH_EVAL, generator=g, num_workers=0,
                                  collate_fn=collate_owner_batch)

        # 模型
        model = ChunkedClassifier(MODEL_NAME, num_labels=2, agg=AGG_METHOD,
                                  micro_chunk=MICRO_CHUNK, use_checkpoint=USE_CHECKPOINT).to(device)

        # 损失（类权重）
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"[{now()}] 类权重：{class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction="mean")

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        # AMP / TF32
        use_bf16_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        scaler = _make_grad_scaler(enabled=not use_bf16_amp)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ====== 学习率调度 ======
        updates_per_epoch = math.ceil(len(train_loader) / max(1, GRAD_ACC_STEPS))
        total_updates = max(1, updates_per_epoch * EPOCHS)
        warmup_steps = int(WARMUP_RATIO * total_updates)

        scheduler = None
        if USE_SCHEDULER:
            if SCHEDULER.lower().startswith("cos"):
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
                )
            else:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
                )

        best_val_f1 = -1.0
        best_state = None
        best_thr_for_fold = 0.5
        no_improve = 0

        # 训练
        for epoch in range(EPOCHS):
            ep_t0 = time.time()
            model.train()
            tr_loss = 0.0
            train_correct = 0
            train_total = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader, start=1):
                ids, attn, c_mask, labels = batch   # 全在 CPU
                labels = labels.to(device, non_blocking=True)

                with AutocastCUDA(torch.bfloat16 if use_bf16_amp else torch.float16):
                    logits = model(ids, attn, c_mask)
                    loss = criterion(logits, labels) / GRAD_ACC_STEPS

                if use_bf16_amp:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()

                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)

                if step % GRAD_ACC_STEPS == 0:
                    if use_bf16_amp:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    else:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    if scheduler is not None:
                        scheduler.step()                  # ← 仅在启用调度时推进
                    optimizer.zero_grad(set_to_none=True)

                tr_loss += loss.item() * GRAD_ACC_STEPS

                if step % 200 == 0:
                    print(f"[{now()}]   epoch {epoch+1} step {step}/{len(train_loader)} | "
                          f"avg_loss={tr_loss/max(1, step):.4f} | train_acc={train_correct/max(1,train_total):.4f}")

                del ids, attn, c_mask, labels, logits, loss

            tr_loss /= max(1, len(train_loader))
            train_acc = train_correct / max(1, train_total)

            # ===== 验证：阈值扫描（先粗后细）
            model.eval()
            y_val_true, y_val_prob = [], []
            with torch.no_grad():
                for batch in val_loader:
                    ids, attn, c_mask, labels = batch
                    labels = labels.to(device, non_blocking=True)
                    with AutocastCUDA(torch.bfloat16 if use_bf16_amp else torch.float16):
                        logits = model(ids, attn, c_mask)
                        probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
                    y_val_prob.extend(probs.tolist())
                    y_val_true.extend(labels.detach().cpu().numpy().tolist())
                    del ids, attn, c_mask, labels, logits

            y_val_true = np.array(y_val_true)
            y_val_prob = np.array(y_val_prob)

            # 粗扫
            cand_thresh = np.linspace(0.05, 0.95, 19)
            f1_list = []
            for t in cand_thresh:
                pred = (y_val_prob >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val_true, pred, labels=[0,1]).ravel()
                _, f1_t, _, _ = compute_scores(tn, fp, fn, tp)
                f1_list.append(f1_t)
            best_idx = int(np.argmax(f1_list))
            thr_coarse = float(cand_thresh[best_idx])

            # 细扫（±0.05，步长 0.01）
            lo = max(0.01, thr_coarse - 0.05)
            hi = min(0.99, thr_coarse + 0.05)
            cand_fine = np.linspace(lo, hi, int(round((hi-lo)/0.01))+1)
            f1_best = -1.0
            thr_best = thr_coarse
            for t in cand_fine:
                pred = (y_val_prob >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val_true, pred, labels=[0,1]).ravel()
                _, f1_t, _, _ = compute_scores(tn, fp, fn, tp)
                if f1_t > f1_best:
                    f1_best = float(f1_t); thr_best = float(t)

            THRESH_FOLD = thr_best
            y_val_pred = (y_val_prob >= THRESH_FOLD).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val_true, y_val_pred, labels=[0,1]).ravel()
            acc, f1, prec, rec = compute_scores(tn, fp, fn, tp)
            pos_rate_pred = float(y_val_pred.mean())

            elapse = time.time() - ep_t0
            print(f"[{now()}] Epoch {epoch+1}/{EPOCHS} | "
                  f"train_acc={train_acc:.4f} train_loss={tr_loss:.4f} | "
                  f"val_acc={acc:.4f} val_f1={f1:.4f} p={prec:.4f} r={rec:.4f} | "
                  f"best_thr={THRESH_FOLD:.2f} | pred_pos_rate={pos_rate_pred:.4%} | {elapse:.1f}s")

            # Early Stopping（看 val F1）
            if f1 > best_val_f1 + MIN_DELTA:
                print(f"[{now()}]   🎯 新最佳验证 F1：{best_val_f1:.4f} -> {f1:.4f}（保存权重）")
                best_val_f1 = f1
                best_thr_for_fold = THRESH_FOLD
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                if USE_EARLY_STOP:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        print(f"[{now()}]   ⛳ 提前停止：连续 {PATIENCE} 个 epoch 无明显提升（Δ<{MIN_DELTA}）")
                        break
                # 关闭提前停止时，不做任何事，继续训练

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 保存最佳
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), model_weights_path)
        print(f"[{now()}] 已保存本折最佳权重：{model_weights_path} | 使用阈值：{best_thr_for_fold:.2f}")

        # ===== 测试：沿用验证最优阈值
        model.eval()
        y_test_pred = []
        with torch.no_grad():
            for batch in test_loader:
                ids, attn, c_mask, labels = batch
                labels = labels.to(device, non_blocking=True)
                with AutocastCUDA(torch.bfloat16 if use_bf16_amp else torch.float16):
                    logits = model(ids, attn, c_mask)
                    probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
                y_test_pred.extend((probs >= best_thr_for_fold).astype(int).tolist())
                del ids, attn, c_mask, labels, logits

        print(f"[{now()}] 测试集报告（样本级，thr={best_thr_for_fold:.2f}）：")
        print(classification_report(y_test, y_test_pred, digits=4))
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0,1]).ravel()
        acc_t, f1_t, prec_t, rec_t = compute_scores(tn, fp, fn, tp)

        print(f"[{now()}] Fold {fold_idx} | test_acc={acc_t:.4f} test_f1={f1_t:.4f} "
              f"p={prec_t:.4f} r={rec_t:.4f} | CM: TN={tn} FP={fp} FN={fn} TP={tp}")

        TN += tn; FP += fp; FN += fn; TP += tp

        fold_time = time.time() - fold_t0
        print(f"[{now()}] 本折耗时：{fold_time / 60:.1f} 分钟")
        print("=" * 90)

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总
    acc, f1, prec, rec = compute_scores(TN, FP, FN, TP)
    out = pd.DataFrame([{
        "Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "PosRate_All": base_pos_rate,
        "DatasetKind": DATASET_KIND
    }])
    out.to_csv(results_file, index=False)
    print(f"[{now()}] OVERALL  acc={acc:.4f} f1={f1:.4f} p={prec:.4f} r={rec:.4f} | "
          f"CM: TN={TN} FP={FP} FN={FN} TP={TP}")
    print(f"[{now()}] 结果写入：{results_file}")
    print(f"[{now()}] Done.")

if __name__ == "__main__":
    main()
