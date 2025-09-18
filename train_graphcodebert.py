# -*- coding: utf-8 -*-
import os, sys, math, gc, random, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# -----------------------
# 超参区（可按需修改）
# -----------------------
MODEL_NAME = r"D:\PYFlaky\models\graphcodebert-base"

# 强制 transformers 离线模式（可选但推荐）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_LEN    = 512
WINDOW_OVERLAP_TOKENS = 256   # ← 按你的建议：块大小=512（含[CLS]/[SEP]），步长=256的重叠
LR        = 2e-5
EPOCHS    = 5
BATCH_TRAIN = 4               # 注意：现在一个样本里包含多个chunk，显存占用更大，建议减小batch
BATCH_EVAL  = 8
SEED      = 42
N_SPLITS  = 10
AGG_METHOD = "mean"           # 块级池化："mean" | "max"
THRESH    = 0.5               # 概率阈值
MERGE_OD_NOD = True           # 目标：OD+NOD 合并为 Flaky=1

BASE_DIR = Path(r"D:\PYFlaky")
DATASET_PATH = BASE_DIR / "dataset" / "Python_dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAMP = time.strftime("%Y%m%d-%H%M%S")
MODEL_WEIGHTS_PATH = OUTPUT_DIR / f"graphcodebert_chunkpool_{STAMP}.bin"
RESULTS_FILE = OUTPUT_DIR / f"results_{STAMP}.csv"

dataset_path = DATASET_PATH
model_weights_path = MODEL_WEIGHTS_PATH
results_file = RESULTS_FILE

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
        import torch, sys, os
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
            "未检测到可用 GPU。请检查：\n"
            "1) 当前解释器是否安装了 GPU 版 torch（torch.version.cuda 应为 12.x）。\n"
            "2) PyCharm 的 Project Interpreter 是否指向正确环境。\n"
            "3) 是否设置了 CUDA_VISIBLE_DEVICES 导致隐藏了 GPU。\n"
            "4) 若网络受限，确认不是装成 CPU 轮子；必要时用离线 whl 安装 cu121/cu122 版本。\n"
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
            raise ImportError("需要 openpyxl：请在当前解释器里执行 `python -m pip install openpyxl`") from e
    else:
        raise ValueError(f"不支持的文件类型: {suf}（仅支持 .csv/.xlsx）")
    print(f"[{now()}] 数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
    return df

# -----------------------
# 标签映射
# -----------------------
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
# 分块编码：返回“每条样本”的若干块
# -----------------------
def encode_chunks_per_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_len: int = 512,
    overlap_tokens: int = 256,
) -> List[Dict[str, torch.Tensor]]:
    """对每条 text 产生若干 chunk：每个 chunk 是 [seq_len] 的 input_ids/attention_mask。
       返回列表长度 = 样本数；每个元素包含：
       {"input_ids": [num_chunks, max_len], "attention_mask": [num_chunks, max_len]}
    """
    out = []
    chunk_size = max_len - 2  # 预留 special tokens
    step = max(1, chunk_size - overlap_tokens)

    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        chunk_inputs = []
        if len(ids) == 0:
            # 空文本：构造仅含special tokens的一块
            piece_ids = tokenizer.build_inputs_with_special_tokens([])
            attn = [1] * len(piece_ids)
            if len(piece_ids) > max_len:
                piece_ids = piece_ids[:max_len]; attn = attn[:max_len]
            else:
                pad_len = max_len - len(piece_ids)
                piece_ids = piece_ids + [tokenizer.pad_token_id] * pad_len
                attn = attn + [0] * pad_len
            chunk_inputs.append((
                torch.tensor(piece_ids, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long)
            ))
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
                    piece_ids = piece_ids + [tokenizer.pad_token_id] * pad_len
                    attn = attn + [0] * pad_len
                chunk_inputs.append((
                    torch.tensor(piece_ids, dtype=torch.long),
                    torch.tensor(attn, dtype=torch.long)
                ))

        X_ids  = torch.stack([ci[0] for ci in chunk_inputs], dim=0)   # [num_chunks, max_len]
        X_mask = torch.stack([ci[1] for ci in chunk_inputs], dim=0)   # [num_chunks, max_len]
        out.append({"input_ids": X_ids, "attention_mask": X_mask})
    return out

# -----------------------
# 数据集（样本级，内部含多个块）
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
    """batch: list of (ids:[C_i,L], mask:[C_i,L], label)
       pad 到 (B, C_max, L)，并给出 chunk_mask: (B, C_max)
    """
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
# 模型：编码器 + 块维度池化 + 线性头
# -----------------------
class ChunkedClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2, agg: str = "mean"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        assert agg in {"mean", "max"}
        self.agg = agg

    def forward(self, input_ids, attention_mask, chunk_mask):
        """
        input_ids:   [B, C, L]
        attention_mask: [B, C, L]
        chunk_mask:  [B, C]  -> 哪些块有效
        """
        B, C, L = input_ids.size()
        x_ids = input_ids.view(B*C, L)
        x_att = attention_mask.view(B*C, L)

        outputs = self.encoder(input_ids=x_ids, attention_mask=x_att, return_dict=True)
        # 取每个块的 [CLS] 向量
        cls = outputs.last_hidden_state[:, 0, :]             # [B*C, H]
        H = cls.size(-1)
        cls = cls.view(B, C, H)                              # [B, C, H]

        # mask 无效块
        mask = chunk_mask.unsqueeze(-1).to(cls.dtype)        # [B, C, 1]

        if self.agg == "mean":
            summed = (cls * mask).sum(dim=1)                 # [B, H]
            denom = mask.sum(dim=1).clamp(min=1e-6)          # [B, 1]
            pooled = summed / denom
        else:  # max
            neg_inf = torch.finfo(cls.dtype).min
            cls_masked = cls.masked_fill(chunk_mask.unsqueeze(-1)==False, neg_inf)
            pooled, _ = torch.max(cls_masked, dim=1)         # [B, H]

        logits = self.classifier(pooled)                     # [B, num_labels]
        return logits

# -----------------------
# 评估指标
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
    df = load_dataset_any(dataset_path)

    # 选择代码列：优先 ExtractedCode，否则 final_code
    code_col = "ExtractedCode" if "ExtractedCode" in df.columns else ("final_code" if "final_code" in df.columns else None)
    if code_col is None:
        raise ValueError("未找到代码列（期望 ExtractedCode 或 final_code）")
    if "flaky" not in df.columns:
        raise ValueError("未找到标签列 flaky")
    print(f"[{now()}] 使用代码列：{code_col}")

    df[code_col] = df[code_col].fillna("").astype(str)

    # 标签映射
    df["label"] = df["flaky"].apply(map_label_merge if MERGE_OD_NOD else map_label_separate).astype(int)
    y = df["label"].values
    uniq = np.unique(y)
    print(f"[{now()}] 全量标签分布(初始)：", {int(v): int((y==v).sum()) for v in uniq})

    if MERGE_OD_NOD and len(uniq) < 2:
        print(f"[{now()}] [WARN] 没有 'not flaky' 样本，合并后只有一个类别。自动切换为 OD(1) vs NOD(0)。")
        df["label"] = df["flaky"].apply(map_label_separate).astype(int)
        y = df["label"].values
        uniq = np.unique(y)
        print(f"[{now()}] 全量标签分布(切换后)：", {int(v): int((y==v).sum()) for v in uniq})

    if len(uniq) < 2:
        raise ValueError("数据仍只有一个类别，无法进行二分类。请补充另一类样本后再试。")

    X = df[code_col].values

    # tokenizer（把 model_max_length 拉大，实际我们手动切到 MAX_LEN）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.model_max_length = int(1e9)
    print(f"[{now()}] 已加载本地 tokenizer，model_max_length={tokenizer.model_max_length}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    TN = FP = FN = TP = 0
    fold_idx = 0

    for train_index, test_index in skf.split(X, y):
        fold_t0 = time.time()
        print("\n" + "="*80)
        print(f"[{now()}] ===== Fold {fold_idx+1}/{N_SPLITS} =====")
        X_train_full, X_test = X[train_index], X[test_index]
        y_train_full, y_test = y[train_index], y[test_index]
        print(f"[{now()}] 原始划分：train={len(X_train_full)}  test={len(X_test)}")

        # 留出验证
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=SEED
        )
        print(f"[{now()}] 划分后：train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

        if len(np.unique(y_train)) < 2:
            print(f"[{now()}] [WARN] 本折训练集只有一个类别，跳过该折。")
            fold_idx += 1
            continue

        # 过采样（对“样本”级别，而非窗口级）
        ros = RandomOverSampler(sampling_strategy="minority", random_state=SEED)
        X_train_os, y_train_os = ros.fit_resample(X_train.reshape(-1,1), y_train.reshape(-1,1))
        X_train_os = X_train_os.ravel()
        y_train_os = y_train_os.ravel()
        print(f"[{now()}] 过采样后：train={len(X_train_os)}（正负均衡）")

        # === 编码为“每样本多个块” ===
        train_chunks = encode_chunks_per_texts(list(X_train_os), tokenizer, MAX_LEN, WINDOW_OVERLAP_TOKENS)
        val_chunks   = encode_chunks_per_texts(list(X_val),      tokenizer, MAX_LEN, WINDOW_OVERLAP_TOKENS)
        test_chunks  = encode_chunks_per_texts(list(X_test),     tokenizer, MAX_LEN, WINDOW_OVERLAP_TOKENS)
        print(f"[{now()}] 块计数（示例）：train[0] chunks = {train_chunks[0]['input_ids'].size(0)}")

        # 构建样本级 Dataset
        train_ds = OwnerChunkDataset(train_chunks, list(y_train_os))
        val_ds   = OwnerChunkDataset(val_chunks,   list(y_val))
        test_ds  = OwnerChunkDataset(test_chunks,  list(y_test))

        # DataLoader
        g = torch.Generator(); g.manual_seed(SEED)
        train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds),
                                  batch_size=BATCH_TRAIN, generator=g, num_workers=0,
                                  collate_fn=collate_owner_batch)
        val_loader   = DataLoader(val_ds, sampler=SequentialSampler(val_ds),
                                  batch_size=BATCH_EVAL, generator=g, num_workers=0,
                                  collate_fn=collate_owner_batch)
        test_loader  = DataLoader(test_ds, sampler=SequentialSampler(test_ds),
                                  batch_size=BATCH_EVAL, generator=g, num_workers=0,
                                  collate_fn=collate_owner_batch)

        # 模型 & 损失
        model = ChunkedClassifier(MODEL_NAME, num_labels=2, agg=AGG_METHOD).to(device)

        classes = np.array([0, 1])
        try:
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_os)
        except Exception as e:
            class_weights = np.array([1.0, 1.0], dtype=np.float32)
            print(f"[{now()}] [WARN] compute_class_weight 失败，使用均等权重。{e}")
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"[{now()}] 类权重：{class_weights.tolist()}")

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        best_val_f1 = -1.0
        best_state = None

        # 训练
        for epoch in range(EPOCHS):
            ep_t0 = time.time()
            model.train()
            tr_loss = 0.0
            train_correct = 0
            train_total = 0

            for step, batch in enumerate(train_loader, start=1):
                ids, attn, c_mask, labels = batch
                ids = ids.to(device); attn = attn.to(device)
                c_mask = c_mask.to(device); labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(ids, attn, c_mask)         # [B,2]
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()

                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)

                if step % 200 == 0:
                    print(f"[{now()}]   epoch {epoch+1} step {step}/{len(train_loader)} | "
                          f"avg_loss={tr_loss/step:.4f} | train_acc={train_correct/max(1,train_total):.4f}")

            tr_loss /= max(1, len(train_loader))
            train_acc = train_correct / max(1, train_total)

            # 验证（样本级）
            model.eval()
            y_val_pred = []
            with torch.no_grad():
                for batch in val_loader:
                    ids, attn, c_mask, labels = batch
                    ids = ids.to(device); attn = attn.to(device); c_mask = c_mask.to(device)
                    logits = model(ids, attn, c_mask)
                    probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
                    y_val_pred.extend((probs >= THRESH).astype(int).tolist())
            tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred, labels=[0,1]).ravel()
            acc, f1, prec, rec = compute_scores(tn, fp, fn, tp)
            elapse = time.time() - ep_t0
            print(f"[{now()}] Epoch {epoch+1}/{EPOCHS} | "
                  f"train_acc={train_acc:.4f} train_loss={tr_loss:.4f} | "
                  f"val_acc={acc:.4f} val_f1={f1:.4f} p={prec:.4f} r={rec:.4f} | {elapse:.1f}s")

            if f1 > best_val_f1:
                print(f"[{now()}]   🎯 新最佳验证 F1：{best_val_f1:.4f} -> {f1:.4f}（已暂存权重）")
                best_val_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 保存最佳
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), model_weights_path)
        print(f"[{now()}] 已保存本折最佳权重到：{model_weights_path}")

        # 测试（样本级）
        model.eval()
        y_test_pred = []
        with torch.no_grad():
            for batch in test_loader:
                ids, attn, c_mask, labels = batch
                ids = ids.to(device); attn = attn.to(device); c_mask = c_mask.to(device)
                logits = model(ids, attn, c_mask)
                probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
                y_test_pred.extend((probs >= THRESH).astype(int).tolist())

        print(f"[{now()}] 测试集报告（样本级）：")
        print(classification_report(y_test, y_test_pred, digits=4))
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0,1]).ravel()
        acc_t, f1_t, prec_t, rec_t = compute_scores(tn, fp, fn, tp)
        print(f"[{now()}] Fold {fold_idx+1}  test_acc={acc_t:.4f} test_f1={f1_t:.4f} p={prec_t:.4f} r={rec_t:.4f} | "
              f"CM: TN={tn} FP={fp} FN={fn} TP={tp}")
        TN += tn; FP += fp; FN += fn; TP += tp

        fold_time = time.time() - fold_t0
        print(f"[{now()}] 本折耗时：{fold_time/60:.1f} 分钟")
        print("="*80)

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总
    acc, f1, prec, rec = compute_scores(TN, FP, FN, TP)
    out = pd.DataFrame([{
        "Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP
    }])
    out.to_csv(results_file, index=False)
    print(f"[{now()}] OVERALL  acc={acc:.4f} f1={f1:.4f} p={prec:.4f} r={rec:.4f} | "
          f"CM: TN={TN} FP={FP} FN={FN} TP={TP}")
    print(f"[{now()}] 结果已写入：{results_file}")
    print(f"[{now()}] Done.")

if __name__ == "__main__":
    main()



