# sft_train_ctl.py
import os, time, random
import pandas as pd
import numpy as np
import torch

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

# ---------------- PATHS (edit) ----------------
CSV_PATH  = r"C:\Users\najeh\OneDrive\Desktop\LLM(specification)\datasets\Cleaned_CTL_Dataset_nusmvOK.csv"
OUT_DIR   = r"C:\Users\najeh\PycharmProjects\T5NuSMVPPO\sft_out"
LOG_CSV   = os.path.join(OUT_DIR, "sft_training_log.csv")

os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
MAX_IN_LEN  = 256
MAX_OUT_LEN = 128

# Training knobs (CPU-friendly; increase later if you have GPU)
TRAIN_SAMPLES = None     # None = full
VAL_SAMPLES   = 800      # keep eval small on CPU to log frequently
TEST_SAMPLES  = 800

EVAL_STEPS    = 500
LOG_STEPS     = 100
SAVE_STEPS    = 500

# ---------------- utils ----------------
def read_csv_fallback(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return pd.read_csv(f)

def normalize_ctl(s: str) -> str:
    # light normalization (safe)
    if s is None:
        return ""
    s = " ".join(str(s).strip().split())
    return s.replace(" (", "(").replace("( ", "(").replace(" )", ")")

def balanced_parens(s: str) -> int:
    bal = 0
    for ch in s:
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0:
                return 0
    return 1 if bal == 0 else 0

def canonicalize_and_or(expr: str) -> str:
    # very small canonicalization: sort top-level & and | chunks if possible
    # (does NOT touch ->, U, temporal ops)
    e = normalize_ctl(expr)
    # try to sort only if the expression is a pure conjunction/disjunction at top-level
    # (best-effort; if parsing fails, return as-is)
    def split_top(s, op):
        parts, cur, depth = [], [], 0
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and s[i:i+len(op)] == op:
                parts.append("".join(cur).strip())
                cur = []
                i += len(op)
                continue
            cur.append(ch)
            i += 1
        parts.append("".join(cur).strip())
        return parts

    # detect top-level '&' or '|'
    if "&" in e:
        parts = split_top(e, "&")
        if len(parts) > 1 and all(p for p in parts):
            return " & ".join(sorted(parts))
    if "|" in e:
        parts = split_top(e, "|")
        if len(parts) > 1 and all(p for p in parts):
            return " | ".join(sorted(parts))
    return e

# ---------------- dataset ----------------
df = read_csv_fallback(CSV_PATH).rename(columns={"input":"prompt", "output":"gold"})
df = df[["prompt","gold"]].dropna().reset_index(drop=True)

# optional downsample for quick dev
if TRAIN_SAMPLES is not None and TRAIN_SAMPLES < len(df):
    df = df.sample(TRAIN_SAMPLES, random_state=SEED).reset_index(drop=True)

# split: train/val/test
rng = np.random.default_rng(SEED)
idx = np.arange(len(df))
rng.shuffle(idx)

n = len(df)
n_test = min(TEST_SAMPLES, max(1, int(0.05*n)))
n_val  = min(VAL_SAMPLES,  max(1, int(0.05*n)))
test_idx = idx[:n_test]
val_idx  = idx[n_test:n_test+n_val]
train_idx= idx[n_test+n_val:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)

print("Rows:", len(df), "| train:", len(df_train), "| val:", len(df_val), "| test:", len(df_test))

# tokenizer + model (slow tokenizer avoids protobuf issue)
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

class CTLDataset(torch.utils.data.Dataset):
    def __init__(self, df_):
        self.df = df_
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        x = str(self.df.loc[i,"prompt"])
        y = str(self.df.loc[i,"gold"])
        enc = tokenizer(
            x, truncation=True, max_length=MAX_IN_LEN
        )
        dec = tokenizer(
            y, truncation=True, max_length=MAX_OUT_LEN
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": dec["input_ids"],
        }
        return item

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # preds can be tuple; keep first
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # replace -100 in labels
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_n = [normalize_ctl(p) for p in pred_texts]
    gold_n = [normalize_ctl(g) for g in gold_texts]

    exact_norm = np.mean([p == g for p,g in zip(pred_n, gold_n)])

    pred_c = [canonicalize_and_or(p) for p in pred_n]
    gold_c = [canonicalize_and_or(g) for g in gold_n]
    canon_match = np.mean([p == g for p,g in zip(pred_c, gold_c)])

    paren_ok = np.mean([balanced_parens(p) for p in pred_n])

    return {
        "exact_norm": float(exact_norm),
        "canon_match": float(canon_match),
        "paren_ok": float(paren_ok),
    }

args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    num_train_epochs=1,                 # set to 3 later for paper run
    per_device_train_batch_size=4,      # CPU-friendly
    per_device_eval_batch_size=4,
    learning_rate=3e-4,
    predict_with_generate=True,
    generation_max_length=MAX_OUT_LEN,
    fp16=False,
    bf16=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=CTLDataset(df_train),
    eval_dataset=CTLDataset(df_val),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("DEVICE:", "cuda" if torch.cuda.is_available() else "cpu")
trainer.train()

# save final model
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# write log csv
log = pd.DataFrame(trainer.state.log_history)
log.to_csv(LOG_CSV, index=False, encoding="utf-8")
print("Wrote:", LOG_CSV)
print("Saved model to:", OUT_DIR)
