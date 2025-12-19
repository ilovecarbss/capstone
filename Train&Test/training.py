#!/usr/bin/env python
# coding: utf-8

# ### Cell 1

# In[2]:


import os
import torch

BASE_DIR = r"C:\Users\User\KS"
INPUT_LOGS      = os.path.join(BASE_DIR, "Dataset")
PROCESSED_OUT   = os.path.join(BASE_DIR, "Processed")
RESULTS_OUT     = os.path.join(BASE_DIR, "Results")
STARTER_FOLDER  = os.path.join(BASE_DIR, "loghub_starter")

for p in (PROCESSED_OUT, RESULTS_OUT, STARTER_FOLDER):
    os.makedirs(p, exist_ok=True)

assert os.path.isdir(INPUT_LOGS), f"INPUT_LOGS not found: {INPUT_LOGS}"

DEVICE = 0 if torch.cuda.is_available() else -1
print("CUDA available :", torch.cuda.is_available())
print("Selected device:", "GPU:0" if DEVICE == 0 else "CPU")

print(f"Input logs : {INPUT_LOGS}")
print(f"Processed  : {PROCESSED_OUT}")
print(f"Results    : {RESULTS_OUT}")
print(f"Starter    : {STARTER_FOLDER}")





# ### Cell 2A

# In[3]:


import pathlib

preprocess_file = pathlib.Path(STARTER_FOLDER) / "preprocess_logs.py"
code = r'''
import re, argparse, gzip
import pandas as pd
from pathlib import Path
from datetime import datetime

RE_IPV4 = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
RE_HEX  = re.compile(r'\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{8,}\b')
RE_UUID = re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b')
RE_PATH = re.compile(r'(/[^ \t\n\r\f\v:]+)+|([A-Za-z]:\\[^ \t\n\r\f\v]+)')
RE_NUM  = re.compile(r'(?<![A-Za-z])\d{3,}(?![A-Za-z])')
RE_PORT = re.compile(r'\bport\s+\d+\b', re.IGNORECASE)

TS_FORMATS = ["%Y-%m-%d %H:%M:%S,%f","%Y-%m-%d %H:%M:%S","%b %d %H:%M:%S"]

def try_parse_ts(prefix: str):
    prefix = prefix.strip()
    m = re.match(r'^(\d{6})\s+(\d{6})', prefix)
    if m:
        date, time_ = m.group(1), m.group(2)
        try:
            return datetime.strptime(date+time_, "%y%m%d%H%M%S").isoformat()
        except:
            pass
    for fmt in TS_FORMATS:
        try:
            dt = datetime.strptime(prefix, fmt)
            if fmt == "%b %d %H:%M:%S":
                dt = dt.replace(year=datetime.now().year)
            return dt.isoformat()
        except:
            pass
    return None

def mask_template(msg: str) -> str:
    s = msg
    s = RE_UUID.sub("<UUID>", s)
    s = RE_IPV4.sub("<IP>", s)
    s = RE_PORT.sub("port <NUM>", s)
    s = RE_PATH.sub("<PATH>", s)
    s = RE_HEX.sub("<HEX>", s)
    s = RE_NUM.sub("<NUM>", s)
    return s

def split_level_component(line: str):
    level = None
    component = None
    for lvl in ["TRACE","DEBUG","INFO","WARN","WARNING","ERROR","FATAL","CRITICAL"]:
        if re.search(rf'\b{lvl}\b', line):
            level = lvl
            break
    m = re.search(r'([A-Za-z_][\w\.-]{2,})(?:\[\d+\])?:', line) or re.search(r'\b([A-Za-z_][\w\.-]{2,})\b', line)
    if m:
        component = m.group(1)
    return level, component

def parse_line(line: str):
    ts = None
    for sep in ["] ", " - ", ": ", "  "]:
        if sep in line[:40]:
            ts = try_parse_ts(line.split(sep, 1)[0])
            break
    level, comp = split_level_component(line)
    return ts, level, comp, line.strip()

def process_file(path: Path):
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="ignore", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            ts, level, comp, msg = parse_line(line)
            if level == "WARNING": level = "WARN"
            if level == "CRITICAL": level = "FATAL"
            rows.append({
                "dataset": path.parent.name or path.stem,
                "line_no": i,
                "timestamp": ts,
                "level": level,
                "component": comp,
                "message": msg,
                "template": mask_template(msg)
            })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    in_dir, out_dir = Path(a.input), Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in sorted(in_dir.rglob("*.log")):
        print("[+] Processing", p.name)
        df = process_file(p)
        frames.append(df)
        df.to_csv(out_dir / f"{p.stem}.csv", index=False, encoding="utf-8")
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "ALL_combined.csv", index=False, encoding="utf-8")
        print("[OK] Wrote ALL_combined.csv")
    else:
        print("[WARN] No .log files found.")

if __name__ == "__main__":
    main()
'''
preprocess_file.write_text(code, encoding="utf-8")
print("Wrote:", preprocess_file)


# ### 2b

# In[4]:


import os, subprocess, pathlib

assert 'INPUT_LOGS' in globals() and 'PROCESSED_OUT' in globals() and 'STARTER_FOLDER' in globals(), "Run Cell 1 first."

preprocess_file = pathlib.Path(STARTER_FOLDER) / "preprocess_logs.py"

print("Logs folder exists:", os.path.isdir(INPUT_LOGS), "-", INPUT_LOGS)
print("Starter script exists:", preprocess_file.exists(), "-", preprocess_file)

os.makedirs(PROCESSED_OUT, exist_ok=True)
env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"

print("Preprocessing ...")
proc = subprocess.run(
    ["python", str(preprocess_file), "--input", INPUT_LOGS, "--output", PROCESSED_OUT],
    text=True, env=env
)
if proc.returncode != 0:
    print("Error running preprocessor.")
else:
    print("OK. CSVs written to:", PROCESSED_OUT)



# ### 3A

# In[5]:


import pathlib

evaluate_file = pathlib.Path(STARTER_FOLDER) / "evaluate_parsers.py"
code = r'''
import argparse
import pandas as pd
from pathlib import Path

def template_stats(df):
    c = df["template"].value_counts().reset_index()
    c.columns = ["template","count"]
    return c

def simple_anomalies(counts, z_thresh=3.0):
    mu = counts["count"].mean()
    sigma = counts["count"].std(ddof=0) or 1.0
    counts["z_score"] = (counts["count"] - mu) / sigma
    return counts[counts["z_score"] <= -z_thresh].sort_values("z_score")

def per_dataset_reports(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(in_dir.glob("*.csv")):
        if p.name == "ALL_combined.csv":
            continue
        df = pd.read_csv(p, dtype=str)
        if not df.empty and "dataset" in df.columns:
            ds = str(df["dataset"].iloc[0]) or p.stem
        else:
            ds = p.stem
        n = len(df)
        u = df["template"].nunique(dropna=False) if "template" in df else 0
        top = template_stats(df).head(20) if "template" in df else pd.DataFrame()
        rare = simple_anomalies(template_stats(df)) if "template" in df else pd.DataFrame()
        if not top.empty:
            top.to_csv(out_dir / f"top_templates_{ds}.csv", index=False, encoding="utf-8")
        if not rare.empty:
            rare.to_csv(out_dir / f"anomalies_{ds}.csv", index=False, encoding="utf-8")
        rows.append({"dataset": ds, "num_lines": n, "unique_templates": u,
                     "template_density": round((u / max(n,1)), 4)})
    pd.DataFrame(rows).sort_values("dataset").to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    per_dataset_reports(Path(a.input), Path(a.output))
    print("[OK] Wrote metrics to", a.output)

if __name__ == "__main__":
    main()
'''
evaluate_file.write_text(code, encoding="utf-8")
print("Wrote:", evaluate_file)


# ### 3B

# In[6]:


import os, subprocess, pathlib

env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"
evaluate_file = pathlib.Path(STARTER_FOLDER) / "evaluate_parsers.py"

print("Evaluating ...")
proc = subprocess.run(
    ["python", str(evaluate_file), "--input", PROCESSED_OUT, "--output", RESULTS_OUT],
    text=True, capture_output=True, env=env
)
print(proc.stdout)
if proc.returncode != 0:
    print("Error:\n", proc.stderr)
else:
    print("OK. Results written to:", RESULTS_OUT)



# ### 4

# In[7]:


import os, glob, pandas as pd
from IPython.display import display

summary_path = os.path.join(RESULTS_OUT, "summary_metrics.csv")
if os.path.exists(summary_path):
    print("Summary metrics (first 20 rows):")
    display(pd.read_csv(summary_path).head(20))
else:
    print("summary_metrics.csv not found. Run Cell 3B first.")

tops = sorted(glob.glob(os.path.join(RESULTS_OUT, "top_templates_*.csv")))
if tops:
    print("\nSample top-templates file:", os.path.basename(tops[0]))
    display(pd.read_csv(tops[0]).head(10))
else:
    print("No top_templates_*.csv produced yet.")





# ### 5 - causes kernel shutdown 

# ### Cell 6

# In[8]:


import os
LABELS_OUT = os.path.join(BASE_DIR, "Labels")
MODEL_OUT  = os.path.join(BASE_DIR, "Models")
os.makedirs(LABELS_OUT, exist_ok=True)
os.makedirs(MODEL_OUT,  exist_ok=True)

TARGET_DATASET = "ALL_combined"

print("Labels dir:", LABELS_OUT)
print("Models dir:", MODEL_OUT)



# ### Cell 7

# In[9]:


import os, json, pandas as pd

csv_path = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")
assert os.path.exists(csv_path), f"Missing {csv_path}"
df = pd.read_csv(csv_path, dtype=str).fillna("")

records = [
    {
        "dataset": TARGET_DATASET,
        "message": r.get("message", ""),
        "level": r.get("level", ""),
        "component": r.get("component", "")
    }
    for _, r in df.iterrows()
]

jsonl_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_to_label.jsonl")
with open(jsonl_path, "w", encoding="utf-8", newline="\n") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Wrote:", jsonl_path, "| rows:", len(records))




# ### Cell 9

# In[10]:


import re

ALLOWED_TAGS = {"<IP>","<NUM>","<UUID>","<PATH>","<HEX>","<DATE>","<TIME>","<MS>","<HOST>"}

def valid_template(t):
    bad = re.findall(r'<[^>]+>', t)
    return all(tag in ALLOWED_TAGS for tag in bad)

def repair_template(raw):
    t = raw.strip()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', t)
    t = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}', '<UUID>', t)
    t = re.sub(r'(?<![A-Za-z])\d{3,}(?![A-Za-z])', '<NUM>', t)
    t = t.replace("`","").replace('"','').replace("'", "")
    return t

def correct_once(t):
    r = repair_template(t)
    if valid_template(r):
        return r
    return re.sub(r'<[^>]+>', '<NUM>', r)



# ### Cell 10

# In[11]:


import os, json, re, pandas as pd
from tqdm import tqdm

jsonl_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_to_label.jsonl")
assert os.path.exists(jsonl_path), "Run Cell 7 first."

def quick_mask(s):
    s = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}', '<UUID>', s)
    s = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', s)
    s = re.sub(r'(?<![A-Za-z])\d{3,}(?![A-Za-z])', '<NUM>', s)
    return s

labeled = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Masking + correcting"):
        r = json.loads(line)
        raw_template = quick_mask(r["message"])
        cleaned = correct_once(raw_template)
        labeled.append({
            "dataset": r["dataset"],
            "message": r["message"],
            "level": r["level"],
            "component": r["component"],
            "template": cleaned
        })

labels_csv = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels.csv")
pd.DataFrame(labeled).to_csv(labels_csv, index=False, encoding="utf-8")
print("Wrote labels:", labels_csv, "| rows:", len(labeled))


# ### Cell 11 - remove 

# In[12]:


import sys, subprocess

def pipi(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

pipi([
    "transformers==4.44.2",
    "datasets==3.0.1",
    "accelerate==0.34.2",
    "evaluate==0.4.2",
    "sentencepiece==0.2.0",
    "scikit-learn==1.5.2",
    "fsspec[http]==2024.6.1",
    "tqdm",
    "drain3==0.9.11"
])



# ### Cell 11A

# In[13]:


import os, pandas as pd
from sklearn.model_selection import train_test_split

labels_csv = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels.csv")
df = pd.read_csv(labels_csv).fillna("")
df = df[(df["message"].str.len()>0) & (df["template"].str.len()>0)].copy()

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)
print("Train:", len(train_df), "Val:", len(val_df))

train_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_train.csv")
val_path   = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_val.csv")
train_df.to_csv(train_path, index=False, encoding="utf-8")
val_df.to_csv(val_path,   index=False, encoding="utf-8")
print("Saved:", train_path, "|", val_path)


# ### Cell 11B

# In[14]:


import os, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

train_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_train.csv")
val_path   = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_val.csv")

train_df = pd.read_csv(train_path).fillna("")
val_df   = pd.read_csv(val_path).fillna("")

model_name = "t5-base"  #  base model
SPECIALS = ["<IP>","<NUM>","<UUID>","<PATH>","<HEX>","<DATE>","<TIME>","<MS>","<HOST>"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})

def to_hfds(frame):
    return Dataset.from_pandas(frame[["message","template"]].rename(columns={"template":"labels"}), preserve_index=False)

train_ds = to_hfds(train_df)
val_ds   = to_hfds(val_df)

MAX_IN, MAX_OUT = 256, 128

def preprocess_batch(batch):
    ins  = tokenizer(batch["message"], max_length=MAX_IN, truncation=True)
    outs = tokenizer(text_target=batch["labels"], max_length=MAX_OUT, truncation=True)
    ins["labels"] = outs["input_ids"]
    return ins

train_tok = train_ds.map(preprocess_batch, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(preprocess_batch,   batched=True, remove_columns=val_ds.column_names)

print("Tokenized:", len(train_tok), "train /", len(val_tok), "val")


# ### Cell 12

# In[24]:


import os, torch
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainingArguments,
)

save_dir = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_base")# rename to largeg 

model = AutoModelForSeq2SeqLM.from_pretrained("t5-Large")
model.resize_token_embeddings(len(tokenizer)) 

fp16_flag = torch.cuda.is_available()

args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    logging_first_step=True,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=fp16_flag,
    report_to=[],                  
    seed=42,
    dataloader_num_workers=0,      
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    disable_tqdm=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

train_result = trainer.train()
metrics = trainer.evaluate()
print(metrics)
print("Model saved to:", save_dir)


# ### Cell 12B

# In[25]:


import os, json, pandas as pd, torch
from transformers import pipeline
from tqdm import tqdm

MODEL_ROOT = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_base") #rename to match 
assert os.path.isdir(MODEL_ROOT), f"Model dir not found: {MODEL_ROOT}"

def pick_model_dir(root):
    cps = [d for d in os.listdir(root) if d.startswith("checkpoint-")]
    if not cps:
        return root
    cps.sort(key=lambda s: int(s.split("-")[-1]))
    return os.path.join(root, cps[-1])

FINAL_MODEL_DIR = pick_model_dir(MODEL_ROOT)
print("Using model:", FINAL_MODEL_DIR)

device = 0 if torch.cuda.is_available() else -1
print("CUDA available:", torch.cuda.is_available(), "| device:", ("GPU:0" if device == 0 else "CPU"))
gen = pipeline("text2text-generation", model=FINAL_MODEL_DIR, tokenizer=FINAL_MODEL_DIR, device=device)

csv_in = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")
assert os.path.exists(csv_in), f"Missing {csv_in}"
raw_df = pd.read_csv(csv_in, dtype=str).fillna("")
assert "message" in raw_df.columns
raw_df = raw_df[raw_df["message"].str.len() > 0].reset_index(drop=True)

BATCH = 64
MAX_NEW = 64

def infer_templates(messages):
    outs = gen(list(messages), max_new_tokens=MAX_NEW, do_sample=False, num_beams=4)
    return [o.get("generated_text", "").strip() for o in outs]

rows = []
for start in tqdm(range(0, len(raw_df), BATCH), desc="Generating templates"):
    batch = raw_df.iloc[start:start+BATCH]
    preds = infer_templates(batch["message"])
    for pred, (_, r) in zip(preds, batch.iterrows()):
        rows.append({
            "timestamp": r.get("timestamp", None),
            "level":     r.get("level", None),
            "source":    (r.get("component", "") or TARGET_DATASET),
            "template":  pred,
            "raw":       r.get("message", "")
        })

parsed_df = pd.DataFrame(rows)
os.makedirs(RESULTS_OUT, exist_ok=True)
parsed_csv  = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_t5.csv")
parsed_json = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_t5.jsonl")

parsed_df.to_csv(parsed_csv, index=False, encoding="utf-8")
with open(parsed_json, "w", encoding="utf-8", newline="\n") as f:
    for rec in rows:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Wrote:", parsed_csv)
print("Wrote:", parsed_json)
print("Rows:", len(parsed_df))






# ### Cell 13 - Drain3 clustering

# In[26]:


import os, json, pandas as pd
try:
    
    from drain3.template_miner_config import TemplateMinerConfig
except Exception:
    
    from drain3.template_miner import TemplateMinerConfig
from drain3 import TemplateMiner

parsed_csv = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_t5.csv")
df = pd.read_csv(parsed_csv, dtype=str).fillna("")
assert "template" in df.columns, "Column 'template' missing in parsed_t5.csv"
df = df[df["template"].str.len() > 0].reset_index(drop=True)


config = TemplateMinerConfig()
miner = TemplateMiner(config=config)


miner.drain.sim_th = 0.40
miner.drain.depth = 4
miner.drain.max_children = 100

cluster_ids, scores = [], []
for t in df["template"]:
    res = miner.add_log_message(t if isinstance(t, str) else "")
    if isinstance(res, dict):
        cid = res.get("cluster_id")
        sim = res.get("similarity", 1.0)
    else:
        cid, sim = None, 1.0
    cluster_ids.append(f"T{int(cid):08d}" if cid is not None else None)
    scores.append(sim)

df["template_id"] = cluster_ids
df["fuzzy_score"] = scores

def _dump_state(miner_obj):
    clusters = []
    for c in getattr(miner_obj.drain, "clusters", []):
       
        tmpl = None
        if hasattr(c, "get_template"):
            try:
                tmpl = c.get_template()
            except Exception:
                pass
        if tmpl is None:
            tmpl = getattr(c, "template", None)
        clusters.append({
            "cluster_id": int(getattr(c, "cluster_id", -1)),
            "template": tmpl,
            "size": int(getattr(c, "size", 0))
        })
    return json.dumps({"clusters": clusters}, ensure_ascii=False)

state_path = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_drain3_state.json")
payload = _dump_state(miner)
with open(state_path, "w", encoding="utf-8") as f:
    f.write(payload)


final_csv  = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_final.csv")
final_json = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_final.jsonl")
df.to_csv(final_csv, index=False, encoding="utf-8")
with open(final_json, "w", encoding="utf-8", newline="\n") as f:
    for rec in df.to_dict(orient="records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Saved Drain3 state:", state_path, f"(clusters={len(getattr(miner.drain, 'clusters', []))})")
print("Wrote:", final_csv)
print("Wrote:", final_json)
print("Rows:", len(df))


# ### Cell 14 - Save Model

# In[29]:


import os, torch
from transformers import pipeline

BASE_DIR      = r"C:\Users\User\KS"
MODEL_OUT     = os.path.join(BASE_DIR, "Models")
RESULTS_OUT   = os.path.join(BASE_DIR, "Results")
TARGET_DATASET = "ALL_combined"

MODEL_ROOT = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_base")
assert os.path.isdir(MODEL_ROOT), f"Model dir not found: {MODEL_ROOT}"

def pick_model_dir(root):
    cps = [d for d in os.listdir(root) if d.startswith("checkpoint-")]
    if not cps:
        return root
    cps.sort(key=lambda s: int(s.split("-")[-1]))
    return os.path.join(root, cps[-1])

FINAL_MODEL_DIR = pick_model_dir(MODEL_ROOT)
print("Using T5 model from:", FINAL_MODEL_DIR)

device = 0 if torch.cuda.is_available() else -1
print("CUDA available:", torch.cuda.is_available(), "| device:", "GPU:0" if device == 0 else "CPU")

t5_pipe = pipeline(
    "text2text-generation",
    model=FINAL_MODEL_DIR,
    tokenizer=FINAL_MODEL_DIR,
    device=device
)




# In[30]:


import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

parsed_t5_csv = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_t5.csv")
assert os.path.exists(parsed_t5_csv), f"Missing {parsed_t5_csv}"

t5_df = pd.read_csv(parsed_t5_csv, dtype=str).fillna("")
assert "template" in t5_df.columns, "Column 'template' missing in parsed_t5.csv"

print("Loaded T5 parsed templates:", len(t5_df))

config = TemplateMinerConfig()
drain_miner = TemplateMiner(config=config)

drain_miner.drain.sim_th = 0.40
drain_miner.drain.depth = 4
drain_miner.drain.max_children = 100

for tmpl in t5_df["template"]:
    _ = drain_miner.add_log_message(str(tmpl))

print("Drain3 ready. Num clusters:", len(drain_miner.drain.clusters))


# In[37]:


import os
import pickle
import json

from transformers import pipeline
import torch

BASE_DIR      = r"C:\Users\User\KS"
MODEL_OUT     = os.path.join(BASE_DIR, "Models")
RESULTS_OUT   = os.path.join(BASE_DIR, "Results")
TARGET_DATASET = "ALL_combined"


MODEL_ROOT = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_base")
assert os.path.isdir(MODEL_ROOT), f"Trained T5 model not found: {MODEL_ROOT}"

def pick_model_dir(root):
    cps = [d for d in os.listdir(root) if d.startswith("checkpoint-")]
    if not cps:
        return root
    cps.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(root, cps[-1])

FINAL_MODEL_DIR = pick_model_dir(MODEL_ROOT)
print("Using trained T5 model:", FINAL_MODEL_DIR)


drain_state_path = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_drain3_state.json")
assert os.path.exists(drain_state_path), f"Drain3 state not found: {drain_state_path}"

with open(drain_state_path, "r", encoding="utf-8") as f:
    drain_state = json.load(f)

print("Loaded Drain3 state with clusters:", len(drain_state.get("clusters", [])))


combined_payload = {
    "t5_model_path": FINAL_MODEL_DIR,
    "drain_state": drain_state,
}

save_path = os.path.join(BASE_DIR, "combined_model.pkl")
with open(save_path, "wb") as f:
    pickle.dump(combined_payload, f)

print("\n Combined model saved to:", save_path)

