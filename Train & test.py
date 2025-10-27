#!/usr/bin/env python
# coding: utf-8

# ### REQUREMENTS

# In[ ]:


#%pip install -r "C:\Users\harik\KS\requirements.txt"





# ### ENV

# ### Cell 1

# In[5]:


import os
import torch

BASE_DIR = r"C:\Users\harik\KS"
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

# In[7]:


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

# In[8]:


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

# In[9]:


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

# In[10]:


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

# In[11]:


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

# In[12]:


import os, pandas as pd, matplotlib.pyplot as plt

summary_path = os.path.join(RESULTS_OUT, "summary_metrics.csv")
if not os.path.exists(summary_path):
    print("summary_metrics.csv not found.")
else:
    df = pd.read_csv(summary_path).sort_values("template_density", ascending=False)
    plt.figure(figsize=(8,5))
    plt.bar(df["dataset"], df["template_density"], color="skyblue", edgecolor="black")
    plt.xticks(rotation=60, ha="right")
    plt.title("Template Density by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Template Density")
    plt.tight_layout()
    plt.show()



# ### Cell 6

# In[16]:


import os
LABELS_OUT = os.path.join(BASE_DIR, "Labels")
MODEL_OUT  = os.path.join(BASE_DIR, "Models")
os.makedirs(LABELS_OUT, exist_ok=True)
os.makedirs(MODEL_OUT,  exist_ok=True)

TARGET_DATASET = "OpenSSH_2k"

print("Labels dir:", LABELS_OUT)
print("Models dir:", MODEL_OUT)



# ### Cell 7

# In[17]:


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




# ### Cell 8 

# In[15]:


FEW_SHOTS = [
    ("Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; rhost=218.188.2.4",
     "<MON> <DAY> <TIME> <HOST> sshd(pam_unix)[<NUM>]: authentication failure; rhost=<IP>"),
    ("2015-10-18 18:01:51,650 INFO org.apache.hadoop.mapreduce Job Transitioned from NEW to INITED",
     "<DATE> <TIME>,<MS> INFO org.apache.hadoop.mapreduce Job Transitioned from NEW to INITED"),
    ("2017-05-16 00:00:10.303 2931 INFO nova.compute.manager [instance: b9000564-....] Took 19.05 seconds to spawn the instance on the hypervisor.",
     "<DATE> <TIME> <NUM> INFO nova.compute.manager [instance: <UUID>] Took <NUM> seconds to spawn the instance on the hypervisor.")
]

SYSTEM_INSTRUCTIONS = (
    "Task: Convert a raw log line into a stable template. "
    "Keep constant words, replace variables with tags: <IP> <NUM> <UUID> <PATH> <HEX> <DATE> <TIME> <MS> <HOST>. "
    "Output only the template (one line)."
)

def build_prompt(message, level, component):
    demos = "\n\n".join([f"Input: {m}\nTemplate: {t}" for m, t in FEW_SHOTS])
    return f"{SYSTEM_INSTRUCTIONS}\n\n{demos}\n\nInput: {message}\nTemplate:"



# ### Cell 9

# In[18]:


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

# In[19]:


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


# ### Cell 11

# In[20]:


import sys, subprocess

def pipi(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

pipi(["transformers==4.44.2",
      "datasets==3.0.1",
      "accelerate==0.34.2",
      "evaluate==0.4.2",
      "sentencepiece==0.2.0",
      "scikit-learn==1.5.2",
      "fsspec[http]==2024.6.1",
      "tqdm"])




# ### Cell 12 – split train/val

# In[21]:


import os, pandas as pd
from sklearn.model_selection import train_test_split

teacher_csv = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels_teacher.csv")   # NEW
llm_csv     = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels_llm.csv")
regex_csv   = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels.csv")

labels_csv = teacher_csv if os.path.exists(teacher_csv) else (
    llm_csv if os.path.exists(llm_csv) else regex_csv
)
print("Using labels:", labels_csv)

df = pd.read_csv(labels_csv).fillna("")
df = df[(df["message"].str.len() > 0) & (df["template"].str.len() > 0)].copy()

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
print("Train:", len(train_df), "Val:", len(val_df))
train_df.head(3)




# ### Cell 13 - tokenize

# In[22]:


from datasets import Dataset
from transformers import AutoTokenizer

model_name = "t5-small"
SPECIALS = ["<IP>","<NUM>","<UUID>","<PATH>","<HEX>","<DATE>","<TIME>","<MS>","<HOST>"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})

def to_ds(frame):
    return Dataset.from_pandas(
        frame[["message", "template"]].rename(columns={"template": "labels"}),
        preserve_index=False
    )

train_ds = to_ds(train_df)
val_ds = to_ds(val_df)

MAX_IN, MAX_OUT = 256, 128

def preprocess_batch(batch):
    ins = tokenizer(batch["message"], max_length=MAX_IN, truncation=True)
    outs = tokenizer(text_target=batch["labels"], max_length=MAX_OUT, truncation=True)
    ins["labels"] = outs["input_ids"]
    return ins

train_tok = train_ds.map(preprocess_batch, batched=True, remove_columns=train_ds.column_names)
val_tok = val_ds.map(preprocess_batch, batched=True, remove_columns=val_ds.column_names)

print("Tokenized:", len(train_tok), "train /", len(val_tok), "val")




# ### Cell 14 - run to here 

# In[23]:


import os
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainingArguments,
)

save_dir = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
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
metrics




# ### Cell 15- Save fine-tuned model/tokenizer

# In[24]:


import os, json, time
from shutil import rmtree
from pathlib import Path

FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")
HF_DIR          = os.path.join(FINAL_MODEL_DIR, "hf_model")
TOK_DIR         = os.path.join(FINAL_MODEL_DIR, "tokenizer")


for p in (HF_DIR, TOK_DIR):
    if os.path.isdir(p): rmtree(p, ignore_errors=True)


tokenizer.save_pretrained(TOK_DIR)
model.save_pretrained(HF_DIR)


schema = {
    "required_keys": ["timestamp","level","source","template_id","template","vars","fuzzy_score","raw"]
}
build_info = {
    "model_version": "v1.0",
    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "target_dataset": TARGET_DATASET,
    "hf_model": "t5-small",
    "save_dir": save_dir,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "notes": "T5 seq2seq fine-tuned on full dataset (message->template)."
}

os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
with open(os.path.join(FINAL_MODEL_DIR, "schema.json"), "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
with open(os.path.join(FINAL_MODEL_DIR, "build_info.json"), "w", encoding="utf-8") as f:
    json.dump(build_info, f, indent=2)

print("Final model saved at:", FINAL_MODEL_DIR)



# ### Cell 16 GPU inference over full dataset to produce structured JSONL/CSV

# In[27]:


import os, json, pandas as pd
from transformers import pipeline
from tqdm import tqdm

csv_in = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")
assert os.path.exists(csv_in), f"Missing {csv_in}"
raw_df = pd.read_csv(csv_in, dtype=str).fillna("")
assert "message" in raw_df.columns, "Expected 'message' column in processed CSV."

gen = pipeline(
    "text2text-generation",
    model=os.path.join(FINAL_MODEL_DIR, "hf_model"),
    tokenizer=os.path.join(FINAL_MODEL_DIR, "tokenizer"),
    device=DEVICE,
    do_sample=False,
    num_beams=4,
    max_new_tokens=64,
)

BATCH = 32

def _extract_generated_text(obj):
    
    if isinstance(obj, dict) and "generated_text" in obj:
        return obj["generated_text"]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "generated_text" in obj[0]:
        return obj[0]["generated_text"]
    # Fallback: stringify
    return str(obj)

def infer_templates(msgs):
    outs = gen(list(msgs))
    return [_extract_generated_text(o).strip() for o in outs]

rows = []
for start in tqdm(range(0, len(raw_df), BATCH), desc="Generating templates"):
    batch = raw_df.iloc[start:start+BATCH]
    preds = infer_templates(batch["message"])
    for pred, (_, r) in zip(preds, batch.iterrows()):
        rows.append({
            "timestamp": r.get("timestamp", None),
            "level": r.get("level", None),
            "source": r.get("component", "") or TARGET_DATASET,
            "template_id": None,          
            "template": pred,
            "vars": {},
            "fuzzy_score": None,          
            "raw": r.get("message", "")
        })

parsed_df  = pd.DataFrame(rows)
parsed_csv  = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_t5.csv")
parsed_json = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_t5.jsonl")

os.makedirs(RESULTS_OUT, exist_ok=True)
parsed_df.to_csv(parsed_csv, index=False, encoding="utf-8")
with open(parsed_json, "w", encoding="utf-8", newline="\n") as f:
    for rec in rows:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Wrote:", parsed_csv)
print("Wrote:", parsed_json)




# ### Cell 17 AdaParser (self-learning) integration with safe fallback

# In[28]:


import os, json, pandas as pd, hashlib

parsed_csv  = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_t5.csv")
assert os.path.exists(parsed_csv), "Run Cell 16 first."
df = pd.read_csv(parsed_csv).fillna("")

adaparser_state = os.path.join(FINAL_MODEL_DIR, "adaparser_state.bin")
templates_csv   = os.path.join(FINAL_MODEL_DIR, "templates.csv")


try:
    from lilac import AdaParser, ParserConfig  
    cfg = ParserConfig(model="t5-small", device=DEVICE)
    ada = AdaParser(cfg)

   
    ada.update(df["template"].tolist())

  
    group_ids, scores = ada.assign(df["template"].tolist())
    df["template_id"] = group_ids
    df["fuzzy_score"] = scores

  
    ada.save(adaparser_state)
    pd.DataFrame({"template_id": group_ids, "template": df["template"]}).drop_duplicates().to_csv(templates_csv, index=False)
    print("AdaParser integrated and saved.")
except Exception as e:
    print("AdaParser not available or API mismatch; using fallback grouping.", e)
 
    def tid(s): return "T" + hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    df["template_id"] = df["template"].apply(tid)
    df["fuzzy_score"] = 1.0
    pd.DataFrame({"template_id": df["template_id"], "template": df["template"]}).drop_duplicates().to_csv(templates_csv, index=False)
 
    with open(adaparser_state, "wb") as f: f.write(b"fallback")


parsed_with_ids_csv  = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_final.csv")
parsed_with_ids_json = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_final.jsonl")
df.to_csv(parsed_with_ids_csv, index=False, encoding="utf-8")
with open(parsed_with_ids_json, "w", encoding="utf-8", newline="\n") as f:
    for rec in df.to_dict(orient="records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Wrote:", parsed_with_ids_csv)
print("Wrote:", parsed_with_ids_json)


# ### Cell 18 Add minimal LILAC config and finalize

# In[29]:


import os, json

lilac_state = {
    "tags": ["<IP>","<NUM>","<UUID>","<PATH>","<HEX>","<DATE>","<TIME>","<MS>","<HOST>"],
    "normalizers": ["mask UUID", "mask IPv4", "mask NUM >=3 digits", "mask PATH", "mask HEX"],
    "notes": "LILAC-style deterministic normalization applied upstream."
}
with open(os.path.join(FINAL_MODEL_DIR, "lilac_state.json"), "w", encoding="utf-8") as f:
    json.dump(lilac_state, f, indent=2)


print("Final model folder contains:")
for root, _, files in os.walk(FINAL_MODEL_DIR):
    for name in files:
        print("-", os.path.relpath(os.path.join(root, name), FINAL_MODEL_DIR))


# ### Cell 19 stopped at 18 

# In[3]:


import sys, subprocess
def pipi(pkgs): subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
try:
    import rapidfuzz  
except Exception:
    pipi(["rapidfuzz==3.10.0"])


# In[ ]:


import os, glob, json, pandas as pd
from transformers import pipeline
from tqdm import tqdm

gen = pipeline(
    "text2text-generation",
    model=os.path.join(FINAL_MODEL_DIR, "hf_model"),
    tokenizer=os.path.join(FINAL_MODEL_DIR, "tokenizer"),
    device=DEVICE,
    do_sample=False,
    num_beams=1,         
    max_new_tokens=64,
)

def _extract_generated_text(obj):
    if isinstance(obj, dict) and "generated_text" in obj:
        return obj["generated_text"]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "generated_text" in obj[0]:
        return obj[0]["generated_text"]
    return str(obj)

def infer_batch(msgs):
    outs = gen(list(msgs))
    return [_extract_generated_text(o).strip() for o in outs]

BATCH = 32  

all_parsed_paths = []
csv_files = sorted(glob.glob(os.path.join(PROCESSED_OUT, "*.csv")))
for csv_in in csv_files:
    ds_name = os.path.splitext(os.path.basename(csv_in))[0]
    if ds_name == "ALL_combined":  # skip helper
        continue
    raw_df = pd.read_csv(csv_in, dtype=str).fillna("")
    if "message" not in raw_df.columns:
        print("Skip (no 'message' column):", ds_name)
        continue

    rows = []
    for start in tqdm(range(0, len(raw_df), BATCH), desc=f"Infer {ds_name}"):
        batch = raw_df.iloc[start:start+BATCH]
        preds = infer_batch(batch["message"])
        for pred, (_, r) in zip(preds, batch.iterrows()):
            rows.append({
                "dataset": ds_name,
                "timestamp": r.get("timestamp"),
                "level": r.get("level"),
                "source": r.get("component") or ds_name,
                "template": pred,
                "raw": r.get("message")
            })
    out_csv  = os.path.join(RESULTS_OUT, f"{ds_name}_parsed_t5.csv")
    out_json = os.path.join(RESULTS_OUT, f"{ds_name}_parsed_t5.jsonl")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    all_parsed_paths.append(out_csv)
    print("Wrote:", out_csv)


# ## Graphs

# In[30]:


import os, pandas as pd, numpy as np, difflib
from pathlib import Path


labels_regex = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels.csv")
labels_llm   = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels_llm.csv")  
pred_csv     = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_parsed_final.csv")  

labels_path = labels_llm if os.path.exists(labels_llm) else labels_regex
assert os.path.exists(labels_path), f"Missing labels file at {labels_path}"
assert os.path.exists(pred_csv), f"Missing predictions file at {pred_csv}"

gold = pd.read_csv(labels_path, dtype=str).fillna("")
pred = pd.read_csv(pred_csv, dtype=str).fillna("")


join_key_gold = "message"
join_key_pred = "raw" if "raw" in pred.columns else "message"

df_eval = gold.merge(
    pred.rename(columns={join_key_pred: "message", "template":"pred_template"}),
    on="message",
    how="inner",
    suffixes=("_gold", "_pred")
)


cols = ["message", "template", "pred_template", "level", "component", "template_id", "fuzzy_score"]
df_eval = df_eval.reindex(columns=[c for c in cols if c in df_eval.columns])

print("Joined rows:", len(df_eval), "/", len(gold), "(dropped:", len(gold)-len(df_eval), ")")
df_eval.head(3)

# helpers
def exact_match(a: str, b: str) -> bool:
    return (a or "").strip() == (b or "").strip()

def norm_edit_sim(a: str, b: str) -> float:
    """Normalized similarity via difflib SequenceMatcher (0..1)"""
    return difflib.SequenceMatcher(None, (a or ""), (b or "")).ratio()


try:
    from rapidfuzz.distance import Levenshtein
    def norm_edit_sim(a: str, b: str) -> float:
        a = (a or ""); b = (b or "")
        if not a and not b: return 1.0
        return 1 - (Levenshtein.distance(a, b) / max(len(a), len(b)))
except Exception:
    pass


# ### Cell 20

# In[31]:


import matplotlib.pyplot as plt

assert "template" in df_eval.columns and "pred_template" in df_eval.columns, "Missing required columns."


df_eval["exact"] = (df_eval["template"].str.strip() == df_eval["pred_template"].str.strip()).astype(int)
df_eval["sim"]   = [norm_edit_sim(g, p) for g, p in zip(df_eval["template"], df_eval["pred_template"])]

acc = df_eval["exact"].mean() if len(df_eval) else 0.0
sim_mean = df_eval["sim"].mean() if len(df_eval) else 0.0
sim_med  = df_eval["sim"].median() if len(df_eval) else 0.0

print(f"Exact match accuracy: {acc*100:.2f}%")
print(f"Avg similarity (edit-ratio): {sim_mean:.3f}  |  median: {sim_med:.3f}")

if "fuzzy_score" in df_eval.columns:
    try:
        df_eval["fuzzy_score"] = pd.to_numeric(df_eval["fuzzy_score"], errors="coerce")
    except: pass
    print("AdaParser fuzzy_score present. Mean:", np.nanmean(df_eval["fuzzy_score"].values))


plt.figure(figsize=(6,4))
plt.hist(df_eval["sim"], bins=30)
plt.title("Template Similarity (gold vs. predicted)")
plt.xlabel("Edit similarity (0..1)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


if "template_id" in df_eval.columns:
    top_pred = df_eval["template_id"].value_counts().head(15)
    plt.figure(figsize=(8,4))
    plt.bar(top_pred.index.astype(str), top_pred.values)
    plt.xticks(rotation=60, ha="right")
    plt.title("Top Predicted Template IDs (count)")
    plt.tight_layout()
    plt.show()


if "level" in df_eval.columns:
    by_lvl = df_eval.groupby("level")["exact"].mean().sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    plt.bar(by_lvl.index.astype(str), (by_lvl.values*100.0))
    plt.title("Exact Match Accuracy by Log Level")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.show()


# ### Cell 21

# In[32]:


from IPython.display import display, HTML


N = 20
worst = df_eval.sort_values("sim", ascending=True).head(N).copy()


cols_show = ["message", "template", "pred_template"]
if "level" in worst.columns: cols_show.insert(1, "level")
if "component" in worst.columns: cols_show.insert(1, "component")
display(worst[cols_show])


def html_diff(a, b):
    a = (a or ""); b = (b or "")
    a, b = a.replace("&","&amp;"), b.replace("&","&amp;")
    differ = difflib.HtmlDiff(wrapcolumn=80)
    return differ.make_table(a.split(), b.split(), fromdesc="gold", todesc="pred", context=True, numlines=1)

html_rows = []
for _, row in worst.head(5).iterrows():
    html_rows.append("<h4>Message</h4><pre style='white-space:pre-wrap'>" + row['message'] + "</pre>")
    html_rows.append(html_diff(row["template"], row["pred_template"]))
report_html = "<html><body>" + "\n".join(html_rows) + "</body></html>"

report_path = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_eval_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_html)

print("Wrote HTML diff report:", report_path)
display(HTML(report_html))


# ### Cell 22

# In[33]:


pairs = (
    df_eval.loc[df_eval["template"] != df_eval["pred_template"], ["template","pred_template"]]
    .value_counts()
    .reset_index(name="count")
)
print("Top confusions:")
display(pairs.head(20))


# In[ ]:





# In[ ]:




