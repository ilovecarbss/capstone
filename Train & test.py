import os
BASE_DIR = r"C:\Users\harik\KS"      
INPUT_LOGS      = os.path.join(BASE_DIR, "Dataset")      
PROCESSED_OUT   = os.path.join(BASE_DIR, "Processed")     
RESULTS_OUT     = os.path.join(BASE_DIR, "Results")       
STARTER_FOLDER  = os.path.join(BASE_DIR, "loghub_starter")
for p in (PROCESSED_OUT, RESULTS_OUT, STARTER_FOLDER):
    os.makedirs(p, exist_ok=True)
print("Input logs :", INPUT_LOGS)
print("Processed  :", PROCESSED_OUT)
print("Results    :", RESULTS_OUT)
print("Starter    :", STARTER_FOLDER)
import pathlib
preprocess_file = pathlib.Path(STARTER_FOLDER) / "preprocess_logs.py"
code = r'''
import re, argparse
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
    m = re.match(r'^(\d{6})\s+(\d{6})', prefix)  # e.g., 081109 203615
    if m:
        date, time_ = m.group(1), m.group(2)
        try: return datetime.strptime(date+time_, "%y%m%d%H%M%S").isoformat()
        except: pass
    for fmt in TS_FORMATS:
        try:
            dt = datetime.strptime(prefix, fmt)
            if fmt == "%b %d %H:%M:%S":
                dt = dt.replace(year=datetime.now().year)
            return dt.isoformat()
        except: pass
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
    level = None; component = None
    for lvl in ["TRACE","DEBUG","INFO","WARN","WARNING","ERROR","FATAL","CRITICAL"]:
        if re.search(rf'\b{lvl}\b', line):
            level = lvl; break
    m = re.search(r'([A-Za-z_][\w\.-]{2,})(?:\[\d+\])?:', line) or re.search(r'\b([A-Za-z_][\w\.-]{2,})\b', line)
    if m: component = m.group(1)
    return level, component
def parse_line(line: str):
    ts = None
    for sep in ["] ", " - ", ": ", "  "]:
        if sep in line[:40]:
            ts = try_parse_ts(line.split(sep, 1)[0]); break
    level, comp = split_level_component(line)
    return ts, level, comp, line.strip()
def process_file(path: Path):
    rows = []
    with path.open("r", errors="ignore", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line: continue
            ts, level, comp, msg = parse_line(line)
            rows.append({
                "dataset": path.stem, "line_no": i, "timestamp": ts,
                "level": level, "component": comp,
                "message": msg, "template": mask_template(msg)
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
        df = process_file(p); frames.append(df)
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
import os, subprocess, pathlib
assert 'INPUT_LOGS' in globals() and 'PROCESSED_OUT' in globals() and 'STARTER_FOLDER' in globals(), "Run Cell 1."
preprocess_file = pathlib.Path(STARTER_FOLDER) / "preprocess_logs.py"
print("Logs folder exists:", os.path.isdir(INPUT_LOGS), "-", INPUT_LOGS)
print("Starter script exists:", preprocess_file.exists(), "-", preprocess_file)
os.makedirs(PROCESSED_OUT, exist_ok=True)
env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8" 
print("Preprocessing ...")
proc = subprocess.run(
    ["python", str(preprocess_file), "--input", INPUT_LOGS, "--output", PROCESSED_OUT],
    text=True, capture_output=True, env=env
)
print(proc.stdout)
if proc.returncode != 0:
    print("Error:\n", proc.stderr)
else:
    print("OK. CSVs written to:", PROCESSED_OUT)
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
        if p.name == "ALL_combined.csv": continue
        df = pd.read_csv(p, dtype=str)
        ds = df.get("dataset", p.stem).iloc[0] if not df.empty else p.stem
        n = len(df)
        u = df["template"].nunique(dropna=False) if "template" in df else 0
        top = template_stats(df).head(20) if "template" in df else pd.DataFrame()
        rare = simple_anomalies(template_stats(df)) if "template" in df else pd.DataFrame()
        if not top.empty: top.to_csv(out_dir / f"top_templates_{ds}.csv", index=False, encoding="utf-8")
        if not rare.empty: rare.to_csv(out_dir / f"anomalies_{ds}.csv", index=False, encoding="utf-8")
        rows.append({"dataset": ds, "num_lines": n, "unique_templates": u, "template_density": round((u / max(n,1)), 4)})
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
import os, subprocess, pathlib
env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"
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
import os, glob, pandas as pd
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
import os, pandas as pd, matplotlib.pyplot as plt
summary_path = os.path.join(RESULTS_OUT, "summary_metrics.csv")
if not os.path.exists(summary_path):
    print("summary_metrics.csv not found.")
else:
    df = pd.read_csv(summary_path).sort_values("template_density", ascending=False)
    plt.figure()
    plt.bar(df["dataset"], df["template_density"])
    plt.xticks(rotation=60, ha="right")
    plt.title("Template Density by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Template Density")
    plt.tight_layout()
    plt.show()
import os
LABELS_OUT = os.path.join(BASE_DIR, "Labels")
MODEL_OUT  = os.path.join(BASE_DIR, "Models")
os.makedirs(LABELS_OUT, exist_ok=True)
os.makedirs(MODEL_OUT,  exist_ok=True)
TARGET_DATASET = "OpenSSH_2k"  
SAMPLE_SIZE    = 600           
print("Labels dir:", LABELS_OUT)
print("Models dir:", MODEL_OUT)
import os, json, pandas as pd
csv_path = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")
assert os.path.exists(csv_path), f"Missing {csv_path}"
df = pd.read_csv(csv_path, dtype=str).fillna("")
df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
records = [{"dataset": TARGET_DATASET,
            "message": r.get("message",""),
            "level": r.get("level",""),
            "component": r.get("component","")} for _, r in df.iterrows()]
jsonl_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_to_label.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("Wrote:", jsonl_path, "| rows:", len(records))
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
    demos = "\n\n".join([f"Input: {m}\nTemplate: {t}" for m,t in FEW_SHOTS])
    return f"{SYSTEM_INSTRUCTIONS}\n\n{demos}\n\nInput: {message}\nTemplate:"
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
    r = re.sub(r'<[^>]+>', '<NUM>', r)  
    return r
import os, json, re, pandas as pd
jsonl_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_to_label.jsonl")
assert os.path.exists(jsonl_path), "Run Cell 7 first."
def quick_mask(s):
    s = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}', '<UUID>', s)
    s = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', s)
    s = re.sub(r'(?<![A-Za-z])\d{3,}(?![A-Za-z])', '<NUM>', s)
    return s
labeled = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        raw_template = quick_mask(r["message"])           
        cleaned      = correct_once(raw_template)          # self-correct
        labeled.append({
            "dataset": r["dataset"],
            "message": r["message"],
            "level":   r["level"],
            "component": r["component"],
            "template": cleaned
        })
labels_csv = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels.csv")
pd.DataFrame(labeled).to_csv(labels_csv, index=False, encoding="utf-8")
print("Wrote labels:", labels_csv, "| rows:", len(labeled))
import os, re, json, time, math, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
PREVIEW_LIMIT   = 80      
BATCH_SIZE      = 16      
MAX_NEW_TOKENS  = 32      
TRUNC_INPUT_CH  = 220     
os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, "hf_cache"))
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
jsonl_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_to_label.jsonl")
assert os.path.exists(jsonl_path), "Run Cell 7 first."
LOCAL_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map="cpu",
)
model.eval()
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass
SYSTEM_INSTRUCTIONS = (
    "Task: Convert a raw log line into a stable template. "
    "Keep constant words, replace variables with tags: <IP> <NUM> <UUID> <PATH> <HEX> <DATE> <TIME> <MS> <HOST>. "
    "Output ONLY the template (one line). No explanations."
)
def build_prompt_fast(message: str) -> str:
    msg = message.strip().replace("\n", " ")
    if len(msg) > TRUNC_INPUT_CH:
        msg = msg[:TRUNC_INPUT_CH] + " ..."
    return f"{SYSTEM_INSTRUCTIONS}\n\nInput: {msg}\nTemplate:"
TEMPLATE_GRAB = re.compile(r"Template:\s*(.*)", re.DOTALL)
def extract_template(full_text: str) -> str:
    m = TEMPLATE_GRAB.search(full_text)
    txt = (m.group(1) if m else full_text).strip()
    return txt.splitlines()[0]
@torch.inference_mode()
def batch_generate(prompts):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [extract_template(t) for t in texts]
records = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        records.append(r)
        if len(records) >= PREVIEW_LIMIT:
            break
prompts = [build_prompt_fast(r["message"]) for r in records]
rows = []
t0 = time.time()
for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    raws = batch_generate(batch_prompts)
    for raw, r in zip(raws, records[i:i+BATCH_SIZE]):
        fixed = correct_once(raw)  
        rows.append({**r, "template": fixed})
dt = time.time() - t0
preview_csv = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels_llm_preview.csv")
pd.DataFrame(rows).to_csv(preview_csv, index=False, encoding="utf-8")
print(f"Wrote PREVIEW labels: {preview_csv} | rows: {len(rows)} | time: {dt:.1f}s (~{dt/max(1,len(rows)):.2f}s/line)")
import sys, subprocess
def pipi(pkgs): subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
pipi(["transformers==4.44.2", "datasets==3.0.1", "accelerate==0.34.2", "evaluate==0.4.2", "sentencepiece==0.2.0", "scikit-learn==1.5.2"])
pipi(['fsspec[http]==2024.6.1'])
import os, pandas as pd
from sklearn.model_selection import train_test_split
llm_csv    = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels_llm.csv")
regex_csv  = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels.csv")
labels_csv = llm_csv if os.path.exists(llm_csv) else regex_csv
print("Using labels:", labels_csv)
df = pd.read_csv(labels_csv).fillna("")
df = df[(df["message"].str.len()>0) & (df["template"].str.len()>0)].copy()
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
print("Train:", len(train_df), "Val:", len(val_df))
train_df.head(3)
from datasets import Dataset
from transformers import AutoTokenizer
model_name = "t5-small"
SPECIALS = ["<IP>","<NUM>","<UUID>","<PATH>","<HEX>","<DATE>","<TIME>","<MS>","<HOST>"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})
def to_ds(frame):
    return Dataset.from_pandas(
        frame[["message","template"]].rename(columns={"template":"labels"}),
        preserve_index=False
    )
train_ds = to_ds(train_df)
val_ds   = to_ds(val_df)
MAX_IN, MAX_OUT = 256, 128
def preprocess_batch(batch):
    ins  = tokenizer(batch["message"], max_length=MAX_IN, truncation=True)
    outs = tokenizer(text_target=batch["labels"], max_length=MAX_OUT, truncation=True)
    ins["labels"] = outs["input_ids"]
    return ins
train_tok = train_ds.map(preprocess_batch, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(preprocess_batch,   batched=True, remove_columns=val_ds.column_names)
print("Tokenized:", len(train_tok), "train /", len(val_tok), "val")
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
    fp16=False,           # CPU
    report_to=[],         
    seed=42,
    dataloader_num_workers=0,  # Windows stability
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
from transformers import pipeline
pipe = pipeline("text2text-generation", model=trainer.model, tokenizer=tokenizer, max_new_tokens=64)
samples = val_df.sample(min(5, len(val_df)), random_state=7)["message"].tolist()
for s in samples:
    out = pipe(s)[0]["generated_text"]
    print("\nINPUT :", s[:160].replace("\n"," "))
    print("OUTPUT:", out)
import re
TAG_WORDS = ["IP","NUM","UUID","PATH","HEX","DATE","TIME","MS","HOST"]
TAG_FIX = re.compile(r'\b(' + '|'.join(TAG_WORDS) + r')\b>?')  
def normalize_template_basic(s: str) -> str:
    t = TAG_FIX.sub(lambda m: f"<{m.group(1)}>", s)
    t = re.sub(r'\s+', ' ', t).strip()
    return correct_once(t)  
def normalize_out(s: str) -> str:
    try:
        return normalize_template_strict(s)  
    except NameError:
        return normalize_template_basic(s)
from transformers import pipeline
pipe = pipeline("text2text-generation", model=trainer.model, tokenizer=tokenizer, max_new_tokens=64, do_sample=False, num_beams=4)
samples = val_df.sample(min(5, len(val_df)), random_state=7)["message"].tolist()
for s in samples:
    raw = pipe(s)[0]["generated_text"]
    fixed = normalize_out(raw)
    print("\nINPUT :", s[:160].replace("\n"," "))
    print("RAW   :", raw)
    print("FIXED :", fixed)
i# 12B — normalize labels before training (run after Cell 12)
import os, pandas as pd
assert 'labels_csv' in globals(), "Run Cell 12 first to set labels_csv"
df = pd.read_csv(labels_csv).fillna("")
def _norm(t):
    try:   return normalize_template_strict(t)   # uses 15C
    except NameError:
        return correct_once(t)                  # fallback
df["template"] = df["template"].astype(str).map(_norm)
clean_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels_clean.csv")
df.to_csv(clean_path, index=False, encoding="utf-8")
print("Wrote cleaned labels:", clean_path)
from sklearn.model_selection import train_test_split
df = pd.read_csv(clean_path).fillna("")
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
print("Train:", len(train_df), "Val:", len(val_df))
import os
final_dir = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_small_final")
os.makedirs(final_dir, exist_ok=True)
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print("Saved model to:", final_dir)
import os, time, pandas as pd, torch
from transformers import pipeline
from tqdm import tqdm
csv_in  = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")
csv_out = os.path.join(RESULTS_OUT,  f"{TARGET_DATASET}_preds.csv")
tmp_out = os.path.join(RESULTS_OUT,  f"{TARGET_DATASET}_preds.tmp.csv")
assert os.path.exists(csv_in), f"Missing {csv_in}"
os.makedirs(RESULTS_OUT, exist_ok=True)
PREVIEW_LIMIT  = None      
BATCH_SIZE     = 128       
MAX_NEW_TOKENS = 40        
NUM_BEAMS      = 1       
def _normalize(s: str) -> str:
    try:    return normalize_template_strict(s)   
    except: return correct_once(s)               
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass
gen = pipeline(
    "text2text-generation",
    model=trainer.model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    num_beams=NUM_BEAMS,
)
df = pd.read_csv(csv_in, dtype=str).fillna("")
if PREVIEW_LIMIT is not None:
    df = df.head(PREVIEW_LIMIT).copy()
msgs = df["message"].tolist()
N = len(msgs)
pred = [None] * N
start_idx = 0
if os.path.exists(tmp_out):
    try:
        tmp_df = pd.read_csv(tmp_out, dtype=str).fillna("")
        if len(tmp_df) == N and "pred_template" in tmp_df:
            pred = tmp_df["pred_template"].tolist()
            for i, v in enumerate(pred):
                if not isinstance(v, str) or v == "" or v.lower() == "nan":
                    start_idx = i
                    break
                start_idx = N
    except Exception:
        pass
t0 = time.time()
with tqdm(total=N, desc="Inferring", initial=start_idx, unit="lines") as pbar:
    for i in range(start_idx, N, BATCH_SIZE):
        j = min(i + BATCH_SIZE, N)
        outs = gen(msgs[i:j])
        for k, o in enumerate(outs):
            pred[i + k] = _normalize(o["generated_text"])
        pbar.update(j - i)
        if ((i // BATCH_SIZE) % 10 == 0) or (j == N):
            df_tmp = df.copy()
            df_tmp["pred_template"] = pred
            df_tmp.to_csv(tmp_out, index=False, encoding="utf-8")
df_out = df.copy()
df_out["pred_template"] = pred
df_out.to_csv(csv_out, index=False, encoding="utf-8")
elapsed = time.time() - t0
print(f"Wrote: {csv_out} | rows: {len(df_out)} | time: {elapsed:.1f}s (~{N/max(1,elapsed):.2f} lines/s)")
print(f"(Checkpoint kept at: {tmp_out})")
import os, pandas as pd
csv_out = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_preds.csv")
assert os.path.exists(csv_out), "Run Cell 17 first."
dfp = pd.read_csv(csv_out, dtype=str).fillna("")
cnt = dfp["pred_template"].value_counts().reset_index()
cnt.columns = ["template","count"]
display(cnt.head(20))
top_path = os.path.join(RESULTS_OUT, f"top_pred_templates_{TARGET_DATASET}.csv")
cnt.to_csv(top_path, index=False, encoding="utf-8")
print("Saved tops to:", top_path)
mu, sigma = cnt["count"].mean(), cnt["count"].std(ddof=0) or 1.0
cnt["z_score"] = (cnt["count"] - mu) / sigma
rare = cnt[cnt["z_score"] <= -3.0].sort_values("z_score")
rare_path = os.path.join(RESULTS_OUT, f"anomalies_pred_{TARGET_DATASET}.csv")
rare.to_csv(rare_path, index=False, encoding="utf-8")
print("Saved anomalies to:", rare_path)
display(rare.head(10))
import os, pandas as pd
from sklearn.model_selection import train_test_split
llm_path   = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels_llm_preview.csv")
quick_path = os.path.join(LABELS_OUT, f"{TARGET_DATASET}_labels.csv") 
assert os.path.exists(llm_path), "Run Cell 10B (LLM labeling) first."
df_llm = pd.read_csv(llm_path).fillna("")
if os.path.exists(quick_path):
    df_quick = pd.read_csv(quick_path).fillna("")
    df_all = pd.concat(
        [df_llm[["message","template"]], df_quick[["message","template"]]],
        ignore_index=True
    ).drop_duplicates()
else:
    df_all = df_llm[["message","template"]].drop_duplicates()
try:
    df_all["template"] = df_all["template"].astype(str).map(normalize_template_strict)
except NameError:
    df_all["template"] = df_all["template"].astype(str).map(correct_once)
df_all = df_all[(df_all["message"].str.len()>0) & (df_all["template"].str.len()>0)].copy()
train_df, val_df = train_test_split(df_all, test_size=0.15, random_state=42)
print("V2 Train:", len(train_df), "Val:", len(val_df))
train_df.head(3)
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, Seq2SeqTrainingArguments
)
base_model_dir = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_small_final")
assert os.path.exists(base_model_dir), "Run Cell 16 to save your first model."
SPECIALS = ["<IP>","<NUM>","<UUID>","<PATH>","<HEX>","<DATE>","<TIME>","<MS>","<HOST>"]
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})
def to_ds(frame):
    return Dataset.from_pandas(frame[["message","template"]].rename(columns={"template":"labels"}), preserve_index=False)
train_ds = to_ds(train_df)
val_ds   = to_ds(val_df)
MAX_IN, MAX_OUT = 256, 128
def preprocess_batch(batch):
    ins  = tokenizer(batch["message"], max_length=MAX_IN, truncation=True)
    outs = tokenizer(text_target=batch["labels"], max_length=MAX_OUT, truncation=True)
    ins["labels"] = outs["input_ids"]
    return ins
train_tok = train_ds.map(preprocess_batch, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(preprocess_batch,   batched=True, remove_columns=val_ds.column_names)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_dir)
model.resize_token_embeddings(len(tokenizer))
save_dir = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_small_v2")
args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    num_train_epochs=1,   # short adaptive pass
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=False,
    report_to=[],
    seed=42,
    dataloader_num_workers=0,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer_v2 = Trainer(
    model=model, args=args,
    train_dataset=train_tok, eval_dataset=val_tok,
    tokenizer=tokenizer, data_collator=data_collator
)
train_result_v2 = trainer_v2.train()
metrics_v2 = trainer_v2.evaluate()
metrics_v2
final_v2_dir = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_small_v2_final")
os.makedirs(final_v2_dir, exist_ok=True)
trainer_v2.model.save_pretrained(final_v2_dir)
tokenizer.save_pretrained(final_v2_dir)
print("Saved v2 model to:", final_v2_dir)
import os, time, math, pandas as pd, torch
from transformers import pipeline
from tqdm import tqdm
csv_in  = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")
csv_out = os.path.join(RESULTS_OUT,  f"{TARGET_DATASET}_preds_v2.csv")
tmp_out = os.path.join(RESULTS_OUT,  f"{TARGET_DATASET}_preds_v2.tmp.csv")
assert os.path.exists(csv_in), f"Missing {csv_in}"
def _normalize(s: str) -> str:
    try:    return normalize_template_strict(s)   
    except: return correct_once(s)                
final_v2_dir = os.path.join(MODEL_OUT, f"{TARGET_DATASET}_t5_small_v2_final")
assert os.path.exists(final_v2_dir), "Run Cells 20–21 first."
os.makedirs(RESULTS_OUT, exist_ok=True)
BATCH_SIZE     = 128     
MAX_NEW_TOKENS = 40      
NUM_BEAMS      = 1       
PREVIEW_LIMIT  = None   
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass
gen = pipeline(
    "text2text-generation",
    model=final_v2_dir,
    tokenizer=final_v2_dir,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    num_beams=NUM_BEAMS,
)
df = pd.read_csv(csv_in, dtype=str).fillna("")
if PREVIEW_LIMIT is not None:
    df = df.head(PREVIEW_LIMIT).copy()
msgs = df["message"].tolist()
N = len(msgs)
pred = [None] * N
start_idx = 0
if os.path.exists(tmp_out):
    try:
        tmp_df = pd.read_csv(tmp_out, dtype=str).fillna("")
        if len(tmp_df) == N and "pred_template" in tmp_df:
            pred = tmp_df["pred_template"].tolist()
            for i, v in enumerate(pred):
                if not isinstance(v, str) or v == "" or v.lower() == "nan":
                    start_idx = i
                    break
                start_idx = N
    except Exception:
        pass
t0 = time.time()
with tqdm(total=N, desc="Inferring", initial=start_idx, unit="lines") as pbar:
    for i in range(start_idx, N, BATCH_SIZE):
        j = min(i + BATCH_SIZE, N)
        outs = gen(msgs[i:j])
        for k, o in enumerate(outs):
            pred[i + k] = _normalize(o["generated_text"])
        pbar.update(j - i)
        if ((i // BATCH_SIZE) % 10 == 0) or (j == N):
            df_tmp = df.copy()
            df_tmp["pred_template"] = pred
            df_tmp.to_csv(tmp_out, index=False, encoding="utf-8")
df_out = df.copy()
df_out["pred_template"] = pred
df_out.to_csv(csv_out, index=False, encoding="utf-8")
elapsed = time.time() - t0
print(f"Wrote: {csv_out} | rows: {len(df_out)} | time: {elapsed:.1f}s (~{N/max(1,elapsed):.2f} lines/s)")
print(f"(Checkpoint kept at: {tmp_out})")
import pandas as pd, os
from sklearn.metrics import accuracy_score, f1_score
csv_ref  = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")          # reference (regex templates)
csv_pred = os.path.join(RESULTS_OUT,  f"{TARGET_DATASET}_preds_v2.csv")  # your model's predictions
assert os.path.exists(csv_ref) and os.path.exists(csv_pred), "Missing input or prediction CSV"
ref  = pd.read_csv(csv_ref, dtype=str).fillna("")
pred = pd.read_csv(csv_pred, dtype=str).fillna("")
merged = ref.merge(pred[["message","pred_template"]], on="message", how="inner")
def normalize_for_eval(t: str) -> str:
    t = str(t).strip().lower()
    t = t.replace(" ", "")
    return t
y_true = merged["template"].map(normalize_for_eval)
y_pred = merged["pred_template"].map(normalize_for_eval)
acc = (y_true == y_pred).mean()
print(f"Template match accuracy: {acc*100:.2f}%")
from difflib import SequenceMatcher
def sim(a,b): return SequenceMatcher(None,a,b).ratio()
fuzzy = [sim(a,b) for a,b in zip(y_true,y_pred)]
print(f"Avg fuzzy similarity: {sum(fuzzy)/len(fuzzy)*100:.2f}%  (higher = better)")
import os, pandas as pd
from difflib import SequenceMatcher
csv_ref  = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")
csv_pred = os.path.join(RESULTS_OUT,  f"{TARGET_DATASET}_preds_v2.csv")
assert os.path.exists(csv_ref) and os.path.exists(csv_pred), "Missing input or prediction CSV"
ref  = pd.read_csv(csv_ref, dtype=str).fillna("")
pred = pd.read_csv(csv_pred, dtype=str).fillna("")
merged = ref.merge(pred[["message","pred_template"]], on="message", how="inner")
assert len(merged) > 0, "No overlap between reference and predictions."
def norm_both(t: str) -> str:
    try:
        return normalize_template_strict(str(t))
    except NameError:
        return correct_once(str(t))
y_true_n = merged["template"].map(norm_both)
y_pred_n = merged["pred_template"].map(norm_both)
exact = (y_true_n == y_pred_n).mean()
def sim(a,b): return SequenceMatcher(None, a, b).ratio()
fuzzy = sum(sim(a,b) for a,b in zip(y_true_n, y_pred_n)) / len(merged)
print(f"Exact match (normalized): {exact*100:.2f}%")
print(f"Avg fuzzy similarity (normalized): {fuzzy*100:.2f}%")
import re
import numpy as np
TAG = re.compile(r'<[A-Z]+>')
def tags(seq: str):
    return TAG.findall(seq or "")
true_tags = y_true_n.map(tags)
pred_tags = y_pred_n.map(tags)
tp = fp = fn = 0
for t, p in zip(true_tags, pred_tags):
    t_counts = {}
    for x in t: t_counts[x] = t_counts.get(x, 0) + 1
    p_counts = {}
    for x in p: p_counts[x] = p_counts.get(x, 0) + 1
    for tag in set(t_counts) | set(p_counts):
        ct, cp = t_counts.get(tag, 0), p_counts.get(tag, 0)
        tp += min(ct, cp)
        fp += max(0, cp - ct)
        fn += max(0, ct - cp)
prec = tp / (tp + fp) if (tp+fp) else 0.0
rec  = tp / (tp + fn) if (tp+fn) else 0.0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
print(f"Tag-only micro P/R/F1: {prec:.3f}/{rec:.3f}/{f1:.3f}")
import os, re
import pandas as pd
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
assert 'PROCESSED_OUT' in globals() and 'RESULTS_OUT' in globals() and 'TARGET_DATASET' in globals(), \
    "Run Cell 1 and Cell 6 first to define paths and TARGET_DATASET."
csv_ref  = os.path.join(PROCESSED_OUT, f"{TARGET_DATASET}.csv")
csv_pred = os.path.join(RESULTS_OUT,  f"{TARGET_DATASET}_preds_v2.csv")
assert os.path.exists(csv_ref) and os.path.exists(csv_pred), "Missing input or prediction CSV"
ref  = pd.read_csv(csv_ref, dtype=str).fillna("")
pred = pd.read_csv(csv_pred, dtype=str).fillna("")
merged = ref.merge(pred[["message","pred_template"]], on="message", how="inner")
assert len(merged) > 0, "No overlap between reference and predictions."
def _norm(t: str) -> str:
    s = str(t)
    try:
        return normalize_template_strict(s)  
    except NameError:
        return correct_once(s)               
y_true = merged["template"].map(_norm)
y_pred = merged["pred_template"].map(_norm)
-
exact_acc = (y_true == y_pred).mean()
def _sim(a, b):
    return SequenceMatcher(None, a, b).ratio()
fuzzy_scores = [ _sim(a, b) for a, b in zip(y_true, y_pred) ]
fuzzy_avg = sum(fuzzy_scores) / len(fuzzy_scores) if len(fuzzy_scores) else 0.0
TAG_RE = re.compile(r'<[A-Z]+>')
def tags(seq: str):
    return TAG_RE.findall(seq or "")
tp = fp = fn = 0
for t_tags, p_tags in zip(y_true.map(tags), y_pred.map(tags)):
    tc, pc = {}, {}
    for x in t_tags: tc[x] = tc.get(x, 0) + 1
    for x in p_tags: pc[x] = pc.get(x, 0) + 1
    all_tags = set(tc) | set(pc)
    for tag in all_tags:
        ct, cp = tc.get(tag, 0), pc.get(tag, 0)
        tp += min(ct, cp)
        fp += max(0, cp - ct)
        fn += max(0, ct - cp)
prec = tp / (tp + fp) if (tp + fp) else 0.0
rec  = tp / (tp + fn) if (tp + fn) else 0.0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
metrics_csv = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_metrics.csv")
pd.DataFrame([{
    "dataset": TARGET_DATASET,
    "n_samples": len(merged),
    "exact_accuracy": exact_acc,
    "fuzzy_similarity": fuzzy_avg,
    "tag_precision": prec,
    "tag_recall": rec,
    "tag_f1": f1
}]).to_csv(metrics_csv, index=False, encoding="utf-8")
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Exact Acc", "Fuzzy Sim", "Tag P", "Tag R", "Tag F1"]
values = [exact_acc, fuzzy_avg, prec, rec, f1]
ax.bar(labels, values)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score (0–1)")
ax.set_title(f"Parsing Metrics — {TARGET_DATASET}  (n={len(merged)})")
for i, v in enumerate(values):
    ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", va="bottom")
plt.tight_layout()
png_path = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_metrics.png")
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.show()
print("Saved metrics CSV:", metrics_csv)
print("Saved metrics PNG:", png_path)
import matplotlib.pyplot as plt
import os
os.makedirs(RESULTS_OUT, exist_ok=True)
save_path = os.path.join(RESULTS_OUT, f"{TARGET_DATASET}_fuzzy_distribution.png")
plt.figure(figsize=(6,4))
plt.hist(fuzzy_scores, bins=30, color="skyblue", edgecolor="black")
plt.title(f"Distribution of Fuzzy Similarity — {TARGET_DATASET}")
plt.xlabel("Similarity Score (0–1)")
plt.ylabel("Number of Lines")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
print(f"✅ Saved fuzzy similarity distribution to: {save_path}")
