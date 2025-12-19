"""
api.py
FastAPI service for log processing with T5 + Drain3 model + rules
"""

import os
import json
import uuid
import re
import threading
import time
import shutil
from pathlib import Path
from typing import Literal, Optional
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse

from dotenv import load_dotenv

from model_wrapper import CombinedModel


# --------------------------------------------------------------------
# Load .env FIRST (so BASE_DIR works)
# --------------------------------------------------------------------
# Loads .env from current working dir OR script dir
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
load_dotenv(override=False)


# --------------------------------------------------------------------
# Universal Paths & globals
# --------------------------------------------------------------------
# BASE_DIR priority:
#   1) .env BASE_DIR
#   2) folder containing this api.py
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent)).resolve()

CONFIG_DIR = (BASE_DIR / os.getenv("CONFIG_DIR", "config")).resolve()
JOBS_DIR = (BASE_DIR / os.getenv("JOBS_DIR", "jobs")).resolve()
UPLOADS_DIR = (BASE_DIR / os.getenv("UPLOADS_DIR", "uploads")).resolve()
RESULTS_DIR = (BASE_DIR / os.getenv("RESULTS_DIR", "results")).resolve()

MASTER_DIR = (BASE_DIR / os.getenv("MASTER_DIR", "master_results")).resolve()
PROCESSED_MASTER_PATH = (MASTER_DIR / os.getenv("MASTER_FILE", "processed_master.csv")).resolve()
MASTER_ARCHIVE_DIR = (MASTER_DIR / os.getenv("MASTER_ARCHIVE_DIR", "archive")).resolve()

MODEL_PATH = (CONFIG_DIR / os.getenv("MODEL_FILE", "combined_model_full.pkl")).resolve()
RULES_PATH = (CONFIG_DIR / os.getenv("RULES_FILE", "label_rules.json")).resolve()
TEMPLATE_MAP_PATH = (CONFIG_DIR / os.getenv("TEMPLATE_MAP_FILE", "template_label_map.json")).resolve()
CLUSTER_MAP_PATH = (CONFIG_DIR / os.getenv("CLUSTER_MAP_FILE", "cluster_label_map.json")).resolve()
LABEL_META_PATH = (CONFIG_DIR / os.getenv("LABEL_META_FILE", "label_meta_map.json")).resolve()

for d in (CONFIG_DIR, JOBS_DIR, UPLOADS_DIR, RESULTS_DIR, MASTER_DIR, MASTER_ARCHIVE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Global model/rules/job state
model: Optional[CombinedModel] = None
label_rules = []
jobs: dict[str, dict] = {}

#
template_label_map: dict[str, str] = {}   
cluster_label_map: dict[str, str] = {}    
label_meta_map: dict[str, dict] = {}      

# Timestamp regex: [2025-09-03T07:52:15.220] ...
TS_PATTERN = re.compile(r"^\[(.*?)\]")


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def extract_ts(raw: str) -> str:
    m = TS_PATTERN.match(str(raw))
    return m.group(1) if m else ""


def load_label_rules():
    global label_rules
    if not RULES_PATH.exists():
        print(f"[Rules] Warning: label_rules.json not found at {RULES_PATH}", flush=True)
        label_rules = []
        return

    with RULES_PATH.open("r", encoding="utf-8") as f:
        label_rules = json.load(f)

    print(f"[Rules] Loaded {len(label_rules)} rules from {RULES_PATH}", flush=True)


def save_json_safe(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def format_eta(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    seconds = int(seconds)
    if seconds < 60:
        return f"≈ {seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"≈ {minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"≈ {hours}h {minutes}m"


def update_job_progress(job_id: Optional[str], *, progress: Optional[float] = None, processed_rows: Optional[int] = None):
    if not job_id or job_id not in jobs:
        return

    job = jobs[job_id]

    if progress is not None:
        job["progress"] = max(0.0, min(1.0, float(progress)))

    if processed_rows is not None:
        job["processed_rows"] = int(processed_rows)

    started_at = job.get("started_at")
    prog = job.get("progress", 0.0)

    if not started_at or prog <= 0.0:
        job["eta_seconds"] = None
        job["eta_human"] = None
        return

    try:
        dt_start = datetime.fromisoformat(started_at)
    except Exception:
        job["eta_seconds"] = None
        job["eta_human"] = None
        return

    elapsed = (datetime.now() - dt_start).total_seconds()
    if elapsed <= 0:
        job["eta_seconds"] = None
        job["eta_human"] = None
        return

    total_est = elapsed / prog
    eta = max(total_est - elapsed, 0.0)
    job["eta_seconds"] = eta
    job["eta_human"] = format_eta(eta)


# --------------------------------------------------------------------
# Rules & auto-learning helpers
# --------------------------------------------------------------------
def apply_label_rules(row: dict, rule_index: int | None = None) -> dict:
    global label_rules

    if not isinstance(row, dict):
        row = {"raw": str(row)}

    row.setdefault("raw", "")
    row.setdefault("t5_template", "")
    row.setdefault("auto_label", "unknown")

    row["final_label"] = row.get("auto_label", "unlabeled")
    row["severity"] = row.get("severity", "info")
    row["owner_team"] = row.get("owner_team", "unknown")
    row["tags"] = row.get("tags", "")

    for rule in label_rules:
        if not isinstance(rule, dict):
            continue

        matched = False
        for condition in rule.get("match_any", []):
            if not isinstance(condition, dict):
                continue
            field = condition.get("field", "raw")
            contains = condition.get("contains", "")
            if str(contains).lower() and str(contains).lower() in str(row.get(field, "")).lower():
                matched = True
                break

        if matched:
            set_values = rule.get("set", {})
            if isinstance(set_values, dict):
                row["final_label"] = set_values.get("final_label", row["final_label"])
                row["severity"] = set_values.get("severity", row["severity"])
                row["owner_team"] = set_values.get("owner_team", row["owner_team"])
                row["tags"] = set_values.get("tags", row["tags"])
            break

    return row


def apply_cluster_memory(df: pd.DataFrame):
    if df.empty or not cluster_label_map:
        return
    for i in range(len(df)):
        cid = df.at[i, "cluster_id"]
        auto_label = df.at[i, "auto_label"]
        if auto_label not in ("unknown", "unlabeled") and str(auto_label).strip():
            continue
        if isinstance(cid, (int, float)) and cid != -1:
            key = str(int(cid))
            if cluster_label_map.get(key):
                df.at[i, "auto_label"] = cluster_label_map[key]


def apply_template_memory(df: pd.DataFrame):
    if df.empty or not template_label_map:
        return
    for i in range(len(df)):
        auto_label = df.at[i, "auto_label"]
        if auto_label not in ("unknown", "unlabeled") and str(auto_label).strip():
            continue
        tpl = str(df.at[i, "t5_template"]).strip()
        if tpl and tpl in template_label_map:
            df.at[i, "auto_label"] = template_label_map[tpl]


def apply_label_metadata(df: pd.DataFrame):
    if df.empty or not label_meta_map:
        return
    for i in range(len(df)):
        label = str(df.at[i, "final_label"]).strip()
        if not label or label in ("unknown", "unlabeled"):
            label = str(df.at[i, "auto_label"]).strip()
            if not label or label in ("unknown", "unlabeled"):
                continue

        meta = label_meta_map.get(label)
        if not isinstance(meta, dict):
            continue

        if df.at[i, "severity"] in ("info", "", None) and meta.get("severity"):
            df.at[i, "severity"] = meta["severity"]

        if df.at[i, "owner_team"] in ("unknown", "", None) and meta.get("owner_team"):
            df.at[i, "owner_team"] = meta["owner_team"]

        if df.at[i, "tags"] in ("", None) and meta.get("tags"):
            df.at[i, "tags"] = meta["tags"]


def auto_learn_from_df(df: pd.DataFrame):
    if df.empty:
        return

    new_templates = 0
    new_clusters = 0
    new_meta = 0

    for _, row in df.iterrows():
        final_label = str(row.get("final_label", "")).strip()
        auto_label = str(row.get("auto_label", "")).strip()

        label = None
        if final_label and final_label not in ("unknown", "unlabeled"):
            label = final_label
        elif auto_label and auto_label not in ("unknown", "unlabeled"):
            label = auto_label

        if not label:
            continue

        tpl = str(row.get("t5_template", "")).strip()
        cid = row.get("cluster_id", -1)

        if tpl and tpl not in template_label_map:
            template_label_map[tpl] = label
            new_templates += 1

        if isinstance(cid, (int, float)) and cid != -1:
            key = str(int(cid))
            if key not in cluster_label_map:
                cluster_label_map[key] = label
                new_clusters += 1

        severity = str(row.get("severity", "")).strip()
        owner_team = str(row.get("owner_team", "")).strip()
        tags = str(row.get("tags", "")).strip()

        has_useful_meta = (severity not in ("", None)) or (owner_team not in ("", "unknown", None)) or bool(tags)
        if has_useful_meta and not label_meta_map.get(label):
            label_meta_map[label] = {
                "severity": severity or "info",
                "owner_team": owner_team or "unknown",
                "tags": tags,
            }
            new_meta += 1

    if new_templates:
        print(f"[AutoLearn] New templates learned: {new_templates}", flush=True)
        save_json_safe(template_label_map, TEMPLATE_MAP_PATH)

    if new_clusters:
        print(f"[AutoLearn] New clusters learned: {new_clusters}", flush=True)
        save_json_safe(cluster_label_map, CLUSTER_MAP_PATH)

    if new_meta:
        print(f"[AutoLearn] New label metadata learned: {new_meta}", flush=True)
        save_json_safe(label_meta_map, LABEL_META_PATH)


# --------------------------------------------------------------------
# Core processing
# --------------------------------------------------------------------
def process_file(file_content: bytes, filename: str, mode: Literal["rules", "ml", "both"], job_id: str | None = None) -> pd.DataFrame:
    job_tag = f"Job {job_id}" if job_id else "Job -"
    def log(msg: str):
        print(f"[{job_tag}] {msg}", flush=True)

    # Read
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(pd.io.common.BytesIO(file_content))
        if "raw" not in df.columns:
            raise ValueError("CSV must contain a 'raw' column")
        df["raw"] = df["raw"].astype(str)
        lines = df["raw"].tolist()
    else:
        content = file_content.decode("utf-8", errors="replace")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        df = pd.DataFrame({"raw": lines})

    df["timestamp"] = df["raw"].apply(extract_ts)

    df["t5_template"] = ""
    df["cluster_id"] = -1
    df["cluster_template"] = ""
    df["auto_label"] = "unknown"
    df["final_label"] = "unlabeled"
    df["severity"] = "info"
    df["owner_team"] = "unknown"
    df["tags"] = ""

    total_rows = len(df)
    if job_id and job_id in jobs:
        jobs[job_id]["total_rows"] = total_rows
        update_job_progress(job_id, progress=0.05, processed_rows=0)

    log(f"Processing file={filename} rows={total_rows} mode={mode}")

    # Stage 1: ML
    if mode in ("ml", "both"):
        if model is None:
            raise RuntimeError("Model not loaded")

        t0 = time.perf_counter()
        log(f"[Stage 1/3] Running prediction (T5+Drain3) on {len(lines)} lines...")

        predictions = model.predict_lines(lines)

        for i, pred in enumerate(predictions):
            df.at[i, "t5_template"] = pred.get("t5_template", "")
            df.at[i, "cluster_id"] = pred.get("cluster_id", -1)
            df.at[i, "cluster_template"] = pred.get("cluster_template", "")
            df.at[i, "auto_label"] = pred.get("auto_label", "unknown")

        apply_template_memory(df)
        apply_cluster_memory(df)

        dt = time.perf_counter() - t0
        log(f"[Stage 1/3] DONE in {dt:.2f}s")
        update_job_progress(job_id, progress=0.30)

    # Stage 2: Rules (with heartbeat)
    if mode in ("rules", "both"):
        t0 = time.perf_counter()
        n = len(df)
        log(f"[Stage 2/3] Applying label rules to {n} rows... (rules={len(label_rules)})")
        last_heartbeat = time.perf_counter()

        for i in range(n):
            updated = apply_label_rules(df.iloc[i].to_dict(), rule_index=None)
            df.at[i, "final_label"] = updated.get("final_label", df.at[i, "final_label"])
            df.at[i, "severity"] = updated.get("severity", df.at[i, "severity"])
            df.at[i, "owner_team"] = updated.get("owner_team", df.at[i, "owner_team"])
            df.at[i, "tags"] = updated.get("tags", df.at[i, "tags"])

            frac = (i + 1) / n if n else 1.0
            progress = 0.30 + 0.60 * frac
            update_job_progress(job_id, progress=progress, processed_rows=i + 1)

            now = time.perf_counter()
            if now - last_heartbeat >= 1.0:
                pct = int(frac * 100)
                eta = jobs.get(job_id, {}).get("eta_human") if job_id else None
                log(f"Rules progress: {i+1}/{n} ({pct}%) ETA: {eta}")
                last_heartbeat = now

        dt = time.perf_counter() - t0
        log(f"[Stage 2/3] DONE in {dt:.2f}s")

        label_counts = df["final_label"].value_counts(dropna=False)
        log("Label distribution after rules:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}", flush=True)
        print("", flush=True)

    # Stage 3: Metadata + AutoLearn + finalize
    t0 = time.perf_counter()
    log("[Stage 3/3] Metadata + auto-learn + finalize...")

    apply_label_metadata(df)
    auto_learn_from_df(df)

    apply_template_memory(df)
    apply_cluster_memory(df)

    for i in range(len(df)):
        fl = str(df.at[i, "final_label"]).strip()
        al = str(df.at[i, "auto_label"]).strip()
        if (not fl) or (fl in ("unlabeled", "unknown")):
            if al and al not in ("unknown", "unlabeled"):
                df.at[i, "final_label"] = al

    apply_label_metadata(df)

    dt = time.perf_counter() - t0
    log(f"[Stage 3/3] DONE in {dt:.2f}s")

    update_job_progress(job_id, progress=0.95)

    column_order = [
        "timestamp", "raw", "final_label", "severity", "owner_team", "tags",
        "cluster_id", "t5_template", "cluster_template", "auto_label",
    ]
    df = df[[c for c in column_order if c in df.columns]]

    log(f"✅ Processing completed rows={len(df)}")
    return df


# --------------------------------------------------------------------
# Master helpers
# --------------------------------------------------------------------
def append_to_master(df: pd.DataFrame, job_id: str, filename: str, mode: str):
    processed_at = datetime.now().isoformat()
    df_master = df.copy()

    df_master.insert(0, "batch_id", job_id)
    df_master.insert(1, "processed_at", processed_at)
    df_master.insert(2, "source_file", filename)
    df_master.insert(3, "job_mode", mode)

    if PROCESSED_MASTER_PATH.exists():
        df_master.to_csv(PROCESSED_MASTER_PATH, mode="a", header=False, index=False)
    else:
        df_master.to_csv(PROCESSED_MASTER_PATH, mode="w", header=True, index=False)

    ts_safe = processed_at.replace(":", "-").replace("T", "_")
    backup_path = MASTER_ARCHIVE_DIR / f"processed_master_{ts_safe}.csv"
    try:
        shutil.copy2(PROCESSED_MASTER_PATH, backup_path)
        print(f"[Master] Archived snapshot -> {backup_path.name}", flush=True)
    except Exception as e:
        print(f"[Master] Failed to archive master: {e}", flush=True)


# --------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------
def ensure_jobs_for_orphan_files():
    tracked_paths = {j.get("stored_path") for j in jobs.values()}
    for f in UPLOADS_DIR.iterdir():
        if not f.is_file():
            continue
        if str(f) in tracked_paths:
            continue
        if not (f.name.lower().endswith(".log") or f.name.lower().endswith(".txt") or f.name.lower().endswith(".csv")):
            continue

        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": f.name,
            "stored_path": str(f),
            "mode": "both",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "finished_at": None,
            "total_rows": None,
            "processed_rows": 0,
            "progress": 0.0,
            "eta_seconds": None,
            "eta_human": None,
            "result_path": None,
            "error": None,
        }
        print(f"[Queue] Discovered orphan file: {f.name} -> job {job_id}", flush=True)


def worker_loop():
    print("[Worker] Background worker started", flush=True)
    while True:
        try:
            ensure_jobs_for_orphan_files()

            queued = [j for j in jobs.values() if j.get("status") == "queued"]
            if not queued:
                time.sleep(1.0)
                continue

            queued.sort(key=lambda j: j.get("created_at", ""))
            job = queued[0]

            job_id = job["job_id"]
            path = Path(job["stored_path"])
            filename = job["filename"]
            mode = job.get("mode", "both")

            if not path.exists():
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = "Source file not found in uploads/"
                continue

            jobs[job_id]["status"] = "running"
            jobs[job_id]["started_at"] = datetime.now().isoformat()
            update_job_progress(job_id, progress=0.01, processed_rows=0)

            try:
                content = path.read_bytes()
                df = process_file(content, filename, mode, job_id=job_id)

                result_path = RESULTS_DIR / f"{job_id}_{filename}.csv"
                df.to_csv(result_path, index=False)

                append_to_master(df, job_id=job_id, filename=filename, mode=mode)

                jobs[job_id]["result_path"] = str(result_path)
                jobs[job_id]["status"] = "done"
                jobs[job_id]["finished_at"] = datetime.now().isoformat()
                jobs[job_id]["total_rows"] = jobs[job_id].get("total_rows") or len(df)
                jobs[job_id]["processed_rows"] = jobs[job_id]["total_rows"]
                update_job_progress(job_id, progress=1.0)

                print(f"[Worker] Job {job_id} completed: {len(df)} rows", flush=True)

            except Exception as e:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
                print(f"[Worker] Job {job_id} failed: {e}", flush=True)

            finally:
                # delete uploaded source file after processing attempt
                try:
                    if path.exists():
                        path.unlink()
                        print(f"[Worker] Deleted source file: {path.name}", flush=True)
                except Exception as e:
                    print(f"[Worker] Failed to delete source file {path.name}: {e}", flush=True)

        except Exception as e:
            print(f"[Worker] Unexpected error: {e}", flush=True)
            time.sleep(2.0)


def start_worker():
    threading.Thread(target=worker_loop, daemon=True).start()



# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------
app = FastAPI(
    title="Log Processing API",
    description="Log processing with auto-learning + queue + master file",
    version="2.2.0",
)


@app.on_event("startup")
async def startup_event():
    global model, template_label_map, cluster_label_map, label_meta_map

    print("\n" + "=" * 40, flush=True)
    print("Starting Log Processing API", flush=True)
    print("=" * 40, flush=True)

    print(f"[Paths] BASE_DIR={BASE_DIR}", flush=True)
    print(f"[Paths] CONFIG_DIR={CONFIG_DIR}", flush=True)
    print(f"[Paths] MODEL_PATH={MODEL_PATH}", flush=True)

    model = CombinedModel(str(MODEL_PATH), base_dir=BASE_DIR, config_dir=CONFIG_DIR)
    model.load()

    template_label_map = model.template_label_map
    print(f"[Startup] Template map entries: {len(template_label_map)}", flush=True)

    if CLUSTER_MAP_PATH.exists():
        try:
            cluster_label_map.update(json.loads(CLUSTER_MAP_PATH.read_text(encoding="utf-8")))
            print(f"[Startup] Loaded cluster map: {len(cluster_label_map)}", flush=True)
        except Exception as e:
            print(f"[Startup] Failed to load cluster map: {e}", flush=True)

    if LABEL_META_PATH.exists():
        try:
            label_meta_map.update(json.loads(LABEL_META_PATH.read_text(encoding="utf-8")))
            print(f"[Startup] Loaded label meta map: {len(label_meta_map)}", flush=True)
        except Exception as e:
            print(f"[Startup] Failed to load label meta map: {e}", flush=True)

    load_label_rules()
    start_worker()

    print("API ready!", flush=True)
    print("=" * 60 + "\n", flush=True)


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "base_dir": str(BASE_DIR),
        "config_dir": str(CONFIG_DIR),
        "uploads_dir": str(UPLOADS_DIR),
        "results_dir": str(RESULTS_DIR),
        "master_file": str(PROCESSED_MASTER_PATH),
        "model_loaded": model is not None,
        "rules_count": len(label_rules),
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    mode: Literal["rules", "ml", "both"] = Query(default="both"),
):
    job_id = str(uuid.uuid4())
    stored_name = f"{job_id}_{file.filename}"
    stored_path = UPLOADS_DIR / stored_name

    content = await file.read()
    stored_path.write_bytes(content)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "stored_path": str(stored_path),
        "mode": mode,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "finished_at": None,
        "total_rows": None,
        "processed_rows": 0,
        "progress": 0.0,
        "eta_seconds": None,
        "eta_human": None,
        "result_path": None,
        "error": None,
    }

    print(f"[API] Upload queued: {file.filename} -> job {job_id}", flush=True)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job not ready: {job['status']}")

    result_path = Path(job["result_path"]) if job.get("result_path") else None
    if not result_path or not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(path=result_path, filename=f"processed_{job['filename']}.csv", media_type="text/csv")


@app.get("/master")
async def get_master():
    if not PROCESSED_MASTER_PATH.exists():
        raise HTTPException(status_code=404, detail="Master file not found yet.")
    return FileResponse(path=PROCESSED_MASTER_PATH, filename="processed_master.csv", media_type="text/csv")
