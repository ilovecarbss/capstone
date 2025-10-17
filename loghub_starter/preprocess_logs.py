
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
