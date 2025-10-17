
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
