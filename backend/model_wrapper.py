"""
model_wrapper.py
Wrapper for loading and using the heavy T5 + Drain3 model
with an automatic template -> label classifier built from rules.csv.

Key points:
- Loads combined_model_full.pkl (T5 + tokenizer + Drain3)
- Loads template_label_map.json if present, else builds from rules.csv
- predict_lines() returns: raw, t5_template, cluster_id, cluster_template, auto_label
- Adds compatibility patch for pickled GenerationConfig mismatches (compile_config)
"""

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import pandas as pd

# transformers is required because your pickle contains a T5 model object
from transformers import GenerationConfig


class CombinedModel:
    """Wrapper for the heavy combined model (T5 + Drain3 + template classifier)."""

    def __init__(
        self,
        model_path: str,
        base_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None,
    ):
        self.model_path = Path(model_path)

        self.base_dir = (
            Path(base_dir)
            if base_dir
            else Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent)).resolve()
        )
        self.config_dir = (
            Path(config_dir)
            if config_dir
            else (self.base_dir / os.getenv("CONFIG_DIR", "config")).resolve()
        )

        # Where to build classifier from (optional)
        rules_csv_candidate = (self.config_dir / "rules.csv").resolve()
        if rules_csv_candidate.exists():
            self.rules_csv = rules_csv_candidate
        else:
            self.rules_csv = (self.base_dir / "rules.csv").resolve()

        # Must match api.py TEMPLATE_MAP_PATH
        self.classifier_path = (
            self.config_dir / os.getenv("TEMPLATE_MAP_FILE", "template_label_map.json")
        ).resolve()

        self.t5_model = None
        self.tokenizer = None
        self.drain = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.template_label_map: Dict[str, str] = {}

        # Generation params
        self.max_src_len = 128
        self.max_gen_len = 64
        self.num_beams = 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self):
        print(f"\n[CombinedModel] Loading heavy model from: {self.model_path}", flush=True)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        with self.model_path.open("rb") as f:
            data = pickle.load(f)

        self.t5_model = data["t5_model"].to(self.device)
        self.tokenizer = data["tokenizer"]
        self.drain = data["drain"]

        # Make inference stable
        self.t5_model.eval()

        print("[CombinedModel] Components loaded:", flush=True)
        print(f"  - t5_model   : {type(self.t5_model)} (device={self.device})", flush=True)
        print(f"  - tokenizer  : {type(self.tokenizer)}", flush=True)
        print(f"  - drain      : {type(self.drain)}", flush=True)
        print(f"  - classifier : {self.classifier_path}", flush=True)
        print(f"  - rules_csv  : {self.rules_csv}", flush=True)

        self._ensure_classifier()

    def predict_lines(self, lines: List[str]) -> List[Dict[str, Any]]:
        if self.t5_model is None or self.tokenizer is None or self.drain is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results: List[Dict[str, Any]] = []

        for line in lines:
            raw = (line or "").strip()

            # 1) T5 template
            try:
                t5_template = self._generate_template(raw)
            except Exception as e:
                print(f"[CombinedModel] T5 error: {e}", flush=True)
                t5_template = ""

            # 2) template -> auto_label
            auto_label = "unknown"
            if t5_template:
                auto_label = self.template_label_map.get(t5_template.strip(), "unknown")

            # 3) Drain3 clustering
            drain_info = self._cluster_with_drain(raw)

            results.append(
                {
                    "raw": raw,
                    "t5_template": t5_template,
                    "cluster_id": drain_info["cluster_id"],
                    "cluster_template": drain_info["cluster_template"],
                    "auto_label": auto_label,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_safe_generation_config(self) -> GenerationConfig:
        """
        Compatibility layer:
        - Your pickled model may carry an older GenerationConfig missing fields
          expected by newer transformers (e.g., compile_config).
        - We clone/build a safe config and patch missing attributes.
        """
        # Start from model's generation_config if present, else build from model config
        base_cfg = getattr(self.t5_model, "generation_config", None)

        if base_cfg is None:
            cfg = GenerationConfig.from_model_config(self.t5_model.config)
        else:
            # safest: rebuild from dict (works across versions)
            try:
                cfg = GenerationConfig.from_dict(base_cfg.to_dict())
            except Exception:
                cfg = GenerationConfig.from_model_config(self.t5_model.config)

        # Patch fields that some transformers versions assume exist
        if not hasattr(cfg, "compile_config"):
            cfg.compile_config = None  # <- fixes: "'GenerationConfig' object has no attribute 'compile_config'"

        # Apply our generation settings
        cfg.max_length = self.max_gen_len
        cfg.num_beams = self.num_beams
        cfg.do_sample = False

        return cfg

    def _generate_template(self, raw: str) -> str:
        if not raw:
            return ""

        inputs = self.tokenizer(
            raw,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_src_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_cfg = self._build_safe_generation_config()

        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs,
                generation_config=gen_cfg,
            )

        template = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return template.strip()

    def _cluster_with_drain(self, raw: str) -> Dict[str, Any]:
        cluster_id = -1
        cluster_template = ""

        try:
            drain_result = self.drain.add_log_message(raw)

            if isinstance(drain_result, dict):
                cluster_id = drain_result.get("cluster_id", -1)
                cluster_template = drain_result.get("template_mined", "")

            elif hasattr(drain_result, "cluster_id"):
                cluster_id = getattr(drain_result, "cluster_id", -1)
                cluster_template = getattr(drain_result, "template_mined", "")

            elif isinstance(drain_result, (str, int)):
                cluster_id = drain_result

        except Exception as e:
            print(f"[CombinedModel] Drain3 error: {e}", flush=True)

        return {"cluster_id": cluster_id, "cluster_template": cluster_template}

    def _ensure_classifier(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Case 1: load existing classifier json
        if self.classifier_path.exists():
            with self.classifier_path.open("r", encoding="utf-8") as f:
                self.template_label_map = json.load(f)
            print(
                f"[CombinedModel] Loaded classifier ({len(self.template_label_map)} templates)",
                flush=True,
            )
            return

        # Case 2: build from rules.csv
        if not self.rules_csv.exists():
            print(
                "[CombinedModel] No classifier JSON and no rules.csv found -> auto_label stays 'unknown'.",
                flush=True,
            )
            return

        print("[CombinedModel] Building template->label map from rules.csv ...", flush=True)

        df_rules = pd.read_csv(self.rules_csv)
        required_cols = {"raw", "final_label"}
        if not required_cols.issubset(df_rules.columns):
            print(
                f"[CombinedModel] rules.csv missing {required_cols} -> cannot build classifier.",
                flush=True,
            )
            return

        template_label_map: Dict[str, str] = {}
        conflicts = 0

        for _, row in df_rules.iterrows():
            raw = str(row["raw"]).strip()
            label = str(row["final_label"]).strip() or "unlabeled"
            if not raw:
                continue

            try:
                template = self._generate_template(raw)
            except Exception as e:
                print(f"[CombinedModel] T5 error while building classifier: {e}", flush=True)
                continue

            key = template.strip()
            if not key:
                continue

            if key in template_label_map and template_label_map[key] != label:
                conflicts += 1
                continue

            template_label_map[key] = label

        self.template_label_map = template_label_map
        print(
            f"[CombinedModel] Built classifier: {len(self.template_label_map)} templates ({conflicts} conflicts)",
            flush=True,
        )

        with self.classifier_path.open("w", encoding="utf-8") as f:
            json.dump(self.template_label_map, f, indent=2, ensure_ascii=False)

        print(f"[CombinedModel] Saved classifier -> {self.classifier_path}", flush=True)
