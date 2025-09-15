#!/usr/bin/env python3

import os
import math
import subprocess
import re
import yaml
import shutil
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import json

sns.set()

ROOT = os.path.dirname(os.path.abspath(__file__))
HW_PATH = os.path.join(ROOT, "configs", "hardware-config", "a100_80GB.yaml")
MODEL_PATH = os.path.join(ROOT, "configs", "model-config", "GEMM.yaml")
LLM_MODEL_PATH = os.path.join(ROOT, "configs", "model-config", "LLM.yaml")
OUT_DIR = os.path.join(ROOT, "results", "astra_test")
os.makedirs(OUT_DIR, exist_ok=True)
TMP_ROOT = os.path.join(OUT_DIR, "tmp")
os.makedirs(TMP_ROOT, exist_ok=True)

# VERY START: ensure cache lock is cleared and cache file exists
_CACHE_PATH = os.path.join(ROOT, "astra_cache", "cache.json")
_LOCK_PATH = _CACHE_PATH + ".lock"
try:
    if os.path.exists(_LOCK_PATH):
        os.remove(_LOCK_PATH)
except Exception:
    pass
try:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    # Only create an empty cache if it does not exist; do not overwrite
    if not os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "w") as f:
            json.dump({}, f)
except Exception:
    pass


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _save_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_model_gemm_cfg(cfg: Dict[str, Any], M: int, K: int, N: int) -> Dict[str, Any]:
    if "model_param" not in cfg:
        cfg["model_param"] = {}
    mp = cfg["model_param"]
    mp["mode"] = "GEMM"
    mp["M"] = int(M)
    mp["K"] = int(K)
    mp["N"] = int(N)
    return cfg


def set_hw_params_cfg(cfg: Dict[str, Any], n_workers: int, t: str, kp1: int, kp2: int, backend: str) -> Dict[str, Any]:
    # Ensure everything is intra-node only
    sys_h = cfg.setdefault("system_hierarchy", {})
    sys_h["num_nodes"] = 1
    sys_h["num_devices_per_node"] = int(n_workers)

    # Keep topology ring (already set), no change required

    # Scheduling parameters
    sch = cfg.setdefault("scheduling_param", {})
    sch["auto"] = False
    sch["dp"] = 1
    sch["lp"] = 1
    sch["t"] = str(t)
    sch["kp1"] = int(kp1)
    sch["kp2"] = int(kp2)

    # Network backend selection
    nb = cfg.setdefault("network_backend", {})
    nb["model"] = str(backend)  # "analytical" | "astra"
    # keep existing astra section if present
    return cfg


def set_hw_params_cfg_llm(cfg: Dict[str, Any], n_workers: int, backend: str, intra_topo: str = None) -> Dict[str, Any]:
    # Ensure single node, N devices
    sys_h = cfg.setdefault("system_hierarchy", {})
    sys_h["num_nodes"] = 1
    sys_h["num_devices_per_node"] = int(n_workers)
    # Force DP to use intra-node fabric
    sys_h["dp_inter"] = False

    # Scheduling: DP only, equals number of workers
    sch = cfg.setdefault("scheduling_param", {})
    sch["auto"] = False
    sch["dp"] = int(n_workers)
    sch["lp"] = 1
    sch["t"] = "RC"
    sch["kp1"] = 1
    sch["kp2"] = 1

    # Network backend selection
    nb = cfg.setdefault("network_backend", {})
    nb["model"] = str(backend)
    # Optionally override intra topology for benchmarking
    if intra_topo is not None:
        nt = cfg.setdefault("network_topology", {})
        nt["intra_node"] = str(intra_topo)
    return cfg


def powers_of_two_factors(n: int) -> List[int]:
    vals = []
    p = 1
    while p <= n:
        if n % p == 0:
            vals.append(p)
        p *= 2
    return vals


def run_once(hw_cfg_path: str, model_cfg_path: str, out_dir: str, log_path: str = None, env_extra: Dict[str, str] = None) -> Tuple[str, Dict[str, float]]:
    """Run the perf script and return (stdout, times_dict). times_dict may include:
    total, reduction, reduction_fwd, reduction_bwd
    """
    cmd = [
        "uv",
        "run",
        "run_perf_llmtest.py",
        "--hardware_config",
        hw_cfg_path,
        "--model_config",
        model_cfg_path,
        "--output_dir",
        out_dir,
    ]
    env = os.environ.copy()
    env["ASTRA_TEST"] = "1"
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    out = proc.stdout
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(out)
    if proc.returncode != 0:
        raise RuntimeError(f"runner failed rc={proc.returncode}; log={log_path}")
    # Parse metrics from stdout
    def _find(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else float("nan")
    total = _find(r"Total time:\s*([0-9.eE+\-]+)")
    red_total_new = _find(r"Reduction time:\s*([0-9.eE+\-]+)")
    red_fwd = _find(r"Reduction FWD time:\s*([0-9.eE+\-]+)")
    red_bwd = _find(r"Reduction BWD time:\s*([0-9.eE+\-]+)")
    red_old = _find(r"Reduction_time:\s*([0-9.eE+\-]+)")
    reduction = red_total_new if not math.isnan(red_total_new) else red_old
    return out, {"total": total, "reduction": reduction, "reduction_fwd": red_fwd, "reduction_bwd": red_bwd}


def make_run_configs(sig: str, M: int, K: int, N: int, n_workers: int, t: str, kp1: int, kp2: int, backend: str) -> Tuple[str, str, str, str]:
    """Create per-run HW and MODEL YAMLs and output dir; return their paths."""
    run_dir = os.path.join(TMP_ROOT, sig)
    os.makedirs(run_dir, exist_ok=True)
    hw_dst = os.path.join(run_dir, "a100_80GB.yaml")
    model_dst = os.path.join(run_dir, "GEMM.yaml")
    out_dir = os.path.join(run_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    # Per-run Astra cache directory (copy from global as baseline)
    per_cache_dir = os.path.join(run_dir, "astra_cache")
    os.makedirs(per_cache_dir, exist_ok=True)
    global_cache = os.path.join(ROOT, "astra_cache", "cache.json")
    if os.path.exists(global_cache):
        try:
            shutil.copy2(global_cache, os.path.join(per_cache_dir, "cache.json"))
        except Exception:
            pass

    base_hw = _load_yaml(HW_PATH)
    base_model = _load_yaml(MODEL_PATH)
    hw_cfg = set_hw_params_cfg(base_hw, n_workers, t, kp1, kp2, backend)
    model_cfg = set_model_gemm_cfg(base_model, M, K, N)
    _save_yaml(hw_dst, hw_cfg)
    _save_yaml(model_dst, model_cfg)
    return hw_dst, model_dst, out_dir, per_cache_dir


def run_task_gemm(task: Dict[str, Any]) -> Dict[str, Any]:
    M = task["M"]; K = task["K"]; N = task["N"]
    n_workers = task["n_workers"]; t = task["t"]; kp1 = task["kp1"]; kp2 = task["kp2"]; backend = task["backend"]
    sig = f"M{M}_K{K}_N{N}_W{n_workers}_t{t}_kp1_{kp1}_kp2_{kp2}_b_{backend}"
    hw_cfg_path, model_cfg_path, out_dir, per_cache_dir = make_run_configs(sig, M, K, N, n_workers, t, kp1, kp2, backend)
    log_path = os.path.join(TMP_ROOT, sig, "run.log")
    env_extra = {"ASTRA_CACHE_DIR": per_cache_dir}
    _, times = run_once(hw_cfg_path, model_cfg_path, out_dir, log_path=log_path, env_extra=env_extra)
    return {
        "M": M, "K": K, "N": N,
        "n_workers": n_workers,
        "t": t,
        "kp1": kp1, "kp2": kp2,
        "backend": backend,
        "reduction_time": times.get("reduction", float("nan")),
        "total_time": times.get("total", float("nan")),
        "reduction_fwd": times.get("reduction_fwd", float("nan")),
        "reduction_bwd": times.get("reduction_bwd", float("nan")),
        "sig": sig,
    }

def run_task_llm(task: Dict[str, Any]) -> Dict[str, Any]:
    variant = task["variant"]
    params = task["llm_params"]
    n_workers = task["n_workers"]
    backend = task["backend"]
    sig = f"LLM_{variant}_W{n_workers}_b_{backend}"
    hw_cfg_path, model_cfg_path, out_dir, per_cache_dir = make_run_configs_llm(sig, params, n_workers, backend)
    log_path = os.path.join(TMP_ROOT, sig, "run.log")
    env_extra = {"ASTRA_CACHE_DIR": per_cache_dir}
    _, times = run_once(hw_cfg_path, model_cfg_path, out_dir, log_path=log_path, env_extra=env_extra)
    return {
        "LLM": variant,
        "n_workers": n_workers,
        "backend": backend,
        "reduction_time": times.get("reduction", float("nan")),
        "total_time": times.get("total", float("nan")),
        "reduction_fwd": times.get("reduction_fwd", float("nan")),
        "reduction_bwd": times.get("reduction_bwd", float("nan")),
        "sig": sig,
    }


def set_model_llm_cfg(cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    mp = cfg.setdefault("model_param", {})
    for k, v in params.items():
        mp[k] = v
    # ensure mode present
    if "mode" not in mp:
        mp["mode"] = "LLM"
    return cfg


def make_run_configs_llm(sig: str, model_params: Dict[str, Any], n_workers: int, backend: str, intra_topo: str = None) -> Tuple[str, str, str, str]:
    run_dir = os.path.join(TMP_ROOT, sig)
    os.makedirs(run_dir, exist_ok=True)
    hw_dst = os.path.join(run_dir, "a100_80GB.yaml")
    model_dst = os.path.join(run_dir, "LLM.yaml")
    out_dir = os.path.join(run_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    # Per-run Astra cache directory (copy from global as baseline)
    per_cache_dir = os.path.join(run_dir, "astra_cache")
    os.makedirs(per_cache_dir, exist_ok=True)
    global_cache = os.path.join(ROOT, "astra_cache", "cache.json")
    if os.path.exists(global_cache):
        try:
            shutil.copy2(global_cache, os.path.join(per_cache_dir, "cache.json"))
        except Exception:
            pass

    base_hw = _load_yaml(HW_PATH)
    base_model = _load_yaml(LLM_MODEL_PATH)
    base_hw = set_hw_params_cfg_llm(base_hw, n_workers, backend, intra_topo=intra_topo)
    base_model = set_model_llm_cfg(base_model, model_params)
    _save_yaml(hw_dst, base_hw)
    _save_yaml(model_dst, base_model)
    return hw_dst, model_dst, out_dir, per_cache_dir


def plot_heatmap_sizes_kp1(matrix: np.ndarray, kp_vals: List[int], size_labels: List[str], title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    # matrix shape: [num_sizes, len(kp_vals)]
    # Use uniform color scale across plots
    vmin = -1.0
    vmax = 1.0
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        xticklabels=[str(k) for k in kp_vals],
        yticklabels=size_labels,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Relative delta (Astra - Analytical)/Analytical"},
    )
    ax.set_xlabel("kp1")
    ax.set_ylabel("GEMM size (M,K,N)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _clear_astra_cache(root: str) -> None:
    cache_path = os.path.join(root, "astra_cache", "cache.json")
    lock_path = cache_path + ".lock"
    try:
        if os.path.exists(cache_path):
            os.remove(cache_path)
    except Exception:
        pass
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass
    # Recreate empty cache.json to avoid missing-file surprises
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({}, f)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Astra vs Analytical sweep for GEMM/LLM")
    parser.add_argument("-c", "--clear-cache", action="store_true", help="Clear AstraSim cache before running")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep per-run temporary logs/configs for debugging")
    parser.add_argument("-b", "--bench", choices=["g", "l", "b"], default="g", help="Benchmark: g=GEMM, l=LLM, b=both")
    parser.add_argument("--mode", choices=["compare", "bench"], default="compare", help="compare: analytical vs Astra; bench: Astra only comparing network topologies (LLM only)")
    args = parser.parse_args()

    if args.clear_cache:
        _clear_astra_cache(ROOT)

    backends = ["analytical", "astra"]

    # -----------------
    # GEMM sweep (old)
    # -----------------
    if args.bench in ("g", "b"):
        # Problem sizes (k = 1024)
        k = 1024
        sizes = [
            (16 * k, 16 * k, 16 * k),
            (64 * k, 8 * k, 8 * k),
            (128 * k, 4 * k, 8 * k),
            (64 * k, 64 * k, 1 * k),
            (4 * k, 512 * k, 2 * k),
        ]
        n_workers_list = [4, 32]
        ts = ["RC", "CR"]

        records = []  # GEMM records

        def _is_power_of_two(x: int) -> bool:
            return x > 0 and (x & (x - 1)) == 0

        total_runs = 0
        for n in n_workers_list:
            base = len(powers_of_two_factors(n))
            extra = 0 if _is_power_of_two(n) else 1
            total_runs += (base + extra) * len(ts) * len(backends)
        total_runs *= len(sizes)

        tasks: List[Dict[str, Any]] = []
        for M, K, N in sizes:
            for n_workers in n_workers_list:
                kp_vals = powers_of_two_factors(n_workers)
                if n_workers not in kp_vals:
                    kp_vals.append(n_workers)
                if 1 not in kp_vals:
                    kp_vals.append(1)
                kp_vals = sorted({k for k in kp_vals if n_workers % k == 0})
                for t in ts:
                    for kp1 in kp_vals:
                        kp2 = n_workers // kp1
                        for backend in backends:
                            tasks.append({
                                "M": M, "K": K, "N": N,
                                "n_workers": n_workers,
                                "t": t,
                                "kp1": kp1, "kp2": kp2,
                                "backend": backend,
                            })

        JOBS = min(16, os.cpu_count() or 4)
        with tqdm(total=len(tasks), desc="GEMM runs", unit="run") as pbar:
            futures = []
            try:
                with ThreadPoolExecutor(max_workers=JOBS) as ex:
                    futures = [ex.submit(run_task_gemm, task) for task in tasks]
                    for fut in as_completed(futures):
                        rec = fut.result()
                        records.append(rec)
                        pbar.update(1)
            finally:
                if not args.keep_tmp:
                    try:
                        shutil.rmtree(TMP_ROOT)
                    except Exception:
                        pass

        # Merge per-run caches back into global cache
        try:
            global_cache_path = os.path.join(ROOT, "astra_cache", "cache.json")
            os.makedirs(os.path.dirname(global_cache_path), exist_ok=True)
            if os.path.exists(global_cache_path):
                with open(global_cache_path, "r") as f:
                    global_cache = json.load(f)
            else:
                global_cache = {}
            if os.path.exists(TMP_ROOT):
                for sig_dir in os.listdir(TMP_ROOT):
                    per_cache_path = os.path.join(TMP_ROOT, sig_dir, "astra_cache", "cache.json")
                    if os.path.exists(per_cache_path):
                        try:
                            with open(per_cache_path, "r") as f:
                                local_cache = json.load(f)
                            for k, v in local_cache.items():
                                if k not in global_cache:
                                    global_cache[k] = v
                        except Exception:
                            pass
            with open(global_cache_path, "w") as f:
                json.dump(global_cache, f, indent=2)
        except Exception:
            pass

        # Heatmaps and report (unchanged behavior)
        size_labels = [f"({M},{K},{N})" for (M, K, N) in sizes]
        for n_workers in n_workers_list:
            kp_vals = powers_of_two_factors(n_workers)
            if n_workers not in kp_vals:
                kp_vals.append(n_workers)
            if 1 not in kp_vals:
                kp_vals.append(1)
            kp_vals = sorted({k for k in kp_vals if n_workers % k == 0})
            for t in ts:
                metric_fields = [
                    ("reduction_time", "reduction"),
                    ("total_time", "total"),
                    ("reduction_fwd", "reduction_fwd"),
                    ("reduction_bwd", "reduction_bwd"),
                ]
                for field, tag in metric_fields:
                    mat = np.full((len(sizes), len(kp_vals)), np.nan, dtype=float)
                    for si, (M, K, N) in enumerate(sizes):
                        for kj, kp1 in enumerate(kp_vals):
                            kp2 = n_workers // kp1
                            anal = next((r.get(field, float("nan")) for r in records if r["M"] == M and r["K"] == K and r["N"] == N and r["n_workers"] == n_workers and r["t"] == t and r["kp1"] == kp1 and r["kp2"] == kp2 and r["backend"] == "analytical"), float("nan"))
                            astr = next((r.get(field, float("nan")) for r in records if r["M"] == M and r["K"] == K and r["N"] == N and r["n_workers"] == n_workers and r["t"] == t and r["kp1"] == kp1 and r["kp2"] == kp2 and r["backend"] == "astra"), float("nan"))
                            if not (math.isnan(anal) or anal == 0.0 or math.isnan(astr)):
                                mat[si, kj] = (astr - anal) / anal
                    if np.isnan(mat).all():
                        continue
                    title = f"Rel. delta ({tag}) — workers={n_workers}, t={t}"
                    out_path = os.path.join(OUT_DIR, f"heatmap_{tag}_W{n_workers}_t{t}.png")
                    plot_heatmap_sizes_kp1(mat, kp_vals, size_labels, title, out_path)
                    if n_workers == 32:
                        txt_path = os.path.join(OUT_DIR, f"heatmap_{tag}_W{n_workers}_t{t}.txt")
                        try:
                            with open(txt_path, "w") as f:
                                f.write("size/kp1," + ",".join(str(k) for k in kp_vals) + "\n")
                                for si, label in enumerate(size_labels):
                                    row_vals = []
                                    for kj in range(len(kp_vals)):
                                        v = mat[si, kj]
                                        row_vals.append("nan" if np.isnan(v) else f"{v:.6f}")
                                    f.write(label + "," + ",".join(row_vals) + "\n")
                        except Exception:
                            pass

        import pandas as pd
        df = pd.DataFrame(records)
        df_a = df[df.backend == "analytical"].rename(columns={"reduction_time": "rt_analytical"})
        df_s = df[df.backend == "astra"].rename(columns={"reduction_time": "rt_astra"})
        join_cols = ["M", "K", "N", "n_workers", "t", "kp1", "kp2"]
        merged = pd.merge(df_a[join_cols + ["rt_analytical"]], df_s[join_cols + ["rt_astra"]], on=join_cols, how="inner")
        merged["rel_delta"] = (merged["rt_astra"] - merged["rt_analytical"]) / merged["rt_analytical"]
        size_grp = merged.groupby(["M", "K", "N"])['rel_delta'].mean().reset_index()
        rc_grp = merged[merged['t'] == 'RC']['rel_delta'].mean()
        cr_grp = merged[merged['t'] == 'CR']['rel_delta'].mean()
        report_path = os.path.join(OUT_DIR, "report.txt")
        with open(report_path, "w") as f:
            f.write("Astra vs Analytical Reduction Time — Relative Delta Report\n")
            f.write("(delta = (astra - analytical) / analytical)\n\n")
            f.write("Per-size averages:\n")
            for _, row in size_grp.iterrows():
                f.write(f"  size=({int(row['M'])},{int(row['K'])},{int(row['N'])}): avg rel delta = {row['rel_delta']:.6f}\n")
            f.write("\n")
            f.write(f"RC average rel delta: {rc_grp:.6f}\n")
            f.write(f"CR average rel delta: {cr_grp:.6f}\n")
        print(f"Wrote GEMM heatmaps and report to: {OUT_DIR}")

    # --------------
    # LLM sweep (compare mode)
    # --------------
    if args.mode == "compare" and args.bench in ("l", "b"):
        # Variants per your spec
        llm_variants: Dict[str, Dict[str, Any]] = {
            "1.3B": {
                "mode": "LLM",
                "batch_size": 128,
                "seq_len": 4096,
                "hidden_dim": 2048,
                "num_heads": 16,
                "ffn_dim": None,
                "ffn_mult": 4.0,
                "vocab_size": 32000,
                "num_layers": 24,
                "n_tokens": 1000000000000,
                "communication_time": 0,
                "N_PP": 1,
            },
            "2B": {
                "mode": "LLM",
                "batch_size": 96,
                "seq_len": 4096,
                "hidden_dim": 2560,
                "num_heads": 20,
                "ffn_dim": None,
                "ffn_mult": 4.0,
                "vocab_size": 32000,
                "num_layers": 24,
                "n_tokens": 1000000000000,
                "communication_time": 0,
                "N_PP": 1,
            },
            "3B": {
                "mode": "LLM",
                "batch_size": 64,
                "seq_len": 4096,
                "hidden_dim": 3072,
                "num_heads": 24,
                "ffn_dim": None,
                "ffn_mult": 4.0,
                "vocab_size": 32000,
                "num_layers": 28,
                "n_tokens": 1000000000000,
                "communication_time": 0,
                "N_PP": 1,
            },
            "3.8B": {
                "mode": "LLM",
                "batch_size": 32,
                "seq_len": 2048,
                "hidden_dim": 3584,
                "num_heads": 28,
                "ffn_dim": None,
                "ffn_mult": 4.0,
                "vocab_size": 32000,
                "num_layers": 24,
                "n_tokens": 1000000000000,
                "communication_time": 0,
                "N_PP": 1,
            },
        }

        n_workers_list = [8, 16, 32, 64]
        llm_records: List[Dict[str, Any]] = []

        tasks_llm: List[Dict[str, Any]] = []
        for variant, params in llm_variants.items():
            for n_workers in n_workers_list:
                for backend in backends:
                    tasks_llm.append({
                        "variant": variant,
                        "llm_params": params,
                        "n_workers": n_workers,
                        "backend": backend,
                    })

        JOBS = min(16, os.cpu_count() or 4)
        with tqdm(total=len(tasks_llm), desc="LLM runs", unit="run") as pbar:
            futures = []
            try:
                with ThreadPoolExecutor(max_workers=JOBS) as ex:
                    futures = [ex.submit(run_task_llm, task) for task in tasks_llm]
                    for fut in as_completed(futures):
                        rec = fut.result()
                        llm_records.append(rec)
                        pbar.update(1)
            finally:
                if not args.keep_tmp:
                    try:
                        shutil.rmtree(TMP_ROOT)
                    except Exception:
                        pass

        # Persist CSV and simple report with relative deltas
        import pandas as pd
        df = pd.DataFrame(llm_records)
        csv_path = os.path.join(OUT_DIR, "llm_results.csv")
        try:
            df.to_csv(csv_path, index=False)
        except Exception:
            pass

        # Relative delta per variant and n_workers for both total and reduction
        lines = []
        for metric in ("total", "reduction"):
            df_a = df[df.backend == "analytical"].rename(columns={f"{metric}_time": f"{metric}_analytical", metric: f"{metric}_analytical"})
            df_s = df[df.backend == "astra"].rename(columns={f"{metric}_time": f"{metric}_astra", metric: f"{metric}_astra"})
            join_cols = ["LLM", "n_workers"]
            merged = pd.merge(df_a[join_cols + [f"{metric}_analytical"]], df_s[join_cols + [f"{metric}_astra"]], on=join_cols, how="inner")
            merged["rel_delta"] = (merged[f"{metric}_astra"] - merged[f"{metric}_analytical"]) / merged[f"{metric}_analytical"]
            lines.append(f"== Metric: {metric} ==\n")
            for variant in llm_variants.keys():
                sub = merged[merged["LLM"] == variant]
                if sub.empty:
                    continue
                lines.append(f"Variant {variant}:\n")
                for n in n_workers_list:
                    row = sub[sub["n_workers"] == n]
                    if row.empty:
                        continue
                    rel = float(row["rel_delta"].iloc[0])
                    lines.append(f"  workers={n}: rel_delta={rel:.6f}\n")
            lines.append("\n")
        rep_path = os.path.join(OUT_DIR, "llm_report.txt")
        try:
            with open(rep_path, "w") as f:
                f.writelines(lines)
        except Exception:
            pass
        print(f"Wrote LLM results to: {OUT_DIR}")

        # Heatmaps: variants (rows) x workers (cols), relative delta (astra-analytical)/analytical
        def plot_heatmap_variants_workers(matrix: np.ndarray, worker_labels: List[int], variant_labels: List[str], title: str, out_path: str) -> None:
            plt.figure(figsize=(10, 6))
            vmin = -1.0
            vmax = 1.0
            ax = sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                xticklabels=[str(w) for w in worker_labels],
                yticklabels=variant_labels,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": "Relative delta (Astra - Analytical)/Analytical"},
            )
            ax.set_xlabel("workers (dp)")
            ax.set_ylabel("LLM variant")
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

        # Build matrices and plot for available metrics
        variant_labels = list(llm_variants.keys())
        # Map requested metric name to record key
        metrics = [
            ("total_time", "llm_total"),
            ("reduction_time", "llm_reduction"),
            ("reduction_fwd", "llm_reduction_fwd"),
            ("reduction_bwd", "llm_reduction_bwd"),
        ]
        for field_key, tag in metrics:
            mat = np.full((len(variant_labels), len(n_workers_list)), np.nan, dtype=float)
            for vi, variant in enumerate(variant_labels):
                for wi, n in enumerate(n_workers_list):
                    # analytical
                    anal = next((r.get(field_key, float("nan"))
                                 for r in llm_records
                                 if r.get("LLM") == variant and r.get("n_workers") == n and r.get("backend") == "analytical"), float("nan"))
                    # astra
                    astr = next((r.get(field_key, float("nan"))
                                 for r in llm_records
                                 if r.get("LLM") == variant and r.get("n_workers") == n and r.get("backend") == "astra"), float("nan"))
                    if not (math.isnan(anal) or anal == 0.0 or math.isnan(astr)):
                        mat[vi, wi] = (astr - anal) / anal
            if np.isnan(mat).all():
                continue
            title = f"LLM rel. delta ({tag})"
            out_path = os.path.join(OUT_DIR, f"heatmap_{tag}.png")
            plot_heatmap_variants_workers(mat, n_workers_list, variant_labels, title, out_path)

    # --------------------
    # LLM benchmarking mode
    # --------------------
    if args.mode == "bench":
        # Variants and workers
        llm_variants: Dict[str, Dict[str, Any]] = {
            "1.3B": {
                "mode": "LLM",
                "batch_size": 128,
                "seq_len": 4096,
                "hidden_dim": 2048,
                "num_heads": 16,
                "ffn_dim": None,
                "ffn_mult": 4.0,
                "vocab_size": 32000,
                "num_layers": 24,
                "n_tokens": 1000000000000,
                "communication_time": 0,
                "N_PP": 1,
            },
            "2B": {
                "mode": "LLM",
                "batch_size": 96,
                "seq_len": 4096,
                "hidden_dim": 2560,
                "num_heads": 20,
                "ffn_dim": None,
                "ffn_mult": 4.0,
                "vocab_size": 32000,
                "num_layers": 24,
                "n_tokens": 1000000000000,
                "communication_time": 0,
                "N_PP": 1,
            },
            "3B": {
                "mode": "LLM",
                "batch_size": 64,
                "seq_len": 4096,
                "hidden_dim": 3072,
                "num_heads": 24,
                "ffn_dim": None,
                "ffn_mult": 4.0,
                "vocab_size": 32000,
                "num_layers": 28,
                "n_tokens": 1000000000000,
                "communication_time": 0,
                "N_PP": 1,
            },
            "3.8B": {
                "mode": "LLM",
                "batch_size": 32,
                "seq_len": 2048,
                "hidden_dim": 3584,
                "num_heads": 28,
                "ffn_dim": None,
                "ffn_mult": 4.0,
                "vocab_size": 32000,
                "num_layers": 24,
                "n_tokens": 1000000000000,
                "communication_time": 0,
                "N_PP": 1,
            },
        }
        variant_labels = list(llm_variants.keys())
        n_workers_list = [8, 16, 32, 64]
        # Include DeepFlow analytical ring as an extra "topology" for comparison
        topologies = ["ring", "fc", "switch", "df_ring"]

        # Run Astra-only across topologies
        bench_records: List[Dict[str, Any]] = []
        tasks_llm: List[Dict[str, Any]] = []
        for variant, params in llm_variants.items():
            for n_workers in n_workers_list:
                for topo in topologies:
                    tasks_llm.append({
                        "variant": variant,
                        "llm_params": params,
                        "n_workers": n_workers,
                        "backend": "astra",
                        "intra_topo": topo,
                    })

        def run_task_llm_bench(task: Dict[str, Any]) -> Dict[str, Any]:
            variant = task["variant"]
            params = task["llm_params"]
            n_workers = task["n_workers"]
            topo = task["intra_topo"]
            # Switch backend: Astra for astra topologies, Analytical for DeepFlow ring
            if topo == "df_ring":
                backend = "analytical"
                topo_override = "ring"
            else:
                backend = "astra"
                topo_override = topo
            sig = f"LLM_BENCH_{variant}_W{n_workers}_topo_{topo}"
            hw_cfg_path, model_cfg_path, out_dir, per_cache_dir = make_run_configs_llm(sig, params, n_workers, backend, intra_topo=topo_override)
            log_path = os.path.join(TMP_ROOT, sig, "run.log")
            env_extra = {"ASTRA_CACHE_DIR": per_cache_dir}
            _, times = run_once(hw_cfg_path, model_cfg_path, out_dir, log_path=log_path, env_extra=env_extra)
            return {
                "LLM": variant,
                "n_workers": n_workers,
                "topology": topo,
                "total_time": times.get("total", float("nan")),
                "reduction_time": times.get("reduction", float("nan")),
            }

        JOBS = min(16, os.cpu_count() or 4)
        with tqdm(total=len(tasks_llm), desc="LLM bench runs", unit="run") as pbar:
            futures = []
            try:
                with ThreadPoolExecutor(max_workers=JOBS) as ex:
                    futures = [ex.submit(run_task_llm_bench, task) for task in tasks_llm]
                    for fut in as_completed(futures):
                        rec = fut.result()
                        bench_records.append(rec)
                        pbar.update(1)
            finally:
                if not args.keep_tmp:
                    try:
                        shutil.rmtree(TMP_ROOT)
                    except Exception:
                        pass

        # Build line plots: one figure per LLM variant per metric; x=workers, lines=topologies
        import pandas as pd
        df = pd.DataFrame(bench_records)
        metric_cols = [
            ("total_time", "llm_bench_total"),
            ("reduction_time", "llm_bench_reduction"),
        ]
        for metric, tag in metric_cols:
            for variant in variant_labels:
                plt.figure(figsize=(8, 5))
                for topo in topologies:
                    yvals = []
                    for n in n_workers_list:
                        row = df[(df["LLM"] == variant) & (df["n_workers"] == n) & (df["topology"] == topo)]
                        yvals.append(float(row[metric].iloc[0]) if not row.empty else float("nan"))
                    label = {
                        "ring": "Astra Ring",
                        "fc": "Astra FullyConnected",
                        "switch": "Astra Switch",
                        "df_ring": "DeepFlow Analytical Ring",
                    }.get(topo, topo)
                    plt.plot(n_workers_list, yvals, marker='o', label=label)
                plt.xlabel("workers (dp)")
                plt.ylabel(metric.replace("_", " "))
                plt.title(f"LLM Astra-only ({metric}) — {variant}")
                plt.legend()
                plt.tight_layout()
                variant_slug = variant.replace('.', '_')
                outp = os.path.join(OUT_DIR, f"{tag}_{variant_slug}.png")
                plt.savefig(outp)
                plt.close()


if __name__ == "__main__":
    main()
