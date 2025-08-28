#!/usr/bin/env python3
import argparse
import time
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def as_dtype(name):
    return np.float32 if str(name).lower() == "float32" else np.float64


def get_mapped_lists(cfg):
    """
    Supports either:
      - explicit mapping lists: deep_bands, wide_bands, deep_errors, wide_errors
      - or a shared list: bands, errors (same for deep and wide)
    """
    if all(k in cfg for k in ("deep_bands", "wide_bands", "deep_errors", "wide_errors")):
        deep_bands = cfg["deep_bands"]
        wide_bands = cfg["wide_bands"]
        deep_errors = cfg["deep_errors"]
        wide_errors = cfg["wide_errors"]
    else:
        # Fallback: same names for both deep and wide
        bands = cfg["bands"]
        errors = cfg["errors"]
        deep_bands = list(bands)
        wide_bands = list(bands)
        deep_errors = list(errors)
        wide_errors = list(errors)

    if not (len(deep_bands) == len(wide_bands) == len(deep_errors) == len(wide_errors)):
        raise ValueError("deep_bands, wide_bands, deep_errors, wide_errors must all have the same length.")

    return deep_bands, wide_bands, deep_errors, wide_errors


def main(cfg):
    # ---- Config ----
    deep_path = cfg["deep_fits"]
    wide_path = cfg["wide_fits"]
    out_path = cfg["output_fits"]

    deep_bands, wide_bands, deep_errs, wide_errs = get_mapped_lists(cfg)
    n_features = len(deep_bands)

    meta_cols = cfg.get("meta_columns", [])

    # SNR cut columns (default to first deep band/error if not specified)
    snr_band = cfg.get("snr_band", deep_bands[0])
    snr_err_band = cfg.get("snr_err_band", deep_errs[0])
    snr_threshold = float(cfg.get("snr_threshold", 10))

    shuffle = bool(cfg.get("shuffle", True))
    truncate_deep = cfg.get("truncate_deep", None)
    if truncate_deep is not None:
        truncate_deep = int(truncate_deep)

    num_reals = int(cfg.get("num_realisations", 10))
    nn_k = int(cfg.get("nn_k", 25))

    rng_seed = cfg.get("random_seed", None)
    log_chunk = int(cfg.get("log_chunk", 2000))
    float_dtype = as_dtype(cfg.get("dtype_float", "float64"))

    rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))

    # ---- Load data ----
    print("Reading FITS tables...")
    deep = Table.read(deep_path)
    wide = Table.read(wide_path)

    # ---- SNR cut on DEEP ----
    print(f"Applying SNR cut: {snr_band}/{snr_err_band} >= {snr_threshold}")
    snr_mask = (deep[snr_band] / deep[snr_err_band]) >= snr_threshold
    deep = deep[snr_mask]

    # ---- Shuffle + truncate ----
    if shuffle:
        idx = rng.permutation(len(deep))
        deep = deep[idx]
    if truncate_deep is not None:
        deep = deep[:truncate_deep]

    total_rows = len(deep)
    if total_rows == 0:
        raise ValueError("No rows left in deep after SNR/selection!")

    n_out_rows = total_rows * num_reals

    # ---- Build KD-tree on WIDE ----
    print("Building KD-tree on wide fluxes...")
    # (N_wide, F)
    wide_flux = np.vstack([wide[col].astype(float_dtype) for col in wide_bands]).T
    # (N_wide, F)
    wide_err = np.vstack([wide[col].astype(float_dtype) for col in wide_errs]).T
    tree = cKDTree(wide_flux)

    # ---- Prepare DEEP arrays ----
    # (N_deep, F)
    deep_flux = np.vstack([deep[col].astype(float_dtype) for col in deep_bands]).T
    # (N_deep, F)
    deep_err = np.vstack([deep[col].astype(float_dtype) for col in deep_errs]).T

    # ---- Preallocate outputs ----
    flux_out = np.empty((n_out_rows, n_features), dtype=float_dtype)
    err_out = np.empty((n_out_rows, n_features), dtype=float_dtype)

    # Precollect meta values (repeat later)
    meta_values = {m: np.asarray(deep[m]) for m in meta_cols}

    # ---- Main loop ----
    start_time = time.perf_counter()
    last_time = start_time
    last_report_i = 0

    eps = np.finfo(float_dtype).tiny

    print(f"Start sampling: Nd={total_rows}, F={n_features}, k={nn_k}, "
          f"realisations={num_reals}, dtype={flux_out.dtype}")

    for i in range(total_rows):
        # Query vector & errors (F,)
        q_flux = deep_flux[i]
        q_err = deep_err[i]

        # k-NN indices
        dist, idxes = tree.query(q_flux, k=nn_k)
        if nn_k == 1:
            idxes = np.array([idxes], dtype=int)

        # Neighbour arrays: (k, F)
        neigh_flux = wide_flux[idxes]    # (k, F)
        neigh_err = wide_err[idxes]      # (k, F)

        # Likelihood per neighbour using product over features
        # Work in (F, k)
        diff = (neigh_flux.T - q_flux[:, None])                         # (F, k)
        var_sum = (q_err[:, None] ** 2) + (neigh_err.T ** 2)            # (F, k)
        var_sum = np.clip(var_sum, eps, None)
        denom = np.sqrt(var_sum)
        chi2 = (diff ** 2) / var_sum
        lik_single = np.exp(-0.5 * chi2) / denom                        # (F, k)
        likelihood = np.prod(lik_single, axis=0)                        # (k,)

        s = likelihood.sum()
        if not np.isfinite(s) or s <= 0:
            p = np.full(idxes.shape[0], 1.0 / idxes.shape[0], dtype=float_dtype)
        else:
            p = (likelihood / s).astype(float_dtype)

        # Draw one neighbour index according to p
        pick = rng.choice(idxes.shape[0], p=p)
        chosen_err = neigh_err[pick]  # (F,)

        # ---- IMPORTANT: total_err = chosen_err (as requested) ----
        total_err = np.clip(chosen_err, eps, None)

        # Draw realisations around q_flux with width = total_err
        reals = rng.normal(loc=q_flux, scale=total_err, size=(num_reals, n_features))

        # Write into output blocks
        base = i * num_reals
        flux_out[base:base + num_reals, :] = reals
        # Same error row repeated
        err_out[base:base + num_reals, :] = total_err

        # Logging
        if ((i + 1) % log_chunk == 0) or (i + 1 == total_rows):
            now = time.perf_counter()
            done = i + 1
            elapsed = now - start_time
            chunk_elapsed = now - last_time
            rows_chunk = done - last_report_i
            rate = rows_chunk / max(chunk_elapsed, 1e-9)
            remain_rows = total_rows - done
            eta = remain_rows / max(rate, 1e-9)
            print(f"[{done}/{total_rows}] elapsed={elapsed:.1f}s | "
                  f"chunk_rate={rate:.1f} rows/s | ETA={eta:.1f}s",
                  flush=True)
            last_time = now
            last_report_i = done

    # ---- Build final table ----
    print("Building output table...")
    out = Table()

    # Output columns follow the WIDE naming (mock is wide-like)
    for j, name in enumerate(wide_bands):
        out[name] = flux_out[:, j]
    for j, name in enumerate(wide_errs):
        out[name] = err_out[:, j]

    # Repeat meta columns from deep
    if meta_cols:
        reps = num_reals
        for m in meta_cols:
            out[m] = np.repeat(meta_values[m], reps)

    # ---- Write ----
    print(f"Writing: {out_path}")
    out.write(out_path, overwrite=True)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-passband transfer")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
