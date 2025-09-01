#!/usr/bin/env python3
import argparse
import time
import numpy as np
from astropy.table import Table,MaskedColumn
from scipy.spatial import cKDTree
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def as_dtype(name):
    return np.float32 if str(name).lower() == "float32" else np.float64


def get_mapped_lists(cfg):
    """
    Returns (deep_bands, wide_bands, deep_errors, wide_errors, use_deep_errors)

    Supported configurations:

    A) Only deep flux + deep error lists:
       - deep_bands, deep_errors provided
       => wide_bands = deep_bands, wide_errors = deep_errors, use_deep_errors = True

    B) Deep flux + wide flux + (wide) error lists, NO deep errors:
       - deep_bands, wide_bands, (wide_errors) provided
       - deep_errors missing or empty
       => deep_errors = None, use_deep_errors = False  (ignore deep errors entirely)

    C) Shared names:
       - bands, errors provided (same for deep & wide)
       => deep_bands = wide_bands = bands
          deep_errors = wide_errors = errors
          use_deep_errors = True

    D) Full explicit:
       - deep_bands, wide_bands, deep_errors, wide_errors provided
       => use_deep_errors = True

    Notes:
      - If only bands are provided (no errors anywhere), deep_errors=wide_errors=None
        and use_deep_errors=False (you must handle SNR & var_sum accordingly).
    """
    # Helper getters
    def _list_or_none(key):
        v = cfg.get(key, None)
        if v is None:
            return None
        return list(v)

    if all(k in cfg for k in ("deep_bands", "wide_bands")):
        deep_bands = list(cfg["deep_bands"])
        wide_bands = list(cfg["wide_bands"])
        deep_errors = _list_or_none("deep_errors")
        wide_errors = _list_or_none("wide_errors")

        if deep_errors is None and wide_errors is None:
            # Bands mapped, no errors anywhere
            use_deep_errors = False
        elif deep_errors is None and wide_errors is not None:
            # Case B: no deep errors; use only wide errors
            use_deep_errors = False
        else:
            # Both error lists present (or at least deep); use deep errors
            use_deep_errors = True
            if wide_errors is None:
                # If deep_errors provided but wide_errors missing, mirror deep->wide
                wide_errors = list(deep_errors)

    else:
        # Shared names or deep-only provided
        bands = _list_or_none("bands")
        errors = _list_or_none("errors")
        deep_bands = _list_or_none("deep_bands") or bands
        deep_errors = _list_or_none("deep_errors") or errors

        if deep_bands is None:
            raise ValueError("You must provide at least 'bands' or 'deep_bands'.")

        # If no wide_bands explicitly, mirror deep->wide
        wide_bands = _list_or_none("wide_bands") or list(deep_bands)

        # Errors logic
        if deep_errors is None:
            # No deep errors at all
            wide_errors = _list_or_none("wide_errors") or None
            use_deep_errors = False
        else:
            # We have deep errors; if no wide errors, mirror
            wide_errors = _list_or_none("wide_errors") or list(deep_errors)
            use_deep_errors = True

    # Length checks where lists exist
    if len(deep_bands) != len(wide_bands):
        raise ValueError("deep_bands and wide_bands must have the same length.")

    if (deep_errors is not None) and (wide_errors is not None):
        if len(deep_errors) != len(wide_errors) or len(deep_errors) != len(deep_bands):
            raise ValueError("Error lists must match band list lengths when provided.")

    return deep_bands, wide_bands, deep_errors, wide_errors, use_deep_errors



def drop_rows_with_masks_or_nonfinite(
    tbl: Table,
    cols,
    also_drop_nonfinite: bool = True,
    *,
    fill_value: float = -1e30,
    cast_dtype = np.float64,
) -> Table:
    """
    1) Drop any row where ANY of `cols` is masked
       (and optionally non-finite if also_drop_nonfinite=True).
    2) De-mask those columns in-place on the returned table:
       replace MaskedColumns with plain ndarrays, filling with `fill_value`.

    Parameters
    ----------
    tbl : astropy.table.Table
        Input table (not modified; a filtered copy is returned).
    cols : list[str]
        Column names to check & then de-mask.
    also_drop_nonfinite : bool, default True
        If True, also drop rows with NaN/Inf in these columns.
    fill_value : float, keyword-only
        Value used to fill any remaining masks when de-masking (finite!).
    cast_dtype : numpy dtype, keyword-only
        Dtype to cast de-masked columns to (default float64).

    Returns
    -------
    astropy.table.Table
        Filtered table with specified columns converted to plain ndarrays.
    """
    if not cols:
        return tbl

    n = len(tbl)
    bad = np.zeros(n, dtype=bool)

    # Pass 1: build bad-row mask
    for name in cols:
        col = tbl[name]

        # Row has mask?
        m = getattr(col, "mask", None)
        if m is not None:
            bad |= np.asarray(m, dtype=bool)

        if also_drop_nonfinite:
            # Convert to array for finiteness test; fill masks with NaN for the test
            arr = np.asanyarray(col)
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            # If it's not float yet, cast a view for the test (no change to table)
            arr = np.asarray(arr, dtype=np.float64)
            bad |= ~np.isfinite(arr)

    kept = (~bad).sum()
    if bad.any():
        print(f"Dropping {bad.sum()} / {n} rows due to masks/non-finite in {len(cols)} columns "
              f"(keeping {kept}).")

    # Filter rows (return a copy)
    out = tbl[~bad]

    # Pass 2: de-mask the specified columns in the filtered table
    for name in cols:
        col = out[name]
        arr = np.asanyarray(col)

        # If masked, fill with finite sentinel
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(fill_value)

        # Ensure finite (replace any lingering NaN/Inf with sentinel)
        arr = np.asarray(arr, dtype=np.float64)
        if not np.isfinite(arr).all():
            arr[~np.isfinite(arr)] = fill_value

        # Cast to desired dtype and replace column
        arr = np.asarray(arr, dtype=cast_dtype)
        out.replace_column(name, arr)

    return out


def main(cfg):
    # ---- Config ----
    deep_path = cfg["deep_fits"]
    wide_path = cfg["wide_fits"]
    out_path = cfg["output_fits"]

    deep_bands, wide_bands, deep_errs, wide_errs, use_deep_errors = get_mapped_lists(cfg)
    n_features = len(deep_bands)

    meta_cols = cfg.get("meta_columns", [])

    # SNR cut columns (default to first deep band/error if not specified)
    snr_band = cfg.get("snr_band", None)
    snr_err_band = cfg.get("snr_err_band", None)
    snr_threshold = float(cfg.get("snr_threshold", False))

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
    used_deep_cols = list(deep_bands) + (list(deep_errs) if deep_errs else [])
    used_wide_cols = list(wide_bands) + (list(wide_errs) if wide_errs else [])
    deep = drop_rows_with_masks_or_nonfinite(deep, used_deep_cols, also_drop_nonfinite=True)
    wide = drop_rows_with_masks_or_nonfinite(wide, used_wide_cols, also_drop_nonfinite=True)

    # ---- SNR cut on DEEP ----
    if snr_threshold:
        print(f"Applying SNR cut: {snr_band}/{snr_err_band} >= {snr_threshold}")
        snr_mask = (deep[snr_band] / deep[snr_err_band]) >= snr_threshold
        deep = deep[snr_mask]
    else:
        None

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
    if use_deep_errors:
        deep_err = np.vstack([deep[col].astype(float_dtype) for col in deep_errs]).T
    else:
        deep_err = np.zeros_like(deep_flux, dtype=float_dtype)

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
