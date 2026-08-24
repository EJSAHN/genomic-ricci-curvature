# -*- coding: utf-8 -*-
"""
Shared utilities for leakage-controlled reference-based QC benchmarking.

All fitted objects in this module are trained only on explicitly supplied
single-source control rows. Evaluation rows are never used for marker
selection, imputation, PCA fitting, source-centroid estimation, or scaling.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


MARKER_FEATURES_FULL = [
    "pair_gain_fraction",
    "best_single_error",
    "heterozygous_marker_fraction",
    "mean_allele_entropy",
    "pca_reconstruction_error",
]
MARKER_FEATURES_NO_PCA = [
    "pair_gain_fraction",
    "best_single_error",
    "heterozygous_marker_fraction",
    "mean_allele_entropy",
]
MAPPING_FEATURES = [
    "mapping_rate_inverse",
    "unique_mapping_rate_inverse",
    "mismatch_rate",
    "discordant_pair_rate",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    return value


def write_json(obj: object, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(obj), fh, indent=2, ensure_ascii=False, allow_nan=False)


def stable_hash(items: Iterable[object]) -> str:
    h = hashlib.sha256()
    for item in sorted(str(x) for x in items):
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_matrix(prefix: Path) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    matrix_path = Path(str(prefix) + "_matrix.npz")
    samples_path = Path(str(prefix) + "_samples.tsv")
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)

    payload = np.load(matrix_path, allow_pickle=False)
    sample_ids = [str(x) for x in payload["sample_ids"].tolist()]
    alt_fraction = payload["alt_fraction"].astype(float)
    depth = payload["depth"].astype(np.uint16)
    samples = pd.read_csv(samples_path, sep="\t")
    observed = samples["sample_id"].astype(str).tolist()
    if sample_ids != observed:
        raise ValueError(f"Sample order mismatch for {prefix}")
    if alt_fraction.shape != depth.shape:
        raise ValueError(f"Matrix/depth shape mismatch for {prefix}")
    if alt_fraction.shape[0] != len(samples):
        raise ValueError(f"Sample count mismatch for {prefix}")
    return alt_fraction, depth, sample_ids, samples


def finite_column_mean(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(matrix, dtype=float)
    finite = np.isfinite(X)
    count = finite.sum(axis=0)
    total = np.where(finite, X, 0.0).sum(axis=0)
    mean = np.zeros(X.shape[1], dtype=float)
    valid = count > 0
    mean[valid] = total[valid] / count[valid]
    mean[~valid] = np.nan
    return mean, count


def finite_column_variance(matrix: np.ndarray, mean: np.ndarray, count: np.ndarray) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    finite = np.isfinite(X)
    centered = np.where(finite, X - mean, 0.0)
    ss = np.square(centered).sum(axis=0)
    var = np.full(X.shape[1], np.nan, dtype=float)
    valid = count > 0
    var[valid] = ss[valid] / count[valid]
    return var


def finite_column_median(matrix: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    med = np.full(X.shape[1], float(fallback), dtype=float)
    for j in range(X.shape[1]):
        vals = X[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            med[j] = float(np.median(vals))
    return med


def choose_markers(
    training_controls: np.ndarray,
    min_control_call_rate: float,
    max_features: int,
) -> np.ndarray:
    controls = np.asarray(training_controls, dtype=float)
    if controls.ndim != 2 or controls.shape[0] < 2:
        return np.asarray([], dtype=int)
    call_rate = np.isfinite(controls).mean(axis=0)
    eligible = np.where(call_rate >= float(min_control_call_rate))[0]
    if len(eligible) == 0:
        return eligible.astype(int)

    subset = controls[:, eligible]
    mean, count = finite_column_mean(subset)
    variance = finite_column_variance(subset, mean, count)
    valid = np.isfinite(mean) & np.isfinite(variance)
    eligible = eligible[valid]
    variance = variance[valid]
    if len(eligible) == 0:
        return eligible.astype(int)

    priority = call_rate[eligible] * (variance + 1e-5)
    order = np.argsort(priority, kind="mergesort")[::-1]
    return eligible[order[: int(max_features)]].astype(int)


def robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return center, scale


def apply_robust_z(values: Sequence[float], center: float, scale: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    z = (x - float(center)) / float(scale)
    z[~np.isfinite(z)] = 0.0
    return z


def evaluate_binary(y: Sequence[int], score: Sequence[float]) -> Dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    mask = np.isfinite(s_arr)
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    prevalence = float(np.mean(y_arr)) if len(y_arr) else float("nan")
    if len(np.unique(y_arr)) < 2:
        return {
            "n": int(len(y_arr)),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
            "prevalence": prevalence,
        }

    thresholds = np.unique(s_arr)
    best_f1 = -1.0
    best_threshold = float("nan")
    for threshold in thresholds:
        pred = (s_arr >= threshold).astype(int)
        f1 = float(f1_score(y_arr, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return {
        "n": int(len(y_arr)),
        "roc_auc": float(roc_auc_score(y_arr, s_arr)),
        "average_precision": float(average_precision_score(y_arr, s_arr)),
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "prevalence": prevalence,
    }


def rank_percentile_high(values: Sequence[float]) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)


def spearman_safe(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    if np.std(xa[mask]) < 1e-15 or np.std(ya[mask]) < 1e-15:
        return float("nan"), float("nan")
    rho, p = spearmanr(xa[mask], ya[mask])
    return float(rho), float(p)


def jaccard_top_n(a: pd.DataFrame, b: pd.DataFrame, score_a: str, score_b: str, n: int = 5) -> float:
    sa = set(a.nlargest(n, score_a)["sample_id"].astype(str))
    sb = set(b.nlargest(n, score_b)["sample_id"].astype(str))
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else float("nan")


def pairwise_overlap_distance(
    target: np.ndarray,
    candidate: np.ndarray,
    min_overlap: int,
) -> tuple[float, int]:
    mask = np.isfinite(target) & np.isfinite(candidate)
    n = int(mask.sum())
    if n < int(min_overlap):
        return float("nan"), n
    return float(np.sqrt(np.mean(np.square(target[mask] - candidate[mask])))), n


def best_single_and_pair_reconstruction(
    target: np.ndarray,
    control_matrix: np.ndarray,
    control_names: Sequence[str],
    min_overlap: int,
) -> Dict[str, object]:
    eligible = list(range(len(control_names)))
    best_single_error = float("inf")
    best_single_name = ""
    best_single_overlap = 0
    for idx in eligible:
        error, overlap = pairwise_overlap_distance(
            target, control_matrix[idx], min_overlap=min_overlap
        )
        if np.isfinite(error) and error < best_single_error:
            best_single_error = error
            best_single_name = str(control_names[idx])
            best_single_overlap = overlap

    best_pair_error = float("inf")
    best_pair_a = ""
    best_pair_b = ""
    best_weight = float("nan")
    best_pair_overlap = 0
    for ii, a_idx in enumerate(eligible):
        a = control_matrix[a_idx]
        for b_idx in eligible[ii + 1 :]:
            b = control_matrix[b_idx]
            mask = np.isfinite(target) & np.isfinite(a) & np.isfinite(b)
            overlap = int(mask.sum())
            if overlap < int(min_overlap):
                continue
            y = target[mask]
            av = a[mask]
            bv = b[mask]
            direction = av - bv
            denom = float(np.dot(direction, direction))
            if denom <= 1e-15:
                weight = 0.5
            else:
                weight = float(np.clip(np.dot(y - bv, direction) / denom, 0.0, 1.0))
            pred = weight * av + (1.0 - weight) * bv
            error = float(np.sqrt(np.mean(np.square(y - pred))))
            if error < best_pair_error:
                best_pair_error = error
                best_pair_a = str(control_names[a_idx])
                best_pair_b = str(control_names[b_idx])
                best_weight = weight
                best_pair_overlap = overlap

    if not np.isfinite(best_single_error):
        best_single_error = float("nan")
    if not np.isfinite(best_pair_error):
        best_pair_error = float("nan")
    if np.isfinite(best_single_error) and np.isfinite(best_pair_error):
        gain = best_single_error - best_pair_error
        gain_fraction = gain / max(best_single_error, 1e-12)
    else:
        gain = float("nan")
        gain_fraction = float("nan")
    return {
        "best_single_error": best_single_error,
        "best_single_name": best_single_name,
        "best_pair_error": best_pair_error,
        "best_pair_a": best_pair_a,
        "best_pair_b": best_pair_b,
        "best_pair_weight_a": best_weight,
        "pair_gain": gain,
        "pair_gain_fraction": gain_fraction,
        "single_overlap": int(best_single_overlap),
        "pair_overlap": int(best_pair_overlap),
    }


def safe_source_ids(samples: pd.DataFrame) -> list[str]:
    if "parents" in samples.columns:
        vals = samples["parents"].astype(str).tolist()
        out = []
        for sample_id, raw in zip(samples["sample_id"].astype(str), vals):
            raw = raw.strip()
            if raw and raw.lower() != "nan" and ";" not in raw and "," not in raw:
                out.append(raw)
            else:
                out.append(sample_id.split("_")[-1])
        return out
    return [str(x).split("_")[-1] for x in samples["sample_id"].astype(str)]


def build_source_centroids(
    training_matrix: np.ndarray,
    training_samples: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    source_ids = safe_source_ids(training_samples)
    ordered = sorted(set(source_ids))
    centroids: list[np.ndarray] = []
    for source in ordered:
        idx = [i for i, value in enumerate(source_ids) if value == source]
        centroids.append(finite_column_median(training_matrix[idx], fallback=float("nan")))
    return np.vstack(centroids), ordered


@dataclass
class MarkerModel:
    selected: np.ndarray
    imputation_median: np.ndarray
    pca: PCA
    pca_components: int
    source_centroids: np.ndarray
    source_names: list[str]
    selected_hash: str


def fit_marker_model(
    training_matrix_full: np.ndarray,
    training_samples: pd.DataFrame,
    min_control_call_rate: float,
    max_features: int,
    max_pca_components: int,
    min_overlap: int,
) -> MarkerModel:
    selected = choose_markers(
        training_matrix_full,
        min_control_call_rate=min_control_call_rate,
        max_features=max_features,
    )
    if len(selected) < int(min_overlap):
        raise RuntimeError(
            f"Only {len(selected)} informative training markers; minimum is {min_overlap}"
        )
    train = np.asarray(training_matrix_full[:, selected], dtype=float)
    medians = finite_column_median(train, fallback=0.0)
    train_imp = np.where(np.isfinite(train), train, medians)
    n_components = int(
        min(
            int(max_pca_components),
            train_imp.shape[0] - 1,
            train_imp.shape[1],
        )
    )
    if n_components < 1:
        raise RuntimeError("PCA requires at least one component")
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(train_imp)
    centroids, names = build_source_centroids(train, training_samples)
    return MarkerModel(
        selected=selected,
        imputation_median=medians,
        pca=pca,
        pca_components=n_components,
        source_centroids=centroids,
        source_names=names,
        selected_hash=stable_hash(selected.tolist()),
    )


def pca_reconstruction_error(model: MarkerModel, target_full: np.ndarray) -> np.ndarray:
    target = np.asarray(target_full[:, model.selected], dtype=float)
    target_imp = np.where(np.isfinite(target), target, model.imputation_median)
    transformed = model.pca.transform(target_imp)
    reconstructed = model.pca.inverse_transform(transformed)
    return np.sqrt(np.mean(np.square(target_imp - reconstructed), axis=1))


def mapping_lookup(mapping: pd.DataFrame) -> pd.DataFrame:
    out = mapping.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    if out["sample_id"].duplicated().any():
        raise ValueError("Mapping table contains duplicated sample_id values after filtering")
    return out.set_index("sample_id", drop=False)


def score_raw_features(
    *,
    model: MarkerModel,
    target_matrix_full: np.ndarray,
    target_samples: pd.DataFrame,
    target_mapping: pd.DataFrame,
    min_overlap: int,
) -> pd.DataFrame:
    X = np.asarray(target_matrix_full[:, model.selected], dtype=float)
    pca_error = pca_reconstruction_error(model, target_matrix_full)
    map_by_id = mapping_lookup(target_mapping)

    rows: list[dict[str, object]] = []
    for i, meta in target_samples.reset_index(drop=True).iterrows():
        sample_id = str(meta["sample_id"])
        reconstruction = best_single_and_pair_reconstruction(
            target=X[i],
            control_matrix=model.source_centroids,
            control_names=model.source_names,
            min_overlap=min_overlap,
        )
        if sample_id not in map_by_id.index:
            raise KeyError(f"Mapping metrics missing sample: {sample_id}")
        map_row = map_by_id.loc[sample_id]
        row: dict[str, object] = {
            "sample_id": sample_id,
            "class_label": str(meta.get("class_label", "")),
            "replicate": int(meta.get("replicate", 0)),
            "scenario": str(meta.get("scenario", "")),
            "pattern_id": str(meta.get("pattern_id", "")),
            "n_parents": int(meta.get("n_parents", 0)),
            "parents": str(meta.get("parents", "")),
            "actual_entropy_norm": float(meta.get("actual_entropy_norm", 0.0)),
            "actual_minor_fraction": float(meta.get("actual_minor_fraction", 0.0)),
            "selected_marker_count": int(len(model.selected)),
            "sample_callable_selected": int(np.isfinite(X[i]).sum()),
            "sample_callable_selected_fraction": float(np.isfinite(X[i]).mean()),
            "pca_reconstruction_error": float(pca_error[i]),
            **reconstruction,
        }
        for name in [
            "heterozygous_marker_fraction",
            "mean_allele_entropy",
            "mapping_rate",
            "unique_mapping_rate",
            "mismatch_rate",
            "discordant_pair_rate",
            "proper_pair_rate",
            "both_mapped_pair_rate",
            "mean_mapq",
            "insert_size_median",
            "insert_size_mad",
        ]:
            row[name] = float(map_row.get(name, float("nan")))
        row["mapping_rate_inverse"] = -float(row["mapping_rate"])
        row["unique_mapping_rate_inverse"] = -float(row["unique_mapping_rate"])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_feature_scalers(reference_raw: pd.DataFrame, feature_names: Sequence[str]) -> Dict[str, tuple[float, float]]:
    scalers: Dict[str, tuple[float, float]] = {}
    for name in feature_names:
        scalers[name] = robust_location_scale(
            pd.to_numeric(reference_raw[name], errors="coerce").to_numpy(dtype=float)
        )
    return scalers


def apply_feature_scalers(
    frame: pd.DataFrame,
    scalers: Mapping[str, tuple[float, float]],
    feature_names: Sequence[str],
    prefix: str,
) -> np.ndarray:
    components: list[np.ndarray] = []
    for name in feature_names:
        center, scale = scalers[name]
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        z = apply_robust_z(values, center, scale)
        z = np.clip(z, -8.0, 12.0)
        frame[f"{prefix}_{name}_z"] = z
        components.append(z)
    return np.mean(np.vstack(components).T, axis=1)


def calibrate_and_score(
    reference_raw: pd.DataFrame,
    target_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    ref = reference_raw.copy()
    target = target_raw.copy()

    all_features = sorted(
        set(MARKER_FEATURES_FULL + MARKER_FEATURES_NO_PCA + MAPPING_FEATURES)
    )
    scalers = fit_feature_scalers(ref, all_features)

    ref["marker_qc_raw"] = apply_feature_scalers(
        ref, scalers, MARKER_FEATURES_FULL, "feature"
    )
    target["marker_qc_raw"] = apply_feature_scalers(
        target, scalers, MARKER_FEATURES_FULL, "feature"
    )
    ref["marker_no_pca_raw"] = apply_feature_scalers(
        ref, scalers, MARKER_FEATURES_NO_PCA, "feature_no_pca"
    )
    target["marker_no_pca_raw"] = apply_feature_scalers(
        target, scalers, MARKER_FEATURES_NO_PCA, "feature_no_pca"
    )
    ref["mapping_qc_raw"] = apply_feature_scalers(
        ref, scalers, MAPPING_FEATURES, "mapping_feature"
    )
    target["mapping_qc_raw"] = apply_feature_scalers(
        target, scalers, MAPPING_FEATURES, "mapping_feature"
    )

    composite_scalers: Dict[str, tuple[float, float]] = {}
    for name in ["marker_qc_raw", "marker_no_pca_raw", "mapping_qc_raw"]:
        composite_scalers[name] = robust_location_scale(ref[name].to_numpy(dtype=float))
        center, scale = composite_scalers[name]
        ref[name + "_z"] = apply_robust_z(ref[name], center, scale)
        target[name + "_z"] = apply_robust_z(target[name], center, scale)

    # Prespecified 75% marker / 25% mapping weighting, retained from v1.
    target["reference_qc_crossfit"] = (
        0.75 * target["marker_qc_raw_z"] + 0.25 * target["mapping_qc_raw_z"]
    )
    target["reference_qc_crossfit_no_pca"] = (
        0.75 * target["marker_no_pca_raw_z"] + 0.25 * target["mapping_qc_raw_z"]
    )
    target["reference_marker_crossfit"] = target["marker_qc_raw_z"]
    target["reference_marker_no_pca_crossfit"] = target["marker_no_pca_raw_z"]
    target["reference_mapping_crossfit"] = target["mapping_qc_raw_z"]

    pca_center, pca_scale = robust_location_scale(
        ref["pca_reconstruction_error"].to_numpy(dtype=float)
    )
    target["pca_only_crossfit"] = apply_robust_z(
        target["pca_reconstruction_error"], pca_center, pca_scale
    )

    reconstruction_ref = ref[
        ["pair_gain_fraction", "best_single_error"]
    ].copy()
    reconstruction_scalers = fit_feature_scalers(
        reconstruction_ref, ["pair_gain_fraction", "best_single_error"]
    )
    reconstruction_target = target[
        ["pair_gain_fraction", "best_single_error"]
    ].copy()
    target["reconstruction_only_crossfit"] = apply_feature_scalers(
        reconstruction_target,
        reconstruction_scalers,
        ["pair_gain_fraction", "best_single_error"],
        "reconstruction_feature",
    )

    for name in [
        "reference_qc_crossfit",
        "reference_qc_crossfit_no_pca",
        "reference_marker_crossfit",
        "reference_marker_no_pca_crossfit",
        "reference_mapping_crossfit",
        "pca_only_crossfit",
        "reconstruction_only_crossfit",
    ]:
        target[name + "_rank_percentile"] = rank_percentile_high(target[name])

    metadata = {
        "feature_scalers": {
            key: {"center": float(value[0]), "scale": float(value[1])}
            for key, value in scalers.items()
        },
        "composite_scalers": {
            key: {"center": float(value[0]), "scale": float(value[1])}
            for key, value in composite_scalers.items()
        },
        "pca_only_scaler": {"center": pca_center, "scale": pca_scale},
        "reference_row_count": int(len(ref)),
    }
    return target, metadata


SCORE_VARIANTS = [
    "reference_qc_crossfit",
    "reference_qc_crossfit_no_pca",
    "reference_marker_crossfit",
    "reference_marker_no_pca_crossfit",
    "reference_mapping_crossfit",
    "pca_only_crossfit",
    "reconstruction_only_crossfit",
]
