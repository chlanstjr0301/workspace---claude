from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from .calibration import aggregate_by_burst, empirical_p_values, threshold_from_quantile
from .data import (
    SIGNAL_COLUMNS,
    WindowSet,
    assign_bursts,
    audit_dataframe,
    fold_masks,
    load_csv,
    make_windows,
    temporal_burst_blocks,
)
from .evaluation import alarm_ids, alarm_summary, binary_metrics, poisson_upper_95
from .explain import repair_group_contributions
from .features import FeatureSet, extract_features, feature_indices
from .models import (
    DynamicPCADetector,
    MahalanobisDetector,
    RawRangeDetector,
    make_feature_detector,
)
from .plotting import plot_block_alarms, plot_model_comparison, plot_robustness, plot_score_scatter
from .stress import perturb


MODEL_IDS = ["B0_Range", "B1_IF", "M1_Mahalanobis", "M2_PCA_MSPC", "M3_DPCA"]
MODEL_GROUPS = {
    "B1_IF": ("A", "S", "R", "O"),
    "M1_Mahalanobis": ("S", "R"),
    "M2_PCA_MSPC": ("A", "S", "R", "O"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _feature_view(feature_set: FeatureSet, model_id: str) -> np.ndarray:
    groups = MODEL_GROUPS[model_id]
    idx = feature_indices(feature_set.names, groups)
    return feature_set.values[:, idx]


def _model_input(model_id: str, windows: WindowSet, features: FeatureSet) -> np.ndarray:
    if model_id in {"B0_Range", "M3_DPCA"}:
        return windows.values
    return _feature_view(features, model_id)


def _make_model(model_id: str, cfg: dict, seed: int):
    if model_id == "B0_Range":
        return RawRangeDetector()
    if model_id == "M3_DPCA":
        return DynamicPCADetector(cfg["dpca_variance"])
    return make_feature_detector(
        model_id,
        seed,
        cfg["pca_variance"],
        cfg["iforest"]["n_estimators"],
        cfg["iforest"]["max_samples"],
    )


def _subset_features(features: FeatureSet, mask: np.ndarray) -> FeatureSet:
    return FeatureSet(features.values[mask], features.names, features.groups)


def _fit_fold_model(model_id: str, cfg: dict, seed: int, train_w: WindowSet, train_f: FeatureSet):
    model = _make_model(model_id, cfg, seed)
    model.fit(_model_input(model_id, train_w, train_f))
    return model


def _score(model, model_id: str, windows: WindowSet, features: FeatureSet) -> np.ndarray:
    return model.score(_model_input(model_id, windows, features))


def _observed_hours(df: pd.DataFrame, burst_ids: np.ndarray) -> float:
    return float(df[df["burst_id"].isin(burst_ids)].shape[0] * 0.1 / 3600.0)


def _manifest(root: Path, normal_path: Path, outlier_path: Path, cfg: dict, mode: str) -> dict:
    providers = []
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:
        pass
    torch_info = {"installed": False}
    try:
        import torch

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        }
    except Exception:
        pass
    return {
        "started_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "mode": mode,
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "onnxruntime_providers": providers,
        "torch": torch_info,
        "native_hardware_inventory": {
            "system": "LENOVO 21QNCTO1WW",
            "cpu": "AMD Ryzen AI 7 PRO 350",
            "cores_threads": "8/16",
            "ram_gb": 27.6,
            "gpu": "AMD Radeon 860M Graphics (4 GB reported adapter memory)",
            "npu": "AMD NPU Compute Accelerator Device",
            "execution_choice": "CPU-native for reproducibility; GPU/NPU not required",
        },
        "data": {
            "normal_path": str(normal_path.relative_to(root)),
            "normal_sha256": _sha256(normal_path),
            "outlier_path": str(outlier_path.relative_to(root)),
            "outlier_sha256": _sha256(outlier_path),
        },
        "config": cfg,
    }


def _calibration_rows(model_id: str, fold: int, calibration_scores: np.ndarray, calibration_bursts: np.ndarray, test_scores: np.ndarray) -> list[dict]:
    alpha = 0.005
    references = {
        "C0_window": calibration_scores,
        "C1_burst_max": aggregate_by_burst(calibration_scores, calibration_bursts, "max"),
        "C2_burst_q95": aggregate_by_burst(calibration_scores, calibration_bursts, "q95"),
    }
    rows = []
    for name, reference in references.items():
        p = empirical_p_values(reference, test_scores)
        rows.append({
            "model": model_id,
            "fold": fold,
            "calibration": name,
            "reference_n": len(reference),
            "alpha": alpha,
            "observed_false_positive_fraction": float(np.mean(p <= alpha)),
            "median_p": float(np.median(p)),
        })
    return rows


def _summary(block_cv: pd.DataFrame) -> pd.DataFrame:
    out = block_cv.groupby("model", as_index=False).agg(
        folds=("fold", "count"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_ap=("ap", "mean"),
        mean_auroc=("auroc", "mean"),
        mean_fpr=("fpr", "mean"),
        mean_normal_alarm_events=("normal_alarm_events", "mean"),
        worst_normal_alarm_events=("normal_alarm_events", "max"),
        mean_far_per_hour=("normal_far_per_hour", "mean"),
        worst_far_upper_95=("normal_far_upper_95", "max"),
        mean_anomaly_burst_detection=("anomaly_burst_detection_rate", "mean"),
    )
    return out.sort_values(["worst_normal_alarm_events", "mean_normal_alarm_events", "mean_recall"], ascending=[True, True, False])


def _choose_fault_model(summary: pd.DataFrame) -> str:
    candidates = summary[summary["model"].isin(["B1_IF", "M1_Mahalanobis", "M2_PCA_MSPC", "M3_DPCA"])].copy()
    candidates = candidates.sort_values(
        ["worst_normal_alarm_events", "mean_normal_alarm_events", "mean_recall", "mean_ap"],
        ascending=[True, True, False, False],
    )
    return str(candidates.iloc[0]["model"])


def _window_sensitivity(
    normal: pd.DataFrame, outlier: pd.DataFrame, cfg: dict, model_id: str
) -> pd.DataFrame:
    rows: list[dict] = []
    blocks = temporal_burst_blocks(normal, cfg["n_folds"])
    for window_size in cfg["window_sizes_sensitivity"]:
        nw = make_windows(normal, int(window_size), "normal")
        ow = make_windows(outlier, int(window_size), "outlier")
        nf, of = extract_features(nw), extract_features(ow)
        for fold in range(cfg["n_folds"]):
            train_mask, cal_mask, test_mask = fold_masks(nw, blocks, fold)
            model = _fit_fold_model(
                model_id, cfg, cfg["seed"], nw.take(train_mask), _subset_features(nf, train_mask)
            )
            cal_score = _score(model, model_id, nw.take(cal_mask), _subset_features(nf, cal_mask))
            test_score = _score(model, model_id, nw.take(test_mask), _subset_features(nf, test_mask))
            anomaly_score = _score(model, model_id, ow, of)
            threshold = threshold_from_quantile(cal_score, cfg["calibration_quantile"])
            metrics = binary_metrics(test_score, anomaly_score, threshold)
            rows.append({"model": model_id, "window_size": window_size, "fold": fold, **metrics})
    detail = pd.DataFrame(rows)
    return detail.groupby(["model", "window_size"], as_index=False).agg(
        mean_f1=("f1", "mean"), mean_recall=("recall", "mean"), mean_ap=("ap", "mean"),
        mean_fpr=("fpr", "mean"), worst_fpr=("fpr", "max")
    )


def _feature_ablation(
    normal_w: WindowSet, outlier_w: WindowSet, normal_f: FeatureSet,
    outlier_f: FeatureSet, blocks: dict[int, np.ndarray], cfg: dict
) -> pd.DataFrame:
    rows: list[dict] = []
    all_groups = ("A", "S", "R", "O")
    variants = {"all": all_groups, **{f"without_{g}": tuple(x for x in all_groups if x != g) for g in all_groups}}
    for variant, groups in variants.items():
        idx = feature_indices(normal_f.names, groups)
        for fold in range(cfg["n_folds"]):
            train_mask, cal_mask, test_mask = fold_masks(normal_w, blocks, fold)
            model = make_feature_detector(
                "B1_IF", cfg["seed"], cfg["pca_variance"],
                cfg["iforest"]["n_estimators"], cfg["iforest"]["max_samples"]
            ).fit(normal_f.values[train_mask][:, idx])
            cal_score = model.score(normal_f.values[cal_mask][:, idx])
            test_score = model.score(normal_f.values[test_mask][:, idx])
            anomaly_score = model.score(outlier_f.values[:, idx])
            threshold = threshold_from_quantile(cal_score, cfg["calibration_quantile"])
            rows.append({"variant": variant, "fold": fold, **binary_metrics(test_score, anomaly_score, threshold)})
    detail = pd.DataFrame(rows)
    return detail.groupby("variant", as_index=False).agg(
        mean_f1=("f1", "mean"), mean_recall=("recall", "mean"), mean_ap=("ap", "mean"),
        mean_fpr=("fpr", "mean"), worst_fpr=("fpr", "max")
    ).sort_values("mean_f1", ascending=False)


def _counterfactual_channel_explanation(
    model,
    model_id: str,
    train_w: WindowSet,
    anomaly_w: WindowSet,
    anomaly_f: FeatureSet,
) -> pd.DataFrame:
    baseline = _score(model, model_id, anomaly_w, anomaly_f)
    channel_centers = np.median(train_w.values.reshape(-1, 3), axis=0)
    rows = []
    for channel, name in enumerate(SIGNAL_COLUMNS):
        repaired_values = anomaly_w.values.copy()
        repaired_values[:, :, channel] = channel_centers[channel]
        repaired_w = replace(anomaly_w, values=repaired_values)
        repaired_f = extract_features(repaired_w)
        repaired_score = _score(model, model_id, repaired_w, repaired_f)
        reduction = np.maximum(baseline - repaired_score, 0.0)
        rows.append({
            "explanation_type": "channel_repair",
            "name": name,
            "mean_score_reduction": float(np.mean(reduction)),
            "median_score_reduction": float(np.median(reduction)),
        })
    return pd.DataFrame(rows)


def run_pipeline(root: Path, mode: str = "quick", output_override: Path | None = None) -> dict:
    start = time.time()
    package = root / "plan_c_submission"
    with (package / "config" / "experiments.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    out = output_override or package / "outputs"
    tables = out / "tables"
    figures = out / "figures"
    runs = out / "runs"
    for path in (out, tables, figures, runs):
        path.mkdir(parents=True, exist_ok=True)

    normal_path = root / "data" / "raw" / "press_data_normal.csv"
    outlier_path = root / "data" / "raw" / "press_data_outlier.csv"
    manifest = _manifest(root, normal_path, outlier_path, cfg, mode)

    normal = assign_bursts(load_csv(normal_path), cfg["gap_seconds"])
    outlier = assign_bursts(load_csv(outlier_path), cfg["gap_seconds"])
    audit = pd.DataFrame([
        {"source": "normal", **audit_dataframe(normal, cfg["gap_seconds"])},
        {"source": "outlier", **audit_dataframe(outlier, cfg["gap_seconds"])},
    ])
    audit.to_csv(tables / "data_audit.csv", index=False, encoding="utf-8-sig")

    w = int(cfg["window_size"])
    normal_w = make_windows(normal, w, "normal")
    outlier_w = make_windows(outlier, w, "outlier")
    normal_f = extract_features(normal_w)
    outlier_f = extract_features(outlier_w)
    pd.DataFrame({"feature": normal_f.names}).to_csv(tables / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    blocks = temporal_burst_blocks(normal, cfg["n_folds"])

    seeds = cfg["seeds"] if mode == "full" else [cfg["seed"]]
    block_rows: list[dict] = []
    calibration_rows: list[dict] = []
    fold_cache: dict[tuple[str, int, int], dict] = {}

    for model_id in MODEL_IDS:
        model_seeds = seeds if model_id == "B1_IF" else [cfg["seed"]]
        for seed in model_seeds:
            for fold in range(cfg["n_folds"]):
                train_mask, cal_mask, test_mask = fold_masks(normal_w, blocks, fold)
                train_w, cal_w, test_w = normal_w.take(train_mask), normal_w.take(cal_mask), normal_w.take(test_mask)
                train_f = _subset_features(normal_f, train_mask)
                cal_f = _subset_features(normal_f, cal_mask)
                test_f = _subset_features(normal_f, test_mask)
                model = _fit_fold_model(model_id, cfg, seed, train_w, train_f)
                s_cal = _score(model, model_id, cal_w, cal_f)
                s_test = _score(model, model_id, test_w, test_f)
                s_anom = _score(model, model_id, outlier_w, outlier_f)
                threshold = threshold_from_quantile(s_cal, cfg["calibration_quantile"])
                metrics = binary_metrics(s_test, s_anom, threshold)
                test_block_bursts = blocks[fold]
                normal_hours = _observed_hours(normal, test_block_bursts)
                normal_alarm = alarm_summary(s_test, threshold, test_w.burst_id, cfg["alarm_persistence"], normal_hours)
                anomaly_hours = len(outlier) * 0.1 / 3600.0
                anomaly_alarm = alarm_summary(s_anom, threshold, outlier_w.burst_id, cfg["alarm_persistence"], anomaly_hours)
                row = {
                    "model": model_id,
                    "seed": seed,
                    "fold": fold,
                    "n_train": len(train_w.values),
                    "n_calibration": len(cal_w.values),
                    "n_test_normal": len(test_w.values),
                    "n_test_anomaly": len(outlier_w.values),
                    **metrics,
                    "normal_alarm_events": normal_alarm["alarm_events"],
                    "normal_far_per_hour": normal_alarm["far_per_observed_hour"],
                    "normal_far_upper_95": poisson_upper_95(normal_alarm["alarm_events"], normal_hours),
                    "anomaly_alarm_events": anomaly_alarm["alarm_events"],
                    "anomaly_burst_detection_rate": anomaly_alarm["burst_detection_rate"],
                }
                block_rows.append(row)
                calibration_rows += _calibration_rows(model_id, fold, s_cal, cal_w.burst_id, s_test)
                if seed == cfg["seed"]:
                    fold_cache[(model_id, fold, seed)] = {
                        "model": model,
                        "train_w": train_w,
                        "train_f": train_f,
                        "cal_w": cal_w,
                        "cal_f": cal_f,
                        "test_w": test_w,
                        "test_f": test_f,
                        "s_cal": s_cal,
                        "s_test": s_test,
                        "s_anom": s_anom,
                        "threshold": threshold,
                    }

    block_cv = pd.DataFrame(block_rows)
    block_cv.to_csv(tables / "block_cv.csv", index=False, encoding="utf-8-sig")
    comparison = _summary(block_cv)
    comparison.to_csv(tables / "model_comparison.csv", index=False, encoding="utf-8-sig")
    calibration_audit = pd.DataFrame(calibration_rows)
    calibration_audit.to_csv(tables / "calibration_audit.csv", index=False, encoding="utf-8-sig")
    selected = _choose_fault_model(comparison)

    selection_gates = comparison.copy()
    baseline_worst = int(selection_gates.loc[selection_gates["model"] == "B0_Range", "worst_normal_alarm_events"].iloc[0])
    selection_gates["beats_baseline_f1"] = selection_gates["mean_f1"] > float(
        selection_gates.loc[selection_gates["model"] == "B0_Range", "mean_f1"].iloc[0]
    )
    selection_gates["worst_alarm_gate"] = selection_gates["worst_normal_alarm_events"] <= baseline_worst
    selection_gates["selected"] = selection_gates["model"].eq(selected)
    selection_gates.to_csv(tables / "selection_gates.csv", index=False, encoding="utf-8-sig")

    window_sensitivity = _window_sensitivity(normal, outlier, cfg, selected)
    window_sensitivity.to_csv(tables / "window_sensitivity.csv", index=False, encoding="utf-8-sig")
    feature_ablation = _feature_ablation(normal_w, outlier_w, normal_f, outlier_f, blocks, cfg)
    feature_ablation.to_csv(tables / "feature_ablation.csv", index=False, encoding="utf-8-sig")

    persistence_rows: list[dict] = []
    for persistence in cfg["alarm_persistence_sensitivity"]:
        for fold in range(cfg["n_folds"]):
            cached = fold_cache[(selected, fold, cfg["seed"])]
            _, _, test_mask = fold_masks(normal_w, blocks, fold)
            test_w = normal_w.take(test_mask)
            normal_alarm = alarm_summary(
                cached["s_test"], cached["threshold"], test_w.burst_id, persistence,
                _observed_hours(normal, blocks[fold])
            )
            anomaly_alarm = alarm_summary(
                cached["s_anom"], cached["threshold"], outlier_w.burst_id, persistence,
                len(outlier) * 0.1 / 3600.0
            )
            persistence_rows.append({
                "model": selected, "persistence": persistence, "fold": fold,
                "normal_alarm_events": normal_alarm["alarm_events"],
                "normal_far_per_hour": normal_alarm["far_per_observed_hour"],
                "anomaly_burst_detection_rate": anomaly_alarm["burst_detection_rate"],
            })
    persistence_detail = pd.DataFrame(persistence_rows)
    persistence_summary = persistence_detail.groupby(["model", "persistence"], as_index=False).agg(
        mean_normal_alarm_events=("normal_alarm_events", "mean"),
        worst_normal_alarm_events=("normal_alarm_events", "max"),
        mean_far_per_hour=("normal_far_per_hour", "mean"),
        mean_anomaly_burst_detection=("anomaly_burst_detection_rate", "mean"),
    )
    persistence_summary.to_csv(tables / "persistence_sensitivity.csv", index=False, encoding="utf-8-sig")

    # Robustness on fold 0 under the frozen threshold.
    stress_names = cfg["stress_full"] if mode == "full" else cfg["stress_quick"]
    robustness_rows: list[dict] = []
    train_mask0, _, _ = fold_masks(normal_w, blocks, 0)
    train_channel_std = normal_w.values[train_mask0].reshape(-1, 3).std(axis=0)
    for model_id in MODEL_IDS:
        cached = fold_cache[(model_id, 0, cfg["seed"])]
        original_recall = float(np.mean(cached["s_anom"] > cached["threshold"]))
        robustness_rows.append({"model": model_id, "perturbation": "original", "recall": original_recall, "recall_delta": 0.0})
        for name in stress_names:
            values = perturb(outlier_w.values, name, train_channel_std, cfg["seed"])
            perturbed_w = replace(outlier_w, values=values)
            perturbed_f = extract_features(perturbed_w)
            score = _score(cached["model"], model_id, perturbed_w, perturbed_f)
            recall = float(np.mean(score > cached["threshold"]))
            robustness_rows.append({
                "model": model_id,
                "perturbation": name,
                "recall": recall,
                "recall_delta": recall - original_recall,
            })
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(tables / "robustness.csv", index=False, encoding="utf-8-sig")

    # Explanation audit using selected model, fold 0.
    cached = fold_cache[(selected, 0, cfg["seed"])]
    explanation_frames = [_counterfactual_channel_explanation(
        cached["model"], selected, cached["train_w"], outlier_w, outlier_f
    )]
    if selected in MODEL_GROUPS:
        selected_idx = feature_indices(normal_f.names, MODEL_GROUPS[selected])
        model_features = FeatureSet(
            outlier_f.values[:, selected_idx],
            [normal_f.names[i] for i in selected_idx],
            {g: np.flatnonzero(np.char.startswith(np.asarray([normal_f.names[i] for i in selected_idx], dtype=str), f"{g}_")) for g in MODEL_GROUPS[selected]},
        )
        train_x = _feature_view(cached["train_f"], selected)
        anomaly_x = _feature_view(outlier_f, selected)
        contributions = repair_group_contributions(cached["model"], train_x, anomaly_x, model_features)
        explanation_frames.append(pd.DataFrame([
            {
                "explanation_type": "feature_group_repair",
                "name": group,
                "mean_score_reduction": float(np.mean(value)),
                "median_score_reduction": float(np.median(value)),
            }
            for group, value in contributions.items()
        ]))
    explanation = pd.concat(explanation_frames, ignore_index=True)
    explanation.to_csv(tables / "explanation.csv", index=False, encoding="utf-8-sig")

    # OOF normal + fold-ensemble anomaly predictions for selected fault model and a shift detector.
    normal_prediction_frames: list[pd.DataFrame] = []
    anomaly_fold_rows: list[pd.DataFrame] = []
    shift_idx = feature_indices(normal_f.names, ("A", "O", "Q"))
    for fold in range(cfg["n_folds"]):
        train_mask, cal_mask, test_mask = fold_masks(normal_w, blocks, fold)
        fault_cache = fold_cache[(selected, fold, cfg["seed"])]
        shift = MahalanobisDetector().fit(normal_f.values[train_mask][:, shift_idx])
        shift_cal = shift.score(normal_f.values[cal_mask][:, shift_idx])
        shift_test = shift.score(normal_f.values[test_mask][:, shift_idx])
        shift_anom = shift.score(outlier_f.values[:, shift_idx])
        p_fault_test = empirical_p_values(fault_cache["s_cal"], fault_cache["s_test"])
        p_fault_anom = empirical_p_values(fault_cache["s_cal"], fault_cache["s_anom"])
        p_shift_test = empirical_p_values(shift_cal, shift_test)
        p_shift_anom = empirical_p_values(shift_cal, shift_anom)
        tw = normal_w.take(test_mask)
        normal_prediction_frames.append(pd.DataFrame({
            "source": "normal",
            "fold": fold,
            "original_row_start": tw.row_start,
            "original_row_end": tw.row_end,
            "burst_id": tw.burst_id,
            "time_start": tw.time_start,
            "time_end": tw.time_end,
            "fault_score": fault_cache["s_test"],
            "shift_score": shift_test,
            "p_fault": p_fault_test,
            "p_shift": p_shift_test,
            "label": 0,
        }))
        anomaly_fold_rows.append(pd.DataFrame({
            "fold": fold,
            "window_index": np.arange(len(outlier_w.values)),
            "fault_score": fault_cache["s_anom"],
            "shift_score": shift_anom,
            "p_fault": p_fault_anom,
            "p_shift": p_shift_anom,
        }))
    normal_predictions = pd.concat(normal_prediction_frames, ignore_index=True)
    anomaly_long = pd.concat(anomaly_fold_rows, ignore_index=True)
    anomaly_agg = anomaly_long.groupby("window_index", as_index=False).agg(
        fault_score=("fault_score", "median"),
        shift_score=("shift_score", "median"),
        p_fault=("p_fault", "median"),
        p_shift=("p_shift", "median"),
    )
    anomaly_predictions = pd.DataFrame({
        "source": "outlier",
        "fold": -1,
        "original_row_start": outlier_w.row_start,
        "original_row_end": outlier_w.row_end,
        "burst_id": outlier_w.burst_id,
        "time_start": outlier_w.time_start,
        "time_end": outlier_w.time_end,
        "fault_score": anomaly_agg["fault_score"],
        "shift_score": anomaly_agg["shift_score"],
        "p_fault": anomaly_agg["p_fault"],
        "p_shift": anomaly_agg["p_shift"],
        "label": 1,
    })
    predictions = pd.concat([normal_predictions, anomaly_predictions], ignore_index=True)
    alpha = 1.0 - cfg["calibration_quantile"]
    fault_flag = predictions["p_fault"].to_numpy() <= alpha
    shift_flag = predictions["p_shift"].to_numpy() <= alpha
    predictions["state"] = np.select(
        [~shift_flag & ~fault_flag, shift_flag & ~fault_flag, ~shift_flag & fault_flag, shift_flag & fault_flag],
        ["Green", "Yellow-S", "Yellow-F", "Red"],
        default="Green",
    )
    predictions["fault_risk"] = 1.0 - predictions["p_fault"]
    predictions["shift_risk"] = 1.0 - predictions["p_shift"]
    predictions["selected_fault_model"] = selected
    predictions["alarm_id"] = -1
    for source in ("normal", "outlier"):
        mask = predictions["source"].eq(source).to_numpy()
        predictions.loc[mask, "alarm_id"] = alarm_ids(
            fault_flag[mask], predictions.loc[mask, "burst_id"].to_numpy(), cfg["alarm_persistence"]
        )
    top_channel = explanation[explanation["explanation_type"] == "channel_repair"].sort_values(
        "mean_score_reduction", ascending=False
    ).iloc[0]["name"]
    predictions["top_channel_global"] = top_channel
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")

    error_conditions = predictions.groupby(["source", "state"], as_index=False).agg(
        windows=("label", "size"), median_fault_risk=("fault_risk", "median"),
        median_shift_risk=("shift_risk", "median")
    )
    error_conditions.to_csv(tables / "error_conditions.csv", index=False, encoding="utf-8-sig")

    plot_model_comparison(comparison, figures / "model_comparison.png")
    plot_block_alarms(block_cv[block_cv["seed"] == cfg["seed"]], figures / "normal_block_false_alarms.png")
    plot_robustness(robustness, figures / "robustness_heatmap.png")
    plot_score_scatter(predictions, figures / "shift_fault_plane.png")

    manifest["selected_fault_model"] = selected
    manifest["runtime_seconds"] = round(time.time() - start, 3)
    manifest["outputs"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    with (out / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, default=str)

    status = pd.DataFrame([
        {"component": "Statistical core", "status": "complete", "detail": ", ".join(MODEL_IDS)},
        {"component": "Sensitivity audits", "status": "complete", "detail": "window 5/10/20, feature ablation, persistence 1/3/5"},
        {"component": "LSTM-AE", "status": "reference_only", "detail": "Prior TensorFlow run retained; same-protocol retraining unavailable in Python 3.13"},
        {"component": "VUS-PR", "status": "not_implemented", "detail": "AP and alarm-range metrics retained; avoid unverified dependency"},
        {"component": "Native acceleration", "status": "cpu", "detail": "16 logical threads; AMD GPU/NPU detected but Python providers absent"},
    ])
    status.to_csv(tables / "implementation_status.csv", index=False, encoding="utf-8-sig")

    chosen = comparison.loc[comparison["model"] == selected].iloc[0]
    best_window = window_sensitivity.sort_values(
        ["worst_fpr", "mean_f1"], ascending=[True, False]
    ).iloc[0]
    report = f"""# Plan C execution result

Generated: {pd.Timestamp.now(tz='Asia/Seoul').isoformat()}  
Mode: `{mode}`  
Selected model: **{selected}**

## Frozen-protocol result

- Mean window F1: {chosen['mean_f1']:.4f}
- Mean window recall: {chosen['mean_recall']:.4f}
- Mean AP: {chosen['mean_ap']:.4f}
- Worst normal-block alarm events: {int(chosen['worst_normal_alarm_events'])}
- Mean anomaly-burst detection: {chosen['mean_anomaly_burst_detection']:.4f}
- Frozen operational window: {int(cfg['window_size'])} samples
- Best sensitivity-only window: {int(best_window['window_size'])} samples (not promoted post hoc)

Selection first minimized worst-block false-alarm events, then mean alarm events,
then recall and AP. This is why {selected} is deployed instead of the numerically
highest-F1 model.

## Native hardware decision

The run used the laptop's native CPU (Ryzen AI 7 PRO 350, 8 cores/16 threads).
The Radeon 860M and AMD NPU are present, but the installed Python providers are
CPU-only. The dataset is small enough that the complete statistical run remains fast.

## Scope notes

- Statistical candidates, blocked validation, conformal audit, stress tests,
  window sensitivity, feature ablation, explanation repair, and alarm persistence are complete.
- The earlier TensorFlow LSTM-AE benchmark is kept as reference evidence, but is not
  mixed into this same-protocol table because TensorFlow is unavailable in Python 3.13.
- VUS-PR is intentionally not claimed; AP and burst/alarm metrics are reported instead.
"""
    (out / "RESULTS.md").write_text(report, encoding="utf-8")
    manifest["runtime_seconds"] = round(time.time() - start, 3)
    manifest["outputs"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    with (out / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, default=str)
    return {"selected_model": selected, "output": out, "comparison": comparison, "manifest": manifest}
