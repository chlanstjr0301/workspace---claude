"""Notebook-preserving ablation: add only lag-wise Pearson Current_ACF_PeakStrength.

Reference: reference/소성가공.ipynb, cells 6–9, 27, 30, 36, 38.
All existing features, burst filtering, splits, scaling and LR defaults are
identical. Only the additional input column differs between A and B.
Run locally; neither LSTM training nor inference is performed.
"""
import ast
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.signal import find_peaks
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/logistic_current_acf_pearson_ablation"
NOTEBOOK = ROOT / "reference/소성가공.ipynb"
ACF = "Current_ACF_PeakStrength"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acf_strength(values):
    """Local peaks of Pearson(x[:-lag], x[lag:]), lags 1..floor(N/2).

    Only this additional feature differs between A and B. Existing notebook
    preprocessing, split, features, scaler and LR settings remain identical.
    find_peaks sees only the specified lag sequence: its endpoints are not
    eligible peaks. Strictly positive peak heights qualify; PeakLag is diagnostic.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 4:
        return 0.0, None, "too_short"
    if not np.isfinite(x).all():
        return 0.0, None, "uncomputable"
    if np.std(x) == 0:
        return 0.0, None, "constant"
    acf_values = []
    for lag in range(1, len(x)//2 + 1):
        x1, x2 = x[:-lag], x[lag:]
        if np.std(x1) == 0 or np.std(x2) == 0:
            acf_values.append(0.0)
            continue
        r = np.corrcoef(x1, x2)[0, 1]
        acf_values.append(float(r) if np.isfinite(r) else 0.0)
    acf_values = np.asarray(acf_values)
    peaks, _ = find_peaks(acf_values, height=0)
    # height=0 includes zero; the declared definition requires height > 0.
    peaks = peaks[acf_values[peaks] > 0]
    if len(peaks) == 0:
        return 0.0, None, "no_positive_peak"
    peak = int(peaks[np.argmax(acf_values[peaks])])
    return float(acf_values[peak]), peak + 1, "positive_peak"


def get_current_acf_peak_strength(x):
    return acf_strength(x)[0]


def metrics(y, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, prediction),
                Precision=precision_score(y, prediction, zero_division=0),
                Recall=recall_score(y, prediction, zero_division=0),
                F1=f1_score(y, prediction, zero_division=0),
                Balanced_Accuracy=balanced_accuracy_score(y, prediction),
                ROC_AUC=roc_auc_score(y, probability),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def table(df):
    # Avoid an optional tabulate dependency.
    return "| " + " | ".join(df.columns) + " |\n| " + " | ".join(
        ["---"]*len(df.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                            for v in row) + " |" for row in df.itertuples(index=False, name=None))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = ROOT / "src/LSTM-AutoEncoder/baseline_original.py"
    expected = (original.with_suffix(".sha256")).read_text().split()[0].lower()
    assert sha(original) == expected, "Immutable LSTM reference SHA256 mismatch"
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in
              [NOTEBOOK, original, ROOT/"data/raw/press_data_normal.csv",
               ROOT/"data/raw/outlier_data.csv"]}
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])}
    (OUT/"notebook_reference.ipynb").write_bytes(NOTEBOOK.read_bytes())
    namespace = dict(np=np, pd=pd, stats=stats)
    # Original CSV options and concatenation order, with workspace-relative paths.
    normal = pd.read_csv(ROOT/"data/raw/press_data_normal.csv")
    outlier = pd.read_csv(ROOT/"data/raw/outlier_data.csv")
    namespace["data"] = pd.concat([normal, outlier], ignore_index=True)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        for cell in [6, 7, 8, 9, 27]:
            exec(compile(sources[cell], f"notebook_cell_{cell}", "exec"), namespace)
    clean = namespace["featureData_clean"].copy()
    data = namespace["data"]
    assignment = next(node for node in ast.parse(sources[38]).body
                      if isinstance(node, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == "features_5" for t in node.targets))
    features = ast.literal_eval(assignment.value)
    assert features == ["AI0_RMS", "AI0_MaxAbs", "AI1_RMS", "AI1_PeakToPeak", "AI1_Kurtosis"]
    assert "AI0_Kurtosis" not in features
    # Reproduce the notebook's original split using its original six-column X.
    exec(sources[13], namespace)
    exec(sources[29], namespace)
    exec(sources[30], namespace)
    train_index = namespace["X_train"].index
    test_index = namespace["X_test"].index
    assert not set(train_index) & set(test_index)
    assert len(clean) == 595 and len(train_index) == 476 and len(test_index) == 119
    assert clean.loc[test_index, "Equipment_state"].value_counts().to_dict() == {0:115, 1:4}
    # Reuse and verify the previously saved notebook split, not a new random split.
    prior = ROOT / "results/logistic_current_acf_ablation"
    saved_split = pd.read_csv(prior/"burst_split.csv")
    saved_train = saved_split[saved_split.split == "train"].sort_values("split_order")["notebook_feature_index"].to_numpy()
    saved_test = saved_split[saved_split.split == "test"].sort_values("split_order")["notebook_feature_index"].to_numpy()
    np.testing.assert_array_equal(train_index, saved_train)
    np.testing.assert_array_equal(test_index, saved_test)
    train_index, test_index = pd.Index(saved_train), pd.Index(saved_test)
    diagnostic = []
    for burst_id, group in data.groupby("burst_id"):
        assert group["Equipment_state"].nunique() == 1
        strength, lag, reason = acf_strength(group["AI2_Current"])
        diagnostic.append(dict(burst_id=burst_id, Current_ACF_PeakStrength=strength,
                               PeakLag_diagnostic_only=lag, reason=reason))
    diagnostics = pd.DataFrame(diagnostic).set_index("burst_id")
    clean[ACF] = clean["burst_id"].map(diagnostics[ACF])
    assert np.isfinite(clean[features+[ACF]].to_numpy()).all()
    assert clean[ACF].between(0, 1).all()
    # Synthetic checks for the definition, without changing any experiment setting.
    assert acf_strength(np.ones(20))[0] == 0
    assert acf_strength([np.nan, 1])[0] == 0
    assert acf_strength(np.tile([1.0, -1.0], 10))[0] > 0
    diagnostics.to_csv(OUT/"acf_diagnostics.csv", encoding="utf-8-sig")
    clean.to_csv(OUT/"burst_feature_dataset.csv", index_label="notebook_feature_index", encoding="utf-8-sig")
    split = clean[["burst_id", "Equipment_state", "n_samples"]].copy()
    split["split"] = np.where(split.index.isin(train_index), "train", "test")
    split["split_order"] = -1
    for ids in [train_index, test_index]:
        split.loc[ids, "split_order"] = np.arange(len(ids))
    split.to_csv(OUT/"burst_split.csv", index_label="notebook_feature_index", encoding="utf-8-sig")
    summary = clean.groupby("Equipment_state")[ACF].agg(["count", "mean", "median", "std", "min", "max"])
    summary.index = summary.index.map({0:"normal", 1:"abnormal"})
    summary.to_csv(OUT/"periodicity_summary.csv", index_label="class", encoding="utf-8-sig")
    # Execute only setup statements from cell 36, excluding its fits and printing.
    setup = []
    for node in ast.parse(sources[36]).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id in ["pipeline", "cv", "scoring"] for t in node.targets)):
            setup.append(node)
    exec(compile(ast.Module(body=setup, type_ignores=[]), "notebook_cv_setup", "exec"), namespace)
    y = clean["Equipment_state"]
    folds = list(namespace["cv"].split(clean[features], y))
    saved_folds = json.loads((prior/"cv_split_indices.json").read_text(encoding="utf-8"))
    for (tr, te), saved in zip(folds, saved_folds):
        np.testing.assert_array_equal(clean.index[tr], saved["train_feature_indices"])
        np.testing.assert_array_equal(clean.index[te], saved["test_feature_indices"])
    assert len(folds) == len(saved_folds) == 50
    folds = [(clean.index.get_indexer(f["train_feature_indices"]),
              clean.index.get_indexer(f["test_feature_indices"])) for f in saved_folds]
    assert all((tr >= 0).all() and (te >= 0).all() for tr,te in folds)
    (OUT/"cv_split_indices.json").write_text(json.dumps([
        dict(fold=i+1, train_feature_indices=clean.index[tr].tolist(),
             test_feature_indices=clean.index[te].tolist())
        for i,(tr,te) in enumerate(folds)], indent=2), encoding="utf-8")
    from sklearn.base import clone
    rows, coefficients, cv_rows, cv_coefficients = [], [], [], []
    transformed_a = None
    for condition, columns in [("A_Baseline", features), ("B_Plus_ACF", features+[ACF])]:
        pipe = clone(namespace["pipeline"])
        pipe.fit(clean.loc[train_index, columns], y.loc[train_index])
        if transformed_a is None:
            transformed_a = pipe["scaler"].transform(clean.loc[train_index, columns])
        else:
            np.testing.assert_array_equal(transformed_a, pipe["scaler"].transform(
                clean.loc[train_index, columns])[:, :len(features)])
        test = clean.loc[test_index, columns]
        pred, prob = pipe.predict(test), pipe.predict_proba(test)[:, 1]
        row = dict(condition=condition, **metrics(y.loc[test_index], pred, prob))
        rows.append(row)
        pd.DataFrame(dict(notebook_feature_index=test_index,
                          burst_id=clean.loc[test_index, "burst_id"].to_numpy(),
                          label=y.loc[test_index].to_numpy(), prediction=pred,
                          probability=prob)).to_csv(OUT/f"{condition}_predictions.csv", index=False)
        pd.DataFrame(confusion_matrix(y.loc[test_index], pred, labels=[0,1]),
                     index=["actual_normal", "actual_abnormal"],
                     columns=["pred_normal", "pred_abnormal"]).to_csv(OUT/f"{condition}_confusion_matrix.csv")
        coef = pipe["model"].coef_[0]
        ranks = pd.Series(np.abs(coef)).rank(ascending=False, method="min").astype(int)
        coefficients.extend(dict(condition=condition, feature=f,
            coefficient=float(c), absolute_coefficient=float(abs(c)),
            absolute_rank=int(r)) for f,c,r in zip(columns,coef,ranks))
        print(condition, row, flush=True)
        results = cross_validate(clone(namespace["pipeline"]), clean[columns], y,
                                 cv=folds, scoring=namespace["scoring"], return_estimator=True)
        for i, ((_, te), estimator) in enumerate(zip(folds, results["estimator"])):
            fold_coef = estimator["model"].coef_[0]
            fold_ranks = pd.Series(np.abs(fold_coef)).rank(ascending=False, method="min").astype(int)
            cv_coefficients.extend(dict(condition=condition, fold=i+1, feature=f,
                coefficient=float(c), absolute_rank=int(r)) for f,c,r in zip(columns,fold_coef,fold_ranks))
            xx = clean.iloc[te][columns]
            cv_rows.append(dict(condition=condition, fold=i+1,
                **metrics(y.iloc[te], estimator.predict(xx), estimator.predict_proba(xx)[:,1])))
        print(condition, "50 paired CV folds complete", flush=True)
    metric_df = pd.DataFrame(rows)
    coef_df = pd.DataFrame(coefficients).sort_values(["condition", "absolute_rank"])
    cv_df = pd.DataFrame(cv_rows)
    numeric = ["Accuracy", "Precision", "Recall", "F1", "Balanced_Accuracy", "ROC_AUC"]
    delta = pd.DataFrame([dict(comparison="B-A", **{k:rows[1][k]-rows[0][k] for k in numeric})])
    cv_summary = []
    for condition, group in cv_df.groupby("condition", sort=False):
        for metric in numeric:
            cv_summary.append(dict(condition=condition, metric=metric,
                mean=group[metric].mean(), std=group[metric].std(ddof=0)))
    cv_summary = pd.DataFrame(cv_summary)
    cv_delta = pd.DataFrame([dict(metric=k, mean_delta_B_minus_A=
        cv_df.loc[cv_df.condition=="B_Plus_ACF", k].mean()-
        cv_df.loc[cv_df.condition=="A_Baseline", k].mean()) for k in numeric])
    for name, frame in [("holdout_metrics",metric_df),("holdout_delta_metrics",delta),
        ("feature_coefficients",coef_df),("cv_fold_metrics",cv_df),
        ("cv_summary",cv_summary),("cv_deltas",cv_delta)]:
        frame.to_csv(OUT/f"{name}.csv", index=False, encoding="utf-8-sig")
    stored_cv = dict(Precision=.7254502164502165, Recall=.9, F1=.7764314574314574,
                     Balanced_Accuracy=.9418260869565219, ROC_AUC=.9478260869565218)
    reproduced = {k:float(cv_df.loc[cv_df.condition=="A_Baseline",k].mean()) for k in stored_cv}
    reproduction = {k:dict(notebook=stored_cv[k], current=reproduced[k],
                           delta=reproduced[k]-stored_cv[k]) for k in stored_cv}
    for path, before in hashes.items():
        assert sha(ROOT/path) == before, f"Reference modified: {path}"
    environment = dict(Python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        reference_hashes=hashes, features_A=features, features_B=features+[ACF],
        LR_parameters=namespace["pipeline"]["model"].get_params(),
        notebook_CV_reproduction=reproduction, CV_std_ddof=0,
        warnings=[str(w.message) for w in recorded])
    (OUT/"experiment_metadata.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    cv_coef = pd.DataFrame(cv_coefficients)
    cv_coef.to_csv(OUT/"cv_fold_coefficients.csv", index=False, encoding="utf-8-sig")
    cv_coef_summary = cv_coef.groupby(["condition", "feature"], sort=False).agg(
        coefficient=("coefficient", "mean"), coefficient_std=("coefficient", lambda v: v.std(ddof=0)),
        mean_absolute_coefficient=("coefficient", lambda v: np.abs(v).mean())).reset_index()
    cv_coef_summary["absolute_coefficient"] = cv_coef_summary.coefficient.abs()
    cv_coef_summary["absolute_rank"] = cv_coef_summary.groupby("condition").absolute_coefficient.rank(
        ascending=False, method="min").astype(int)
    cv_coef_summary["fit_protocol"] = "50_fold_CV_mean"
    coef_df["fit_protocol"] = "fixed_80pct_train"
    pd.concat([coef_df, cv_coef_summary], ignore_index=True).to_csv(
        OUT/"feature_coefficients.csv", index=False, encoding="utf-8-sig")
    primary = cv_summary.pivot(index="condition", columns="metric", values="mean").reset_index()
    primary = primary[["condition"]+numeric]
    for condition in primary.condition:
        group = cv_df[cv_df.condition == condition]
        for count in ["TN", "FP", "FN", "TP"]:
            primary.loc[primary.condition == condition, "mean_"+count] = group[count].mean()
    primary.to_csv(OUT/"metrics.csv", index=False, encoding="utf-8-sig")
    cv_delta.rename(columns={"mean_delta_B_minus_A":"delta_B_minus_A"}).to_csv(
        OUT/"delta_metrics.csv", index=False, encoding="utf-8-sig")
    for condition, group in cv_df.groupby("condition"):
        # Average fold confusion, not a pooled independent-sample estimate.
        matrix = np.array([[group.TN.mean(),group.FP.mean()], [group.FN.mean(),group.TP.mean()]])
        pd.DataFrame(matrix, index=["actual_normal","actual_abnormal"],
                     columns=["pred_normal","pred_abnormal"]).to_csv(OUT/f"{condition}_CV_mean_confusion_matrix.csv")
    for k in stored_cv:
        np.testing.assert_allclose(reproduced[k], stored_cv[k], rtol=0, atol=1e-12)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    grouped = [clean.loc[y==i,ACF].to_numpy() for i in [0,1]]
    axes[0].boxplot(grouped, tick_labels=["Normal (575)","Abnormal (20)"])
    axes[0].set_ylabel("Current_ACF_PeakStrength (Pearson)")
    for values, label in zip(grouped,["Normal","Abnormal"]):
        axes[1].hist(values,bins=np.linspace(0,1,21),density=True,alpha=.5,label=label)
    axes[1].set_xlabel("Current_ACF_PeakStrength (Pearson)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT/"periodicity_distribution.png",dpi=180)
    plt.close(fig)
    acf_cv = cv_coef_summary[cv_coef_summary.feature==ACF].iloc[0]
    acf_holdout = coef_df[coef_df.feature==ACF].iloc[0]
    report = "# Logistic Regression: lag-wise Pearson ACF ablation\n\n"
    report += "## Baseline 및 고정 조건\n\n"
    report += f"Baseline A는 notebook cell 38의 feature {features}이며 AI0_Kurtosis는 제외한다. B는 Current_ACF_PeakStrength 하나만 추가한다.\n\n"
    report += "정상/이상 CSV concat 후 TimeStamp 정렬, gap>1초 또는 Equipment_state 변경 시 burst 분할. "
    report += "Notebook 원래 signed 진동 feature 함수를 그대로 실행하며 길이 4 미만 burst를 기존과 동일하게 제외한다. "
    report += "모델 대상 595 burst(정상575/이상20), 총 raw burst620. 새 abs 처리·imputation·추가 필터는 없다.\n\n"
    report += "StandardScaler와 LogisticRegression(class_weight='balanced', random_state=42)의 나머지 defaults를 유지한다. "
    report += "노트북과 기존 저장 split을 대조한 뒤 동일 index와 순서를 재사용한다. "
    report += "주 평가는 RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=42)의 동일 50 fold 평균이다. "
    report += "각 fold train에서만 scaler fit. CV std는 notebook과 동일한 ddof=0. 클래스 기술통계 std는 ddof=1. "
    report += "기존 baseline 5개 CV 지표가 1e-12 허용 오차 이내 재현됨을 assert했다. "
    report += "원본 notebook·CSV·공식 LSTM baseline의 실행 전후 SHA256을 검증했다.\n\n"
    report += "## Pearson ACF 정의\n\n"
    report += "각 burst의 전체 signed AI2_Current를 float array로 읽는다. N<4 또는 constant/nonfinite이면 0. "
    report += "각 lag=1..floor(N/2)에 대해 Pearson(x[:-lag],x[lag:])를 np.corrcoef로 직접 계산한다. "
    report += "각 lag의 두 부분 신호를 각각 중심화·표준화하는 Pearson 상관이며, np.correlate의 lag0 normalization은 사용하지 않는다. "
    report += "부분 신호가 constant 또는 correlation이 NaN/nonfinite이면 해당 lag를 0으로 처리한다. "
    report += "해당 lag sequence에만 find_peaks(height=0)를 적용하고 height>0을 추가 필터한다. "
    report += "끝점 lag1과 floor(N/2)는 scipy find_peaks의 endpoint 규칙에 따라 peak로 선택되지 않는다. "
    report += "양수 local peak의 최대 correlation, 없으면0. PeakLag는 분석용으로만 저장한다. 고정 lag/period는 입력에 없다.\n\n"
    report += "## 주 결과: 동일 반복 CV\n\n"+table(primary)+"\n\n"
    report += "mean_TN/FP/FN/TP는 fold별 confusion count의 평균이며 고유 독립 사례의 합계가 아니다.\n\n"
    report += "### A→B 변화량\n\n"+table(cv_delta)+"\n\n"
    report += "### CV mean/std\n\n"+table(cv_summary)+"\n\n"
    report += "## Coefficient\n\n"+table(cv_coef_summary)+"\n\n"
    report += f"ACF CV 평균 coefficient={acf_cv.coefficient:.6f}, std={acf_cv.coefficient_std:.6f}, "
    report += f"절댓값 평균계수 순위={int(acf_cv.absolute_rank)}/6. "
    report += f"고정 80% train fit coefficient={acf_holdout.coefficient:.6f}, 순위={int(acf_holdout.absolute_rank)}/6. "
    report += "모두 StandardScaler 적용 후 계수다. CV 계수 평균은 단일 배포 모델의 계수가 아니며 절댓값 순위는 성능 기여 또는 인과성 증명이 아니다.\n\n"
    report += "## 클래스별 주기성 통계\n\n"+table(summary.reset_index(names="class"))+"\n\n"
    report += "![Pearson ACF distribution](periodicity_distribution.png)\n\n"
    report += "## 참고: 기존 고정 80/20 split\n\n"
    report += "train476(460/16), test119(115/4), notebook stratified test_size=.2/random_state42 그대로. "
    report += "predict() 및 positive label1, ROC-AUC는 positive class probability를 사용한다. "
    report += "이 test와 전체595 대상 CV는 독립된 검증 두 개가 아니다.\n\n"
    report += table(metric_df)+"\n\n"+table(delta)+"\n\n"
    report += "## 최종 해석\n\n"
    for k in numeric:
        d = float(cv_delta.loc[cv_delta.metric==k,"mean_delta_B_minus_A"].iloc[0])
        direction = "상승" if d>1e-12 else "하락" if d < -1e-12 else "동일"
        report += f"- CV {k}: {direction}, Δ={d:+.6f}.\n"
    f1_delta = float(cv_delta.loc[cv_delta.metric=="F1","mean_delta_B_minus_A"].iloc[0])
    report += "\n" + ("F1 평균은 개선됐다. 다만 다른 지표와 변동을 함께 해석해야 한다." if f1_delta>0 else
        "F1 평균이 개선되지 않았으므로 주기성 feature가 유용하다고 일괄 결론 내리지 않는다.")
    report += " 반복 fold는 서로 독립이 아니며 이상 burst는20개다. 본 비교는 효과의 기술적 관찰이며 유의성 또는 외부 세션 일반화의 증거는 아니다. "
    report += "Threshold/hyperparameter tuning 및 test 기반 선택은 수행하지 않았다.\n"
    (OUT/"analysis_report.md").write_text(report,encoding="utf-8")
    print("PRIMARY: same 50-fold CV")
    print(primary.to_string(index=False))
    print(cv_delta.to_string(index=False))
    print("CV coefficients:",cv_coef_summary.to_string(index=False))
    print("Holdout coefficients:",coef_df.to_string(index=False))
    print(summary.to_string())
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
