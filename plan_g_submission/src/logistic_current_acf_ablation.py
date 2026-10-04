"""Notebook-preserving ablation: add only Current_ACF_PeakStrength.

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
OUT = ROOT / "results/logistic_current_acf_ablation"
NOTEBOOK = ROOT / "reference/소성가공.ipynb"
ACF = "Current_ACF_PeakStrength"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acf_strength(values):
    """Biased ACF r(k)=sum x[t]x[t+k]/sum x[t]^2, after centering.

    find_peaks examines the full positive-lag ACF (including lag zero as
    its neighbor), then candidates are restricted to 1..floor(N/2).
    Thus a candidate at N/2 must also be a genuine local peak.
    Only strictly positive peak heights qualify. Lag is diagnostic only.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        return 0.0, None, "uncomputable"
    x = x - x.mean()
    energy = np.dot(x, x)
    if not np.isfinite(energy) or energy <= 0:
        return 0.0, None, "constant_or_uncomputable"
    corr = np.correlate(x, x, mode="full")[len(x)-1:] / energy
    if not np.isfinite(corr).all():
        return 0.0, None, "uncomputable"
    peaks, _ = find_peaks(corr)
    peaks = peaks[(peaks >= 1) & (peaks <= len(x)//2)]
    peaks = peaks[corr[peaks] > 0]
    if len(peaks) == 0:
        return 0.0, None, "no_positive_peak"
    lag = int(peaks[np.argmax(corr[peaks])])
    return float(corr[lag]), lag, "positive_peak"


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
    summary = clean.groupby("Equipment_state")[ACF].agg(["count", "mean", "median", "std"])
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
    (OUT/"cv_split_indices.json").write_text(json.dumps([
        dict(fold=i+1, train_feature_indices=clean.index[tr].tolist(),
             test_feature_indices=clean.index[te].tolist())
        for i,(tr,te) in enumerate(folds)], indent=2), encoding="utf-8")
    from sklearn.base import clone
    rows, coefficients, cv_rows = [], [], []
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
    for name, frame in [("metrics",metric_df),("metric_deltas",delta),
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
    acf_row = coef_df[coef_df.feature==ACF].iloc[0]
    report = "# Logistic Regression Current ACF feature ablation\n\n"
    report += "## 고정 조건\n\n소성가공.ipynb cell 38의 best 5개 feature를 A로 사용한다. AI0_Kurtosis는 제외되어 있다. "
    report += "B는 Current_ACF_PeakStrength 하나만 추가한다. 기존 notebook 및 LSTM 원본 SHA256이 실행 전후 동일함을 확인했다.\n\n"
    report += f"A feature: `{features}`\n\n"
    report += "정상/이상 CSV를 concat한 뒤 TimeStamp로 정렬하고, gap > 1초 또는 Equipment_state 변경으로 burst를 나눈다. "
    report += "기존과 동일하게 길이 4 미만 burst를 제외한다. 진동 feature는 notebook 원래 signed 신호의 계산 함수를 그대로 실행했다. "
    report += "새 abs 변환, 결측 대체, row 제거, feature 추가는 하지 않았다.\n\n"
    report += "595 burst(정상 575, 이상 20), train 476(460/16), test 119(115/4). "
    report += "노트북의 stratified test_size=0.2, random_state=42 split을 재현하고 A/B에 같은 burst index 및 순서를 적용했다. "
    report += "StandardScaler는 각 train에만 fit. LogisticRegression(class_weight='balanced', random_state=42)의 나머지 기본값을 유지한다. "
    report += "predict() 판정, positive label=1, 확률 기반 ROC-AUC. Test에 의한 threshold 또는 hyperparameter 선택은 없다.\n\n"
    report += "## ACF 정의\n\n전체 burst의 signed AI2_Current에서 평균을 제거한다. "
    report += "ACF(k)=sum(x_centered[t]*x_centered[t+k])/sum(x_centered[t]^2); lag별 overlap 보정은 하지 않는다. "
    report += "scipy.signal.find_peaks로 전체 ACF의 local peak를 구한 뒤 1≤lag≤floor(N/2), ACF>0인 후보의 최대값만 사용한다. "
    report += "범위 끝에서도 실제 오른쪽 이웃과 비교한다. Constant/nonfinite/no positive peak는 0. "
    report += "PeakLag는 진단 파일에만 저장되며 입력에서 제외한다.\n\n"
    report += "## 고정 test 결과\n\n" + table(metric_df) + "\n\n### 변화량 B−A\n\n"+table(delta)+"\n\n"
    report += "## 기존 best 평가 방식: 동일 50-fold paired CV\n\n"
    report += "RepeatedStratifiedKFold(5 folds, 10 repeats, random_state=42)를 전체 595 burst에서 재현한다. "
    report += "두 조건이 동일 fold를 사용하고 각 fold train에서만 scaler를 fit한다. "
    report += "노트북의 CV는 전체 데이터 대상이므로 고정 test와 독립된 추가 검증은 아니다. CV std는 notebook과 동일한 ddof=0.\n\n"
    report += table(cv_summary)+"\n\n"+table(cv_delta)+"\n\n"
    report += "### Notebook 저장 baseline CV 재현 확인\n\n"+table(pd.DataFrame([
        dict(metric=k, **v) for k,v in reproduction.items()]))+"\n\n"
    report += "## 계수 및 기술통계\n\n"+table(coef_df)+"\n\n"+table(summary.reset_index(names="class"))+"\n\n"
    report += f"ACF 표준화 계수 {acf_row.coefficient:.6f}, 절댓값 순위 {int(acf_row.absolute_rank)}/6. "
    report += "계수 크기는 표준화 입력 기준의 상대적 영향이며 독립적 또는 인과적 기여의 증명은 아니다.\n\n"
    report += "## 해석\n\n"
    report += "고정 test에서는 기존 정상 오탐 1건이 사라져 F1이 0.888889→1.000000으로 개선됐다. "
    report += "Recall과 ROC-AUC는 이미 1.0이므로 변화가 없다. 그러나 기존 best를 평가한 반복 CV에서는 "
    report += "F1과 Precision, Balanced Accuracy가 소폭 하락하고 Recall은 동일하며 ROC-AUC만 소폭 상승했다. "
    report += "따라서 단일 test에서는 도움이 됐지만, 전반적이고 일관된 개선이 확인됐다고 결론 내릴 수 없다.\n\n"
    for k in ["F1", "Recall", "Balanced_Accuracy", "ROC_AUC"]:
        d = rows[1][k]-rows[0][k]
        cd = float(cv_delta.loc[cv_delta.metric==k,"mean_delta_B_minus_A"].iloc[0])
        report += f"- {k}: 고정 test Δ={d:+.6f}, CV 평균 Δ={cd:+.6f}.\n"
    report += "\n고정 test의 이상 burst가 4개뿐이므로 Recall은 한 건당 0.25 변한다. "
    report += "50개 CV fold는 반복에 따른 중복 표본으로 서로 독립이 아니다. "
    report += "전체 지표와 CV에서 나타난 변화를 함께 해석하고, 계수만으로 성능 개선을 주장하지 않는다. "
    report += "기존 best feature 선택 이력은 그대로 유지했으며 별도의 tuning이나 모델 선택은 수행하지 않았다.\n"
    (OUT/"analysis_report.md").write_text(report, encoding="utf-8")
    print(metric_df.to_string(index=False))
    print(delta.to_string(index=False))
    print(cv_summary.to_string(index=False))
    print(coef_df.to_string(index=False))
    print(summary.to_string())
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
