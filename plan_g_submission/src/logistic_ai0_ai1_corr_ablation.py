"""One-factor notebook ablation: add signed AI0_AI1_Corr only.

Notebook preprocessing, burst boundaries, minimum length, original five
features, StandardScaler, balanced LogisticRegression and repeated CV stay
identical. No abs(corr), tuning, independent random split, or notebook edits.
"""
import ast
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from logistic_current_acf_ablation import metrics, sha, table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/logistic_ai0_ai1_corr_ablation"
CORR = "AI0_AI1_Corr"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = ROOT / "reference/소성가공.ipynb"
    original = ROOT / "src/LSTM-AutoEncoder/baseline_original.py"
    assert sha(original) == original.with_suffix(".sha256").read_text().split()[0].lower()
    inputs = [notebook, original, ROOT/"data/raw/press_data_normal.csv", ROOT/"data/raw/outlier_data.csv"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    source = {i:"".join(c["source"]) for i,c in enumerate(nb["cells"])}
    (OUT/"notebook_reference.ipynb").write_bytes(notebook.read_bytes())
    ns = dict(np=np, pd=pd, stats=stats)
    ns["data"] = pd.concat([pd.read_csv(inputs[2]),pd.read_csv(inputs[3])], ignore_index=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for i in [6,7,8,9,27]:
            exec(compile(source[i],f"notebook_cell_{i}","exec"),ns)
    clean = ns["featureData_clean"].copy()
    features = ast.literal_eval(next(n.value for n in ast.parse(source[38]).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="features_5" for t in n.targets)))
    assert features == ["AI0_RMS","AI0_MaxAbs","AI1_RMS","AI1_PeakToPeak","AI1_Kurtosis"]
    assert "AI0_Kurtosis" not in features
    # Exactly the notebook cell 51's signed Pearson calculation, plus requested NaN fallback.
    corr_rows = []
    for burst_id, group in ns["data"].groupby("burst_id"):
        if burst_id not in set(clean.burst_id):
            continue
        assert len(group) >= 4 and group.Equipment_state.nunique() == 1
        value = group.AI0_Vibration.corr(group.AI1_Vibration)
        corr_rows.append(dict(burst_id=burst_id, AI0_AI1_Corr=0.0 if pd.isna(value) else float(value),
                              nan_fallback=bool(pd.isna(value))))
    corr = pd.DataFrame(corr_rows).set_index("burst_id")
    clean[CORR] = clean.burst_id.map(corr[CORR])
    assert np.isfinite(clean[features+[CORR]].to_numpy()).all()
    assert clean[CORR].between(-1,1).all()
    assert len(clean)==595 and clean.Equipment_state.value_counts().to_dict()=={0:575,1:20}
    prior = ROOT / "results/logistic_current_acf_ablation"
    previous = pd.read_csv(prior/"burst_feature_dataset.csv",index_col="notebook_feature_index")
    pd.testing.assert_frame_equal(clean.drop(columns=CORR),previous.drop(columns="Current_ACF_PeakStrength"),check_names=False)
    clean.to_csv(OUT/"burst_feature_dataset.csv",index_label="notebook_feature_index",encoding="utf-8-sig")
    corr.to_csv(OUT/"burst_correlations.csv",encoding="utf-8-sig")
    summary = clean.groupby("Equipment_state")[CORR].agg(["count","mean","median","std"])
    summary.index = summary.index.map({0:"normal",1:"abnormal"})
    summary.to_csv(OUT/"correlation_summary.csv",index_label="class",encoding="utf-8-sig")
    setup = [n for n in ast.parse(source[36]).body if isinstance(n,(ast.Import,ast.ImportFrom)) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
            ["pipeline","cv","scoring"] for t in n.targets))]
    exec(compile(ast.Module(body=setup,type_ignores=[]),"notebook_cv_setup","exec"),ns)
    y = clean.Equipment_state
    expected_folds = list(ns["cv"].split(clean[features],y))
    saved_folds = json.loads((prior/"cv_split_indices.json").read_text(encoding="utf-8"))
    assert len(expected_folds)==len(saved_folds)==50
    for (tr,te),saved in zip(expected_folds,saved_folds):
        np.testing.assert_array_equal(clean.index[tr],saved["train_feature_indices"])
        np.testing.assert_array_equal(clean.index[te],saved["test_feature_indices"])
    # Use the already saved splits, after exact notebook reproduction checks.
    folds = [(clean.index.get_indexer(f["train_feature_indices"]),
              clean.index.get_indexer(f["test_feature_indices"])) for f in saved_folds]
    (OUT/"cv_split_indices.json").write_text(json.dumps(saved_folds,indent=2),encoding="utf-8")
    fold_metrics, fold_coef, predictions, scalers = [], [], [], []
    for condition,columns in [("A_Baseline",features),("B_Plus_Corr",features+[CORR])]:
        result = cross_validate(clone(ns["pipeline"]),clean[columns],y,cv=folds,
                                scoring=ns["scoring"],return_estimator=True)
        condition_scalers = []
        for i,((tr,te),pipe) in enumerate(zip(folds,result["estimator"])):
            x_test = clean.iloc[te][columns]
            pred,prob = pipe.predict(x_test),pipe.predict_proba(x_test)[:,1]
            fold_metrics.append(dict(condition=condition,fold=i+1,**metrics(y.iloc[te],pred,prob)))
            coefficients = pipe["model"].coef_[0]
            ranks = pd.Series(np.abs(coefficients)).rank(ascending=False,method="min").astype(int)
            fold_coef.extend(dict(condition=condition,fold=i+1,feature=f,coefficient=float(c),
                absolute_coefficient=float(abs(c)),absolute_rank=int(r)) for f,c,r in zip(columns,coefficients,ranks))
            predictions.extend(dict(condition=condition,fold=i+1,notebook_feature_index=int(idx),
                burst_id=int(clean.loc[idx,"burst_id"]),label=int(y.loc[idx]),prediction=int(pr),
                probability=float(pb)) for idx,pr,pb in zip(clean.index[te],pred,prob))
            condition_scalers.append((pipe["scaler"].mean_[:5],pipe["scaler"].scale_[:5]))
        scalers.append(condition_scalers)
        print(condition,"50 folds complete",flush=True)
    for a,b in zip(*scalers):
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[1],b[1])
    fm,fc = pd.DataFrame(fold_metrics),pd.DataFrame(fold_coef)
    numeric = ["Accuracy","Precision","Recall","F1","Balanced_Accuracy","ROC_AUC"]
    mean = fm.groupby("condition",sort=False)[numeric+["TN","FP","FN","TP"]].mean().reset_index()
    std = fm.groupby("condition",sort=False)[numeric].std(ddof=0).reset_index()
    delta = pd.DataFrame([dict(comparison="B-A",**{k:mean.iloc[1][k]-mean.iloc[0][k] for k in numeric})])
    paired = fm[fm.condition=="B_Plus_Corr"].set_index("fold")[numeric] - fm[fm.condition=="A_Baseline"].set_index("fold")[numeric]
    cs = fc.groupby(["condition","feature"],sort=False).agg(
        coefficient=("coefficient","mean"),coefficient_std=("coefficient",lambda v:v.std(ddof=0)),
        mean_absolute_coefficient=("absolute_coefficient","mean"),
        mean_fold_absolute_rank=("absolute_rank","mean")).reset_index()
    cs["absolute_coefficient"] = cs.coefficient.abs()
    cs["absolute_rank"] = cs.groupby("condition").absolute_coefficient.rank(ascending=False,method="min").astype(int)
    cs = cs.sort_values(["condition","absolute_rank"])
    for filename,frame in [("metrics",mean),("metrics_std",std),("delta_metrics",delta),
                           ("cv_fold_metrics",fm),("feature_coefficients",cs),("cv_fold_coefficients",fc),
                           ("predictions",pd.DataFrame(predictions))]:
        frame.to_csv(OUT/f"{filename}.csv",index=False,encoding="utf-8-sig")
    paired.to_csv(OUT/"paired_fold_deltas.csv",encoding="utf-8-sig")
    for _,row in mean.iterrows():
        pd.DataFrame([[row.TN,row.FP],[row.FN,row.TP]],index=["actual_normal","actual_abnormal"],
            columns=["pred_normal","pred_abnormal"]).to_csv(OUT/f"{row.condition}_mean_confusion_matrix.csv")
    stored = dict(Precision=.7254502164502165,Recall=.9,F1=.7764314574314574,
                  Balanced_Accuracy=.9418260869565219,ROC_AUC=.9478260869565218)
    for k,value in stored.items():
        np.testing.assert_allclose(mean.iloc[0][k],value,rtol=0,atol=1e-12)
    for p in inputs:
        assert sha(p)==hashes[str(p.relative_to(ROOT))]
    metadata = dict(reference_hashes=hashes,features_A=features,features_B=features+[CORR],
        Python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,model_parameters=ns["pipeline"]["model"].get_params(),
        CV=dict(n_splits=5,n_repeats=10,random_state=42,std_ddof=0),
        correlation="signed Pearson on original AI0_Vibration and AI1_Vibration; NaN -> 0; no abs",
        baseline_reproduced=True,existing_split_reused=True,
        nan_fallback_count=int(corr.nan_fallback.sum()),warnings=[str(w.message) for w in captured])
    (OUT/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    ac = cs[cs.feature==CORR].iloc[0]
    report = "# Logistic Regression AI0–AI1 signed correlation ablation\n\n"
    report += f"Baseline A: `{features}`. AI0_Kurtosis는 제외한다. B는 `{CORR}` 하나만 추가한다.\n\n"
    report += "Notebook cell6–9,27의 기존 preprocessing/feature 생성 코드를 그대로 실행했다. CSV concat 후 timestamp 정렬, "
    report += "gap>1초 또는 Equipment_state 변경에 따른 burst 분할, 길이4 미만 제외. "
    report += "575 정상/20 이상 burst와 기존 feature 값이 이전 실험과 완전히 동일함을 확인했다.\n\n"
    report += "추가 feature는 cell51과 동일하게 각 burst의 signed 원신호 AI0_Vibration과 AI1_Vibration 사이 Pearson correlation이다. "
    report += "NaN이면0, abs(corr)는 사용하지 않는다. 다른 feature·행·전처리는 변경하지 않았다.\n\n"
    report += "StandardScaler + LogisticRegression(class_weight='balanced',random_state=42)의 다른 defaults 유지. "
    report += "RepeatedStratifiedKFold(5 folds,10 repeats,random_state=42)의 기존 저장 index를 재사용하고 notebook 생성 결과와 대조했다. "
    report += "각 fold train에서만 scaler fit, 두 조건의 기존5개 feature scaler mean/scale도 동일함을 assert했다. "
    report += "판정은 원래 predict(), positive class1, ROC-AUC는 확률 기준. Threshold/hyperparameter tuning은 없다. "
    report += "baseline 5개 CV 지표를 1e-12 이내 재현했으며 notebook/CSV/공식 baseline hash는 실행 전후 동일하다.\n\n"
    report += "## 동일 50-fold CV 평균\n\n"+table(mean)+"\n\n"
    report += "TN/FP/FN/TP는 fold별 평균이며 총 독립 sample 수가 아니다.\n\n"
    report += "### B−A\n\n"+table(delta)+"\n\n### CV std (ddof=0)\n\n"+table(std)+"\n\n"
    report += "## 표준화 coefficient\n\n"+table(cs)+"\n\n"
    report += f"추가 correlation의 CV 평균 coefficient={ac.coefficient:.6f}, std={ac.coefficient_std:.6f}, "
    report += f"평균계수의 절댓값 순위={int(ac.absolute_rank)}/6. "
    report += "평균 coefficient는 fold 모델을 요약한 값이며 단일 fit 모델 계수는 아니다. "
    report += "absolute_rank는 |mean(coefficient)| 기준, mean_absolute_coefficient와 fold별 rank도 별도로 저장했다.\n\n"
    report += "## 클래스별 correlation 통계 (std ddof=1)\n\n"+table(summary.reset_index(names="class"))+"\n\n"
    report += "## 해석\n\n"
    for k in numeric:
        d=float(delta.iloc[0][k])
        report += f"- {k}: B−A={d:+.6f}.\n"
    report += "\n"+("CV 평균 F1은 상승했다." if delta.iloc[0].F1>0 else "CV 평균 F1은 개선되지 않았다.")
    report += " 각 지표의 방향과 변동을 함께 고려하며 coefficient 크기만으로 추가 효용을 결론 내리지 않는다. "
    report += "반복 fold는 독립이 아니고 이상 burst는20개이므로 일반화·통계적 유의성으로 확대 해석하지 않는다.\n"
    (OUT/"analysis_report.md").write_text(report,encoding="utf-8")
    print(mean.to_string(index=False))
    print(delta.to_string(index=False))
    print(cs.to_string(index=False))
    print(summary.to_string())
    print("Saved:",OUT)


if __name__=="__main__":
    main()
