"""Notebook LR ablation: class_weight is the sole experimental factor.

Keep the five features, burst preprocessing, all 595 samples, StandardScaler,
LR defaults/random_state, original 50 CV folds and predict() threshold 0.5.
No held-out test evaluation or threshold tuning is performed.
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
OUT = ROOT / "results/logistic_class_weight_ablation"
F1_TOLERANCE = 0.001  # Fixed before evaluation: near-tie in mean CV F1.
RECALL_FLOOR = 0.85
CONDITIONS = [("None",None)] + [(f"{{0:1,1:{w}}}",{0:1,1:w}) for w in [3,5,7,10,15,20]] + [("balanced","balanced")]


def select(frame):
    top = frame.F1_mean.max()
    candidates = frame[frame.F1_mean >= top-F1_TOLERANCE-1e-12]
    return candidates.sort_values(["Recall_mean","F1_mean","Precision_mean","condition_order"],
                                  ascending=[False,False,False,True]).iloc[0]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    nb_path = ROOT/"reference/소성가공.ipynb"
    original = ROOT/"src/LSTM-AutoEncoder/baseline_original.py"
    assert sha(original)==original.with_suffix(".sha256").read_text().split()[0].lower()
    inputs = [nb_path,original,ROOT/"data/raw/press_data_normal.csv",ROOT/"data/raw/outlier_data.csv"]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    source = {i:"".join(c["source"]) for i,c in enumerate(nb["cells"])}
    (OUT/"notebook_reference.ipynb").write_bytes(nb_path.read_bytes())
    ns = dict(np=np,pd=pd,stats=stats,data=pd.concat([pd.read_csv(inputs[2]),pd.read_csv(inputs[3])],ignore_index=True))
    with warnings.catch_warnings(record=True) as prep_warnings:
        warnings.simplefilter("always")
        for i in [6,7,8,9,27]:
            exec(compile(source[i],f"notebook_cell_{i}","exec"),ns)
    features = ast.literal_eval(next(n.value for n in ast.parse(source[38]).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="features_5" for t in n.targets)))
    assert features==["AI0_RMS","AI0_MaxAbs","AI1_RMS","AI1_PeakToPeak","AI1_Kurtosis"]
    clean = ns["featureData_clean"]
    prior = ROOT/"results/logistic_current_acf_ablation"
    old = pd.read_csv(prior/"burst_feature_dataset.csv",index_col="notebook_feature_index")
    pd.testing.assert_frame_equal(clean,old.drop(columns="Current_ACF_PeakStrength"),check_names=False)
    X,y = clean[features],clean.Equipment_state
    assert len(X)==595 and y.value_counts().to_dict()=={0:575,1:20}
    assert np.isfinite(X.to_numpy()).all()
    setup = [n for n in ast.parse(source[36]).body if isinstance(n,(ast.Import,ast.ImportFrom)) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
            ["pipeline","cv","scoring"] for t in n.targets))]
    exec(compile(ast.Module(body=setup,type_ignores=[]),"notebook_cv_setup","exec"),ns)
    saved = json.loads((prior/"cv_split_indices.json").read_text(encoding="utf-8"))
    folds = list(ns["cv"].split(X,y))
    assert len(folds)==len(saved)==50
    for (tr,te),f in zip(folds,saved):
        np.testing.assert_array_equal(clean.index[tr],f["train_feature_indices"])
        np.testing.assert_array_equal(clean.index[te],f["test_feature_indices"])
    folds = [(clean.index.get_indexer(f["train_feature_indices"]),clean.index.get_indexer(f["test_feature_indices"])) for f in saved]
    (OUT/"cv_split_indices.json").write_text(json.dumps(saved,indent=2),encoding="utf-8")
    clean[["burst_id","Equipment_state","n_samples"]+features].to_csv(OUT/"feature_dataset.csv",index_label="notebook_feature_index",encoding="utf-8-sig")
    metric_names = ["Accuracy","Precision","Recall","F1","Balanced_Accuracy","ROC_AUC","TN","FP","FN","TP"]
    fold_rows,mean_rows,predictions,weight_rows,training_warnings = [],[],[],[],[]
    scaler_reference = None
    base_params = ns["pipeline"]["model"].get_params()
    for order,(label,weight) in enumerate(CONDITIONS):
        pipe = clone(ns["pipeline"])
        pipe.set_params(model__class_weight=weight)
        assert {k:v for k,v in pipe["model"].get_params().items() if k!="class_weight"} == {
            k:v for k,v in base_params.items() if k!="class_weight"}
        with warnings.catch_warnings(record=True) as ww:
            warnings.simplefilter("always")
            result = cross_validate(pipe,X,y,cv=folds,scoring=ns["scoring"],return_estimator=True)
        training_warnings.extend(dict(weight=label,message=str(w.message)) for w in ww)
        rows,scalers = [],[]
        for i,((tr,te),model) in enumerate(zip(folds,result["estimator"])):
            pred,prob = model.predict(X.iloc[te]),model.predict_proba(X.iloc[te])[:,1]
            # Original binary predict() uses the same 0.5 probability boundary.
            np.testing.assert_array_equal(pred,(prob>0.5).astype(int))
            row = dict(weight=label,fold=i+1,**metrics(y.iloc[te],pred,prob))
            rows.append(row)
            predictions.extend(dict(weight=label,fold=i+1,burst_id=int(clean.iloc[idx].burst_id),
                label=int(y.iloc[idx]),prediction=int(pr),probability=float(pb)) for idx,pr,pb in zip(te,pred,prob))
            counts = y.iloc[tr].value_counts()
            w0,w1 = (len(tr)/(2*counts[0]),len(tr)/(2*counts[1])) if weight=="balanced" else (
                (1,1) if weight is None else (weight[0],weight[1]))
            weight_rows.append(dict(weight=label,fold=i+1,normal_count=int(counts[0]),abnormal_count=int(counts[1]),
                                    effective_normal_weight=w0,effective_abnormal_weight=w1,relative_abnormal_weight=w1/w0))
            scalers.append((model["scaler"].mean_,model["scaler"].scale_))
        if scaler_reference is None:
            scaler_reference=scalers
        else:
            for a,b in zip(scaler_reference,scalers):
                np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])
        frame=pd.DataFrame(rows)
        summary=dict(weight=label,condition_order=order)
        for name in metric_names:
            summary[name+"_mean"]=float(frame[name].mean())
            summary[name+"_std"]=float(frame[name].std(ddof=0))
        summary["recall_below_0_85"]=summary["Recall_mean"]<RECALL_FLOOR
        mean_rows.append(summary)
        fold_rows.extend(rows)
        print(label, "F1",summary["F1_mean"],"Precision",summary["Precision_mean"],"Recall",summary["Recall_mean"],flush=True)
    means=pd.DataFrame(mean_rows)
    balanced=means[means.weight=="balanced"].iloc[0]
    stored=dict(Precision=.7254502164502165,Recall=.9,F1=.7764314574314574,
                Balanced_Accuracy=.9418260869565219,ROC_AUC=.9478260869565218)
    for k,v in stored.items():
        np.testing.assert_allclose(balanced[k+"_mean"],v,atol=1e-12,rtol=0)
    delta=means[["weight"]].copy()
    for k in metric_names:
        delta["delta_"+k]=means[k+"_mean"]-balanced[k+"_mean"]
    chosen=select(means)
    strict=means.sort_values(["F1_mean","Recall_mean","condition_order"],ascending=[False,False,True]).iloc[0]
    acceptable=means[~means.recall_below_0_85]
    constrained=select(acceptable) if len(acceptable) else None
    means["strict_F1_max"]=means.weight==strict.weight
    means["selected_by_F1_near_tie_rule"]=means.weight==chosen.weight
    means["best_with_Recall_at_least_0_85"]=False if constrained is None else means.weight==constrained.weight
    means=means.sort_values(["F1_mean","Recall_mean"],ascending=False)
    for name,frame in [("metrics",means),("delta_vs_balanced",delta),("cv_fold_metrics",pd.DataFrame(fold_rows)),
                       ("predictions",pd.DataFrame(predictions)),("effective_class_weights",pd.DataFrame(weight_rows))]:
        frame.to_csv(OUT/f"{name}.csv",index=False,encoding="utf-8-sig")
    best=dict(selection_uses="mean scores from same 50 CV folds only; no held-out test",
        near_tie_tolerance=F1_TOLERANCE,recall_warning_floor=RECALL_FLOOR,
        selection_rule="within 0.001 of maximum mean F1, highest mean Recall, then F1, Precision, declared order",
        strict_F1_max_weight=strict.weight,selected_weight=chosen.weight,
        strict_F1_max_weights=means.loc[np.isclose(means.F1_mean,strict.F1_mean,rtol=0,atol=1e-12),"weight"].tolist(),
        selected_class_weight=dict(CONDITIONS)[chosen.weight],
        selected_metrics={k:float(chosen[k+"_mean"]) for k in metric_names},
        selected_recall_below_floor=bool(chosen.recall_below_0_85),
        best_weight_with_recall_at_least_0_85=None if constrained is None else constrained.weight)
    (OUT/"best_weight.json").write_text(json.dumps(best,indent=2,ensure_ascii=False),encoding="utf-8")
    for p in inputs:
        assert sha(p)==hashes[str(p.relative_to(ROOT))]
    metadata=dict(reference_hashes=hashes,features=features,random_state=42,
        model_parameters_except_class_weight={k:v for k,v in base_params.items() if k!="class_weight"},
        scoring=ns["scoring"],CV=dict(n_splits=5,n_repeats=10,random_state=42,std_ddof=0),
        decision_threshold=.5,baseline_reproduced=True,split_reused=True,
        Python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        preprocessing_warnings=[str(w.message) for w in prep_warnings],training_warnings=training_warnings)
    (OUT/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    display=means[["weight"]+[k+"_mean" for k in metric_names]+["recall_below_0_85"]]
    report="# Logistic Regression class_weight ablation\n\n"
    report+=f"유일한 변경 요소는 class_weight다. 기존 5개 입력: `{features}`. AI0_Kurtosis 제외, 추가 feature 없음.\n\n"
    report+="Notebook cell6–9/27의 signed feature 계산, concat/timestamp 정렬, gap>1초 또는 state change burst, n_samples≥4를 그대로 실행했다. "
    report+="모델 대상575 정상/20 이상 burst 및 모든 feature가 이전 저장 자료와 동일함을 확인했다. "
    report+="StandardScaler는 fold train에만 fit하며 조건별 mean/scale 동일성을 assert했다. LR 나머지 defaults/random_state42 유지. "
    report+="Notebook cell36의 5-fold×10 repeats 및 이전에 저장한 동일50fold 사용. predict()와 probability>0.5의 일치를 검사했다. "
    report+="원래 precision/recall/f1/balanced_accuracy/roc_auc scoring 그대로, Accuracy와 confusion을 추가 집계. "
    report+="std는 notebook과 동일하게 ddof=0. Separate test 및 threshold tuning 없음. Notebook/CSV/공식 baseline SHA256 실행 전후 동일.\n\n"
    report+="## Mean 결과 (F1 내림차순)\n\n"+table(display)+"\n\n"
    report+="TN/FP/FN/TP는 fold별 평균이며 평균 matrix=[[TN,FP],[FN,TP]].\n\n"
    report+="## Fold 간 std\n\n"+table(means[["weight"]+[k+"_std" for k in metric_names]])+"\n\n"
    report+="## Balanced 대비 변화\n\n"+table(delta)+"\n\n"
    report+=f"## 선택\n\nStrict F1 maximum: **{strict.weight}**. 사전 고정한 F1 near-tie≤{F1_TOLERANCE}에서 Recall 우선 선택: **{chosen.weight}**. "
    report+=f"Recall≥0.85 조건을 추가로 만족하는 후보 중 동일 규칙의 최고: **{None if constrained is None else constrained.weight}**. "
    report+="Recall floor는 주 선택의 자동 제외 조건이 아니라 별도 warning 및 운영 제약 참고값이다.\n\n"
    selected_delta=delta[delta.weight==chosen.weight]
    report+=table(selected_delta)+"\n\n"
    report+="## 해석과 한계\n\nBalanced는 각 fold에서 n/(2*n_class)를 적용하므로 상대 이상 가중치는28.75지만 "
    report+="절대 가중치는 normal≈0.51739/abnormal=14.875다. 직접 지정 {0:1,1:w}와는 loss 전체 scale 및 L2 penalty 대비 효과도 다르므로 단순 ratio 동일 모델로 해석하지 않는다. "
    report+="이 차이를 임의 rescaling으로 고치지 않고 요청한 class_weight 그대로 비교했다.\n\n"
    report+="F1·Recall·Precision과 FP/FN 변화를 함께 판단한다. Recall<0.85는 표에 표시했다. "
    report+="현재 CV에 의한 weight 선택 결과이며 같은 CV의 최고 성능은 선택 후 독립 평가 성능이 아니다. "
    report+="반복 fold는 독립이 아니고 이상 burst가20개라 작은 평균 차이를 유의한 개선 또는 새 세션 일반화로 해석하지 않는다.\n"
    report+=f"\nF1 공동 최고 조건: {best['strict_F1_max_weights']}. 고정 선택 규칙에 따른 조건은 {chosen.weight}다. "
    report+=f"Balanced 대비 Precision Δ={float(chosen.Precision_mean-balanced.Precision_mean):+.6f}, "
    report+=f"Recall Δ={float(chosen.Recall_mean-balanced.Recall_mean):+.6f}, F1 Δ={float(chosen.F1_mean-balanced.F1_mean):+.6f}, "
    report+=f"FP Δ={float(chosen.FP_mean-balanced.FP_mean):+.2f}, FN Δ={float(chosen.FN_mean-balanced.FN_mean):+.2f}, "
    report+=f"ROC-AUC Δ={float(chosen.ROC_AUC_mean-balanced.ROC_AUC_mean):+.6f}. "
    report+="추천은 사전 지정 F1/Recall 기준에 따른 것이며 모든 지표에서 우월하다는 뜻은 아니다.\n"
    (OUT/"analysis_report.md").write_text(report,encoding="utf-8")
    print(display.to_string(index=False))
    print("BEST:",json.dumps(best,ensure_ascii=False))
    print("DELTAS:",selected_delta.to_string(index=False))
    print("Saved:",OUT)


if __name__=="__main__":
    main()
