"""Direct 50-fold validation tracking for the notebook's five-feature LR.

Use class_weight=None and fixed probability > 0.5. All other notebook
preprocessing, sample membership, model settings and saved CV indices stay
unchanged. No cross_validate execution, new split, threshold tuning or test fit.
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
from logistic_current_acf_ablation import metrics, sha, table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/logistic_fn_burst_tracking"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    notebook = ROOT/"reference/소성가공.ipynb"
    original = ROOT/"src/LSTM-AutoEncoder/baseline_original.py"
    assert sha(original)==original.with_suffix(".sha256").read_text().split()[0].lower()
    inputs=[notebook,original,ROOT/"data/raw/press_data_normal.csv",ROOT/"data/raw/outlier_data.csv"]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    nb=json.loads(notebook.read_text(encoding="utf-8"))
    sources={i:"".join(c["source"]) for i,c in enumerate(nb["cells"])}
    ns=dict(np=np,pd=pd,stats=stats,data=pd.concat([pd.read_csv(inputs[2]),pd.read_csv(inputs[3])],ignore_index=True))
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter("always")
        for i in [6,7,8,9,27]:
            exec(compile(sources[i],f"notebook_cell_{i}","exec"),ns)
    features=ast.literal_eval(next(n.value for n in ast.parse(sources[38]).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="features_5" for t in n.targets)))
    assert features==["AI0_RMS","AI0_MaxAbs","AI1_RMS","AI1_PeakToPeak","AI1_Kurtosis"]
    clean=ns["featureData_clean"]
    prior=ROOT/"results/logistic_current_acf_ablation"
    old=pd.read_csv(prior/"burst_feature_dataset.csv",index_col="notebook_feature_index")
    pd.testing.assert_frame_equal(clean,old.drop(columns="Current_ACF_PeakStrength"),check_names=False)
    setup=[n for n in ast.parse(sources[36]).body if isinstance(n,(ast.Import,ast.ImportFrom)) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ["pipeline","cv","scoring"] for t in n.targets))]
    exec(compile(ast.Module(body=setup,type_ignores=[]),"notebook_model_setup","exec"),ns)
    X,y=clean[features],clean.Equipment_state
    saved=json.loads((prior/"cv_split_indices.json").read_text(encoding="utf-8"))
    generated=list(ns["cv"].split(X,y))
    assert len(generated)==len(saved)==50
    for (tr,va),record in zip(generated,saved):
        np.testing.assert_array_equal(clean.index[tr],record["train_feature_indices"])
        np.testing.assert_array_equal(clean.index[va],record["test_feature_indices"])
    (OUT/"cv_split_indices.json").write_text(json.dumps(saved,indent=2),encoding="utf-8")
    rows,fold_rows,coef_rows=[],[],[]
    # Direct loop: train scaler, train LR, predict validation, fixed threshold.
    for i,record in enumerate(saved):
        train_index=pd.Index(record["train_feature_indices"])
        validation_index=pd.Index(record["test_feature_indices"])
        assert not set(train_index)&set(validation_index)
        scaler=clone(ns["pipeline"]["scaler"])
        model=clone(ns["pipeline"]["model"]).set_params(class_weight=None)
        train_scaled=scaler.fit_transform(X.loc[train_index])
        validation_scaled=scaler.transform(X.loc[validation_index])
        model.fit(train_scaled,y.loc[train_index])
        probabilities=model.predict_proba(validation_scaled)[:,1]
        predicted=(probabilities>0.5).astype(int)
        np.testing.assert_array_equal(predicted,model.predict(validation_scaled))
        repeat=i//5+1
        fold=i%5+1
        fold_rows.append(dict(cv_index=i+1,repeat=repeat,fold=fold,
            **metrics(y.loc[validation_index],predicted,probabilities)))
        coef_rows.extend(dict(cv_index=i+1,repeat=repeat,fold=fold,feature=f,coefficient=float(c))
                         for f,c in zip(features,model.coef_[0]))
        for idx,pred,prob in zip(validation_index,predicted,probabilities):
            truth=int(y.loc[idx])
            rows.append(dict(cv_index=i+1,repeat=repeat,fold=fold,
                notebook_feature_index=int(idx),burst_id=int(clean.loc[idx,"burst_id"]),
                true_label=truth,predicted_label=int(pred),probability_y1=float(prob),
                is_FN=int(truth==1 and pred==0),is_FP=int(truth==0 and pred==1)))
        if fold==5:
            print(f"Repeat {repeat}/10 complete",flush=True)
    predictions=pd.DataFrame(rows)
    fold_metrics=pd.DataFrame(fold_rows)
    assert len(predictions)==5950
    assert predictions.groupby("burst_id").size().eq(10).all()
    assert predictions.groupby(["repeat","burst_id"]).size().eq(1).all()
    assert predictions.true_label.eq(1).sum()==200
    # Independent agreement with previously saved class_weight=None results.
    previous=pd.read_csv(ROOT/"results/logistic_class_weight_ablation/predictions.csv",keep_default_na=False)
    previous=previous[previous.weight=="None"].sort_values(["fold","burst_id"])
    current=predictions.sort_values(["cv_index","burst_id"])
    np.testing.assert_array_equal(current.burst_id,previous.burst_id)
    np.testing.assert_array_equal(current.true_label,previous.label)
    np.testing.assert_array_equal(current.predicted_label,previous.prediction)
    np.testing.assert_allclose(current.probability_y1,previous.probability,rtol=0,atol=1e-12)
    summary=predictions.groupby("burst_id").agg(true_label=("true_label","first"),
        validation_count=("true_label","size"),FN_count=("is_FN","sum"),FP_count=("is_FP","sum"),
        predicted_abnormal_count=("predicted_label","sum"),
        probability_mean=("probability_y1","mean"),probability_median=("probability_y1","median"),
        probability_std=("probability_y1","std"),probability_min=("probability_y1","min"),
        probability_max=("probability_y1","max")).reset_index()
    summary["FN_rate"]=summary.FN_count/summary.validation_count
    summary["FP_rate"]=summary.FP_count/summary.validation_count
    temporal=ns["data"].groupby("burst_id").agg(start_timestamp=("TimeStamp","min"),end_timestamp=("TimeStamp","max"))
    summary=summary.merge(clean[["burst_id","n_samples"]+features],on="burst_id",validate="one_to_one").merge(
        temporal,on="burst_id",validate="one_to_one")
    abnormal=summary[summary.true_label==1].sort_values(["FN_count","probability_mean"],ascending=[False,True])
    events=predictions[predictions.is_FN==1].sort_values(["burst_id","repeat"])
    numeric=["Accuracy","Precision","Recall","F1","Balanced_Accuracy","ROC_AUC","TN","FP","FN","TP"]
    overall=pd.DataFrame([dict(metric=k,mean=fold_metrics[k].mean(),std=fold_metrics[k].std(ddof=0)) for k in numeric])
    assert np.isclose(fold_metrics.Recall.mean(),.9)
    by_repeat=predictions.groupby("repeat").agg(validation_count=("true_label","size"),
        abnormal_count=("true_label","sum"),FN_count=("is_FN","sum"),FP_count=("is_FP","sum"))
    by_repeat["Recall"]=1-by_repeat.FN_count/by_repeat.abnormal_count
    total_fn=int(events.shape[0])
    concentration=dict(total_abnormal_bursts=20,validation_predictions_per_burst=10,
        total_abnormal_validation_predictions=200,total_FN_predictions=total_fn,
        unique_FN_bursts=int((abnormal.FN_count>0).sum()),
        always_FN_bursts=abnormal.loc[abnormal.FN_count==10,"burst_id"].astype(int).tolist(),
        intermittent_FN_bursts=abnormal.loc[(abnormal.FN_count>0)&(abnormal.FN_count<10),"burst_id"].astype(int).tolist(),
        top1_FN_share=float(abnormal.FN_count.iloc[0]/total_fn) if total_fn else 0,
        top2_FN_share=float(abnormal.FN_count.iloc[:2].sum()/total_fn) if total_fn else 0)
    for filename,frame in [("validation_predictions",predictions),("fold_metrics",fold_metrics),
        ("metrics_summary",overall),("burst_prediction_summary",summary),
        ("abnormal_burst_summary",abnormal),("FN_events",events),("fold_coefficients",pd.DataFrame(coef_rows))]:
        frame.to_csv(OUT/f"{filename}.csv",index=False,encoding="utf-8-sig")
    by_repeat.to_csv(OUT/"repeat_summary.csv",encoding="utf-8-sig")
    (OUT/"FN_concentration.json").write_text(json.dumps(concentration,indent=2),encoding="utf-8")
    for path in inputs:
        assert sha(path)==hashes[str(path.relative_to(ROOT))]
    metadata=dict(reference_hashes=hashes,features=features,class_weight=None,threshold=.5,
        decision_rule="P(y=1)>0.5, equality is normal; identical to original predict()",
        CV=dict(n_splits=5,n_repeats=10,random_state=42),saved_folds_reused=True,
        direct_CV_loop=True,prior_None_predictions_reproduced=True,
        model_parameters=model.get_params(),Python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        preprocessing_warnings=[str(w.message) for w in ww])
    (OUT/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    report="# Logistic Regression: burst別 FN tracking\n\n"
    report+=f"5-feature set: `{features}`. AI0_Kurtosis 제외, class_weight=None, threshold0.5. "
    report+="Notebook의 signed feature, burst(gap>1초 또는 state change), 길이4 미만 제외, StandardScaler와 LR 나머지 defaults/random_state42 유지. "
    report+="기존 저장된5-fold×10 repeats의50fold를 대조한 뒤 직접 loop로 train scaler fit→LR fit→validation probability→판정했다. "
    report+="새 split/별도 test/threshold tuning은 없으며 이전 None 조건의 전체 prediction/probability를 재현했다. "
    report+="각 burst는 repeat마다 정확히1번 validation에 포함되어 총10회 예측된다. Notebook/CSV/공식 baseline hash는 그대로다.\n\n"
    report+="## CV metrics (std ddof=0)\n\n"+table(overall)+"\n\n"
    report+=f"## FN 집중도\n\n200개 이상 validation 예측에서 FN은 {total_fn}회, "
    report+=f"FN 발생 burst는20개 중 {concentration['unique_FN_bursts']}개다. "
    report+=f"10회 모두 FN인 burst: {concentration['always_FN_bursts']}; 간헐 FN burst: {concentration['intermittent_FN_bursts']}. "
    report+=f"FN 상위2개 burst가 전체 FN의 {100*concentration['top2_FN_share']:.1f}%를 차지한다.\n\n"
    report+="## Abnormal burst별 기록\n\n"+table(abnormal[["burst_id","validation_count","FN_count","FN_rate",
        "probability_mean","probability_min","probability_max","n_samples","start_timestamp"]])+"\n\n"
    report+="## Repeat별 집계\n\n"+table(by_repeat.reset_index())+"\n\n"
    report+="## 해석\n\n"
    if concentration["always_FN_bursts"] and not concentration["intermittent_FN_bursts"]:
        report+="FN은 여러 burst에 랜덤하게 흩어지기보다 같은 이상 burst에 모든 repeat에서 반복된다. "
        report+="현재 split과 학습 변동에서는 해당 사례가 안정적으로 정상 판정을 받는다는 뜻이다. "
    else:
        report+="FN 빈도와 확률 범위를 함께 보아 고정적으로 어려운 burst와 fold에 따라 판정이 바뀌는 burst를 구분한다. "
    report+="이 결과만으로 label 오류나 실제 신호 원인을 단정할 수 없다. 반복10회는 동일 burst의 반복 평가이지10개의 독립 고장 사례가 아니다. "
    report+="원인 검토에는 해당 timestamp·feature·raw signal을 확인해야 하며, 이번 추적을 근거로 threshold 또는 feature를 자동 변경하지 않았다.\n"
    (OUT/"analysis_report.md").write_text(report.replace("burst別","burst별"),encoding="utf-8")
    print(json.dumps(concentration,indent=2))
    print(abnormal[["burst_id","FN_count","FN_rate","probability_mean","probability_min","probability_max","start_timestamp"]].to_string(index=False))
    print(overall.to_string(index=False))
    print("Saved:",OUT)


if __name__=="__main__":
    main()
