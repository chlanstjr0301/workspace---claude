"""Read-only diagnosis of bursts 619/620; no classifier training or tuning.

Reproduce notebook raw features; recover fold scaling from saved train indices.
Recover intercept via saved probability logit minus saved weighted features.
All conclusions concern the existing five-feature representation, not labels.
"""
import ast
import itertools
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from logistic_current_acf_ablation import sha, table

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/logistic_fn_feature_analysis"
TRACK=ROOT/"results/logistic_fn_burst_tracking"
FN_IDS=[619,620]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plots=OUT/"plots"
    plots.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"]=str(OUT/".matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    notebook=ROOT/"reference/소성가공.ipynb"
    original=ROOT/"src/LSTM-AutoEncoder/baseline_original.py"
    assert sha(original)==original.with_suffix(".sha256").read_text().split()[0].lower()
    input_files=[notebook,original,ROOT/"data/raw/press_data_normal.csv",ROOT/"data/raw/outlier_data.csv",
                 TRACK/"validation_predictions.csv",TRACK/"fold_coefficients.csv",TRACK/"cv_split_indices.json"]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in input_files}
    nb=json.loads(notebook.read_text(encoding="utf-8"))
    sources={i:"".join(c["source"]) for i,c in enumerate(nb["cells"])}
    features=ast.literal_eval(next(n.value for n in ast.parse(sources[38]).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="features_5" for t in n.targets)))
    assert features==["AI0_RMS","AI0_MaxAbs","AI1_RMS","AI1_PeakToPeak","AI1_Kurtosis"]
    ns=dict(np=np,pd=pd,stats=stats,data=pd.concat([pd.read_csv(input_files[2]),pd.read_csv(input_files[3])],ignore_index=True))
    with warnings.catch_warnings(record=True):
        for i in [6,7,8,9,27]:
            exec(compile(sources[i],f"notebook_cell_{i}","exec"),ns)
    clean=ns["featureData_clean"].copy()
    previous=pd.read_csv(ROOT/"results/logistic_current_acf_ablation/burst_feature_dataset.csv",index_col="notebook_feature_index")
    pd.testing.assert_frame_equal(clean,previous.drop(columns="Current_ACF_PeakStrength"),check_names=False)
    clean["group"]=np.where(clean.Equipment_state==0,"Normal",
        np.where(clean.burst_id.isin(FN_IDS),"FN Abnormal","Detected Abnormal"))
    assert clean.groupby("group").size().to_dict()=={"Detected Abnormal":18,"FN Abnormal":2,"Normal":575}
    normal=clean[clean.group=="Normal"]
    detected=clean[clean.group=="Detected Abnormal"]
    fn=clean[clean.group=="FN Abnormal"].sort_values("burst_id")
    group_rows=[]
    for group in ["Normal","Detected Abnormal","FN Abnormal"]:
        for feature in features:
            v=clean.loc[clean.group==group,feature]
            group_rows.append(dict(group=group,feature=feature,count=len(v),mean=v.mean(),median=v.median(),
                std=v.std(ddof=1),min=v.min(),Q1=v.quantile(.25),Q3=v.quantile(.75),max=v.max()))
    group_stats=pd.DataFrame(group_rows)
    comparison=[]
    for _,row in fn.iterrows():
        for feature in features:
            n=group_stats[(group_stats.group=="Normal")&(group_stats.feature==feature)].iloc[0]
            a=group_stats[(group_stats.group=="Detected Abnormal")&(group_stats.feature==feature)].iloc[0]
            x=row[feature]
            status="inside normal IQR" if n.Q1<=x<=n.Q3 else (
                "inside normal range but outside IQR" if n["min"]<=x<=n["max"] else "outside normal range")
            comparison.append(dict(burst_id=int(row.burst_id),feature=feature,value=x,
                normal_mean=n["mean"],normal_median=n["median"],normal_std=n["std"],
                normal_min=n["min"],normal_Q1=n.Q1,normal_Q3=n.Q3,normal_max=n["max"],
                detected_abnormal_mean=a["mean"],detected_abnormal_median=a["median"],detected_abnormal_std=a["std"],
                detected_abnormal_Q1=a.Q1,detected_abnormal_Q3=a.Q3,status=status,
                z_normal=(x-n["mean"])/n["std"] if n["std"] else np.nan,
                z_detected_abnormal=(x-a["mean"])/a["std"] if a["std"] else np.nan,
                median_distance_normal_std=abs(x-n["median"])/n["std"] if n["std"] else np.nan))
    comparison=pd.DataFrame(comparison)
    fn[["burst_id","Equipment_state","n_samples"]+features].to_csv(OUT/"fn_619_620_feature_values.csv",index=False,encoding="utf-8-sig")
    group_stats.to_csv(OUT/"group_feature_statistics.csv",index=False,encoding="utf-8-sig")
    comparison.to_csv(OUT/"fn_vs_normal_zscores.csv",index=False,encoding="utf-8-sig")
    # Boxplots: two reference distributions, then individually marked FN cases.
    for feature in features:
        fig,ax=plt.subplots(figsize=(8,5))
        ax.boxplot([normal[feature],detected[feature]],positions=[1,2],widths=.5,
                   tick_labels=["Normal (575)","Detected abnormal (18)"])
        for pos,bid,marker,color in [(3,619,"X","crimson"),(4,620,"*","darkviolet")]:
            ax.scatter(pos,float(fn.loc[fn.burst_id==bid,feature].iloc[0]),marker=marker,s=170,c=color,zorder=5)
        ax.set_xticks([1,2,3,4],["Normal","Detected abnormal","FN 619","FN 620"])
        ax.set_ylabel(feature)
        ax.set_title("Existing signed-signal burst features")
        ax.grid(axis="y",alpha=.2)
        fig.tight_layout(); fig.savefig(plots/f"boxplot_{feature}.png",dpi=180); plt.close(fig)
    # Exploratory pair ranking excludes the two FN cases; no model/feature selection.
    separation={}
    for feature in features:
        pooled=np.sqrt(((len(normal)-1)*normal[feature].var(ddof=1)+(len(detected)-1)*detected[feature].var(ddof=1))/(len(normal)+len(detected)-2))
        separation[feature]=abs(detected[feature].mean()-normal[feature].mean())/pooled if pooled else 0.0
    pair_rows=[dict(feature_x=a,feature_y=b,descriptive_separation_score=separation[a]**2+separation[b]**2)
               for a,b in itertools.combinations(features,2)]
    pairs=pd.DataFrame(pair_rows).sort_values("descriptive_separation_score",ascending=False)
    pairs.to_csv(OUT/"feature_pair_ranking.csv",index=False,encoding="utf-8-sig")
    for rank,(_,pair) in enumerate(pairs.head(3).iterrows(),start=1):
        a,b=pair.feature_x,pair.feature_y
        fig,ax=plt.subplots(figsize=(7,5))
        ax.scatter(normal[a],normal[b],s=15,alpha=.35,c="steelblue",label="Normal")
        ax.scatter(detected[a],detected[b],s=45,c="darkorange",label="Detected abnormal")
        for bid,marker,color in [(619,"X","crimson"),(620,"*","darkviolet")]:
            point=fn[fn.burst_id==bid].iloc[0]
            ax.scatter(point[a],point[b],s=180,marker=marker,c=color,edgecolors="black",label=f"FN {bid}",zorder=5)
        ax.set_xlabel(a); ax.set_ylabel(b); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(plots/f"scatter_top{rank}_{a}_vs_{b}.png",dpi=180); plt.close(fig)
    predictions=pd.read_csv(TRACK/"validation_predictions.csv")
    coefficients=pd.read_csv(TRACK/"fold_coefficients.csv")
    folds=json.loads((TRACK/"cv_split_indices.json").read_text(encoding="utf-8"))
    X=clean[features]
    contribution_rows,decision_rows,distance_rows,intercepts=[],[],[],[]
    max_probability_error=0.0
    for i,fold in enumerate(folds,start=1):
        tr=pd.Index(fold["train_feature_indices"])
        va=pd.Index(fold["test_feature_indices"])
        # Scaling-statistics recovery only. No LogisticRegression.fit call.
        scaler=StandardScaler().fit(X.loc[tr])
        scaled=scaler.transform(X.loc[va])
        coef=coefficients[coefficients.cv_index==i].set_index("feature").loc[features,"coefficient"].to_numpy()
        p=predictions[predictions.cv_index==i].set_index("notebook_feature_index").loc[va]
        probability=p.probability_y1.to_numpy()
        safe=(probability>1e-6)&(probability<1-1e-6)
        assert safe.any()
        recovered=logit(probability[safe])-scaled[safe]@coef
        intercept=float(np.median(recovered))
        score=scaled@coef+intercept
        error=float(np.max(np.abs(expit(score)-probability)))
        max_probability_error=max(max_probability_error,error)
        assert error<1e-10
        np.testing.assert_array_equal((score>0).astype(int),p.predicted_label)
        intercepts.append(dict(cv_index=i,intercept_recovered=intercept,
            intercept_identity_spread=float(np.ptp(recovered)),max_probability_error=error,
            method="median(logit(saved_probability)-scaled_feature_dot_saved_coefficient)"))
        train_normal=clean.loc[tr].Equipment_state.eq(0)
        normal_centroid=scaler.transform(X.loc[tr][train_normal]).mean(axis=0)
        # Only OOF abnormal cases; same scaler and training-normal centroid as each fold.
        for k,idx in enumerate(va):
            row=clean.loc[idx]
            if row.Equipment_state!=1:
                continue
            distance_rows.append(dict(cv_index=i,repeat=(i-1)//5+1,burst_id=int(row.burst_id),
                group=row.group,distance=float(np.linalg.norm(scaled[k]-normal_centroid))))
            if int(row.burst_id) not in FN_IDS:
                continue
            decision_rows.append(dict(cv_index=i,repeat=(i-1)//5+1,fold=(i-1)%5+1,burst_id=int(row.burst_id),
                feature_sum=float(scaled[k]@coef),intercept=intercept,decision_score=float(score[k]),
                reconstructed_probability=float(expit(score[k])),saved_probability=float(probability[k])))
            for f,z,c in zip(features,scaled[k],coef):
                contribution_rows.append(dict(cv_index=i,repeat=(i-1)//5+1,fold=(i-1)%5+1,
                    burst_id=int(row.burst_id),feature=f,standardized_value=float(z),coefficient=float(c),
                    contribution=float(z*c),intercept=intercept,final_decision_score=float(score[k]),
                    probability=float(probability[k])))
    contributions=pd.DataFrame(contribution_rows)
    decisions=pd.DataFrame(decision_rows)
    distances=pd.DataFrame(distance_rows)
    assert len(contributions)==100 and len(decisions)==20 and len(distances)==200
    contribution_summary=contributions.groupby(["burst_id","feature"],sort=False).agg(
        standardized_value_mean=("standardized_value","mean"),coefficient_mean=("coefficient","mean"),
        contribution_mean=("contribution","mean"),contribution_std=("contribution","std"),
        contribution_min=("contribution","min"),contribution_max=("contribution","max")).reset_index()
    decision_summary=decisions.groupby("burst_id").agg(feature_sum_mean=("feature_sum","mean"),
        intercept_mean=("intercept","mean"),decision_score_mean=("decision_score","mean"),
        decision_score_min=("decision_score","min"),decision_score_max=("decision_score","max"),
        probability_mean=("saved_probability","mean")).reset_index()
    dist_summary=distances.groupby(["burst_id","group"],sort=False).distance.agg(["mean","std","min","max"]).reset_index()
    dist_summary=dist_summary.rename(columns={k:"distance_"+k for k in ["mean","std","min","max"]})
    distance_groups=dist_summary.groupby("group").distance_mean.agg(["count","mean","median","min","max"]).reset_index()
    for filename,frame in [("fn_feature_contributions",contributions),("fn_feature_contribution_summary",contribution_summary),
        ("fn_decision_scores",decisions),("fn_decision_summary",decision_summary),
        ("recovered_fold_intercepts",pd.DataFrame(intercepts)),("abnormal_fold_distances",distances),
        ("abnormal_distance_to_normal_centroid",dist_summary),("centroid_distance_group_summary",distance_groups)]:
        frame.to_csv(OUT/f"{filename}.csv",index=False,encoding="utf-8-sig")
    # Representative raw bursts chosen descriptively by distance to each group's median.
    reps=[]
    for group,frame in [("Normal",normal),("Detected Abnormal",detected)]:
        denom=frame[features].std(ddof=1).replace(0,1)
        rank=(((frame[features]-frame[features].median())/denom)**2).sum(axis=1).sort_values()
        reps.extend(dict(group=group,burst_id=int(clean.loc[idx,"burst_id"]),
                         selection="two closest to within-group feature median (standardized by group std)") for idx in rank.index[:2])
    reps.extend(dict(group="FN Abnormal",burst_id=bid,selection="specified FN case") for bid in FN_IDS)
    reps=pd.DataFrame(reps)
    signal_rows=[]
    channels=["AI0_Vibration","AI1_Vibration","AI2_Current"]
    for _,rep in reps.iterrows():
        raw=ns["data"][ns["data"].burst_id==rep.burst_id].copy()
        raw["elapsed_seconds"]=(raw.TimeStamp-raw.TimeStamp.iloc[0]).dt.total_seconds()
        raw["analysis_group"]=rep.group
        signal_rows.append(raw[["burst_id","analysis_group","TimeStamp","elapsed_seconds"]+channels])
        fig,axes=plt.subplots(3,1,figsize=(9,6),sharex=True)
        for ax,channel in zip(axes,channels):
            ax.plot(raw.elapsed_seconds,raw[channel],marker=".",linewidth=1)
            ax.set_ylabel(channel); ax.grid(alpha=.2)
        axes[-1].set_xlabel("Seconds from burst start")
        fig.suptitle(f"{rep.group} burst {rep.burst_id}, N={len(raw)}, start={raw.TimeStamp.iloc[0]}")
        fig.tight_layout(); fig.savefig(plots/f"raw_burst_{rep.burst_id}.png",dpi=180); plt.close(fig)
    signals=pd.concat(signal_rows,ignore_index=True)
    signals.to_csv(OUT/"representative_raw_signals.csv",index=False,encoding="utf-8-sig")
    reps.to_csv(OUT/"raw_representatives.csv",index=False,encoding="utf-8-sig")
    fig,axes=plt.subplots(3,6,figsize=(19,8),sharey="row")
    for col,(_,rep) in enumerate(reps.iterrows()):
        raw=signals[signals.burst_id==rep.burst_id]
        for row,channel in enumerate(channels):
            ax=axes[row,col]
            ax.plot(raw.elapsed_seconds,raw[channel],marker=".",markersize=3,linewidth=1)
            ax.grid(alpha=.2)
            if col==0: ax.set_ylabel(channel)
            if row==0: ax.set_title(f"{rep.group}\nBurst {rep.burst_id}, N={len(raw)}",fontsize=9)
            if row==2: ax.set_xlabel("Seconds")
    fig.suptitle("Raw signed signals: common y-axis per sensor; full burst duration")
    fig.tight_layout(); fig.savefig(plots/"raw_signal_comparison.png",dpi=160); plt.close(fig)
    # Brief raw statistics: no new model features or inputs.
    raw_stats=[]
    for (bid,group),frame in signals.groupby(["burst_id","analysis_group"]):
        for ch in channels:
            x=frame[ch].to_numpy()
            raw_stats.append(dict(burst_id=bid,group=group,channel=ch,n_samples=len(x),
                duration_seconds=float(frame.elapsed_seconds.max()),mean=x.mean(),RMS=np.sqrt(np.mean(x*x)),
                min=x.min(),max=x.max()))
    pd.DataFrame(raw_stats).to_csv(OUT/"representative_raw_statistics.csv",index=False,encoding="utf-8-sig")
    closest=comparison.groupby("feature").median_distance_normal_std.mean().sort_values()
    most_different=comparison.assign(abs_z_detected=comparison.z_detected_abnormal.abs()).groupby("feature").abs_z_detected.mean().sort_values(ascending=False)
    pushes=contribution_summary.sort_values(["burst_id","contribution_mean"]).groupby("burst_id").first().reset_index()
    normal_scores=predictions[predictions.true_label==0].groupby("burst_id").probability_y1.mean()
    score_overlap=pd.DataFrame([dict(burst_id=bid,FN_mean_probability=float(decision_summary.loc[
        decision_summary.burst_id==bid,"probability_mean"].iloc[0]),normal_mean_probability_median=normal_scores.median(),
        normal_empirical_CDF_at_FN_probability=float((normal_scores<=decision_summary.loc[
            decision_summary.burst_id==bid,"probability_mean"].iloc[0]).mean())) for bid in FN_IDS])
    score_overlap.to_csv(OUT/"normal_score_overlap.csv",index=False,encoding="utf-8-sig")
    for p in input_files:
        assert sha(p)==hashes[str(p.relative_to(ROOT))]
    metadata=dict(reference_hashes=hashes,features=features,class_weight=None,threshold=.5,
        classifier_retrained=False,threshold_tuned=False,model_features_added=False,
        intercept_source="recovered algebraically from existing fold probabilities and coefficients; not a fresh fit",
        max_reconstructed_probability_error=max_probability_error,
        distance_definition="OOF feature scaled using original fold train statistics; Euclidean distance to that fold training-normal centroid; summarize each burst's 10 OOF distances",
        pair_ranking="sum of squared absolute pooled-SD mean differences for Normal vs Detected Abnormal; descriptive only, excludes FN; no redundancy correction",
        closest_to_normal_median_feature=closest.index[0],largest_abs_detected_z_feature=most_different.index[0])
    (OUT/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    report="# FN burst 619/620: five-feature space diagnosis\n\n"
    report+=f"입력: `{features}`. Notebook cell38 best5, AI0_Kurtosis 제외. "
    report+="Normal575, Detected Abnormal18, FN Abnormal2. 기존 signed preprocessing, burst 정의·길이4 기준 그대로. "
    report+="모델 재학습·threshold tuning·모델 feature 추가 없음. 기존 None/threshold0.5의 OOF 결과만 설명한다.\n\n"
    report+="## Group statistics (std ddof=1)\n\n"+table(group_stats)+"\n\n"
    report+="## FN values and distribution overlap\n\n"+table(comparison)+"\n\n"
    report+="z_normal은 normal mean/std 기준이고 median_distance_normal_std는 normal median과의 거리다. "
    report+="둘은 다른 지표다. IQR/min-max 포함 여부는 개별 feature의 주변 분포에 관한 것이며 5D joint overlap을 증명하지 않는다.\n\n"
    report+="## Feature pair selection\n\n"+table(pairs.head(3))+"\n\n"
    report+="Pair 순위는 Normal–Detected Abnormal의 pooled-SD mean difference 제곱합이며 모델 최적 feature 선택이 아니다. "
    report+="FN을 pair ranking에 사용하지 않았다. Correlated axes를 보정하지 않아 중복 설명력이 포함될 수 있다.\n\n"
    report+="## 5D normal centroid distances\n\n"+table(dist_summary)+"\n\n"+table(distance_groups)+"\n\n"
    report+="각 abnormal의 기존 validation fold에서 train-only StandardScaler 통계를 복원하고 그 fold train 정상 centroid와의 거리를 계산했다. "
    report+="10회 OOF distance의 평균을 burst별로 비교한다. 다른 scaler로 전체 데이터에 새 model을 fit하지 않았다.\n\n"
    report+="## Existing logistic decision decomposition\n\n"
    report+="각 fold의 저장 계수를 사용한다. Scaler는 같은 train index에서 통계만 재계산했다. "
    report+="Intercept가 저장되어 있지 않아 nonsaturated validation probability에서 "
    report+="median(logit(p)−z·coef)로 복원했다. 이는 classifier 학습이 아닌 역산이다. "
    report+=f"모든5950 validation probability의 최대 복원 오차={max_probability_error:.3g}; 예측 일치도도 검사했다.\n\n"
    report+=table(contribution_summary)+"\n\n"+table(decision_summary)+"\n\n"
    report+="평균 contribution은 mean(z×coef)이며 mean(z)×mean(coef)와 같다고 가정하지 않는다. "
    report+="평균 score와 평균 probability도 sigmoid 관계를 직접 적용하면 일치하지 않을 수 있다. "
    report+="Fold별 정확한 분해는 fn_feature_contributions.csv 및 fn_decision_scores.csv에 저장했다.\n\n"
    report+="## Raw signal comparison\n\n"+table(reps)+"\n\n"
    report+="대표는 각 그룹 feature median에 가장 가까운2개로 고정했으며 잘 분리되는 그림을 보고 골라낸 것이 아니다. "
    report+="채널별 비교 그림은 같은 y축 범위를 사용하고 전체 burst 시간 구간을 유지한다. 짧은 FN burst의 추정치 변동에 유의한다.\n\n"
    report+="![Raw signals](plots/raw_signal_comparison.png)\n\n"
    for f in features:
        report+=f"![{f}](plots/boxplot_{f}.png)\n\n"
    for rank,(_,pair) in enumerate(pairs.head(3).iterrows(),start=1):
        report+=f"![Pair {rank}](plots/scatter_top{rank}_{pair.feature_x}_vs_{pair.feature_y}.png)\n\n"
    report+="## Key findings\n\n"
    report+=f"- Normal median과의 평균 표준화 거리가 가장 작은 feature: {closest.index[0]} ({closest.iloc[0]:.4f}).\n"
    report+=f"- Detected Abnormal 기준 평균 |z|가 가장 큰 feature: {most_different.index[0]} ({most_different.iloc[0]:.4f}).\n"
    for bid in FN_IDS:
        cc=comparison[comparison.burst_id==bid]
        push=pushes[pushes.burst_id==bid].iloc[0]
        report+=f"- Burst {bid}: Normal IQR 포함 {int((cc.status=='inside normal IQR').sum())}/5, "
        report+=f"normal min-max 포함 {int((cc.status!='outside normal range').sum())}/5. "
        report+=f"가장 큰 negative feature contribution: {push.feature}, mean={push.contribution_mean:.4f}.\n"
    report+="\nFN은619/620에10회씩 반복된다. 확률이 threshold 주변에서 흔들리는 경우와 구별해 "
    report+="현재 5-feature/linear decision의 representation limitation 근거를 검토한다. "
    report+="관찰만으로 feature 요약의 정보 손실과 linear model의 제약을 완전히 분리할 수 없다. "
    report+="개별 범위 포함·centroid 근접·기존 negative score를 종합하며 라벨 오류를 단정하지 않는다.\n"
    report+="\n## 질문별 결론\n\n"
    report+="1. **Normal 분포와 겹치는가?** 619는5개 모두 정상 min-max 내부이고 AI1_RMS/AI1_PeakToPeak는 정상 IQR 내부다. "
    report+="620은3개가 정상 min-max 내부이며 AI1_RMS/AI1_PeakToPeak는 정상 최솟값보다 낮다. "
    report+="따라서 둘 모두 정상 분포와 완전히 같다고 할 수 없다. 620의 낮은 진동도 현재 linear decision에서는 이상 쪽으로 이동시키지 못한다.\n"
    report+="2. **가장 정상에 가까운 feature?** Normal median 거리/normal std 기준 두 burst 평균은 AI1_RMS가 가장 작다. "
    report+="개별로619는AI1_RMS,620은AI1_Kurtosis가 가장 가깝다.\n"
    report+="3. **탐지된 이상18개와 가장 다른 feature?** 표준화된 차이로AI1_PeakToPeak가 가장 크다. "
    report+="619의z=-3.3832,620의z=-3.9848이며 둘 모두 훨씬 작은 진동 폭이다. AI0_RMS/MaxAbs도 낮다.\n"
    report+="4. **정상 쪽으로 미는 contribution?** 두 경우 모두AI0_MaxAbs가 가장 큰 음의 feature contribution이다 "
    report+="(619:-0.7424,620:-0.5607). AI0_RMS도 음의 기여이고,620은AI1_PeakToPeak도-0.4580이다. "
    report+="Intercept 약-5.8/-5.9도 score에 크게 작용한다. AI1_Kurtosis는 약+0.46/+0.43의 이상 쪽 기여여서 이를 상쇄하지 못한다.\n"
    report+="5. **Threshold보다 representation 문제인가?** 기존 score 공간에서 두 경우는 경계가 아니라 정상 score 영역의 낮은 쪽이다. "
    report+="정상 burst 평균 probability 중앙값은약0.003346이고,619/620의 평균은각각0.001333/0.001028이다. "
    report+="정상 score 분포의 경험적CDF 위치도약4.17%/2.09%다. "
    report+="이는단순한 경계 근처 오분류보다 현재5-feature/linear score가 두 이상 사례를 분리하지 못한다는 근거다. "
    report+="Threshold 변경의 효과를 실험하거나 불가능하다고 증명한 것은 아니다.\n\n"
    report+=table(score_overlap)+"\n\n"
    report+="Raw에서FN619/620은10/15 sample의0.9/1.4초 짧은 burst이며 진동 크기가 작다. "
    report+="선택한 정상 대표240/348과 달리 전류는FN의 관측 구간에서 양수이며 mean≈198.48/180.09다. "
    report+="현재5개 모델 입력에는 전류가 없다. 이 관찰은 표현의 누락 가능성을 보여주지만 "
    report+="짧은 구간·수집 상태 영향과 분리되지 않았고, 전류를 추가하면 탐지된다는 증거도 아니다. "
    report+="원자료 label은 그대로 유지하며 라벨 오류·고장 원인을 단정하지 않는다.\n"
    (OUT/"analysis_report.md").write_text(report,encoding="utf-8")
    print(comparison[["burst_id","feature","value","status","z_normal","z_detected_abnormal","median_distance_normal_std"]].to_string(index=False))
    print("Closest to Normal median:",closest.to_string())
    print("Most different from Detected Abnormal:",most_different.to_string())
    print(contribution_summary.to_string(index=False))
    print(decision_summary.to_string(index=False))
    print(distance_groups.to_string(index=False))
    print("Probability reconstruction max error:",max_probability_error)
    print("Saved:",OUT)


if __name__=="__main__":
    main()
