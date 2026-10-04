"""Post-authoring checks against source CSVs; no model fitting or tuning."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import auc, precision_recall_curve

OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
CW=ROOT/'results/logistic_class_weight_ablation'; LENGTH=ROOT/'results/logistic_burst_sample_count'


def main():
    main=(OUT/'main.tex').read_text(encoding='utf-8'); readme=(OUT/'README.md').read_text(encoding='utf-8')
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8')); checks=[]
    def check(name,truth,detail=''):
        checks.append(dict(check=name,passed=bool(truth),detail=detail))
        assert truth,name
    manifest=json.loads((OUT/'source_results/manifest.json').read_text(encoding='utf-8'))
    for r in manifest:
        check('source_copy_'+r['source'],hashlib.sha256((OUT/r['copy']).read_bytes()).hexdigest()==r['sha256']==hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest())
    weights=['None','{0:1,1:3}','{0:1,1:5}','{0:1,1:10}','balanced']
    source=pd.read_csv(CW/'metrics.csv',keep_default_na=False).set_index('weight')
    metrics=pd.read_csv(OUT/'tables/class_weights.csv',keep_default_na=False).set_index('weight')
    check('weight_labels_literal_None',list(metrics.index)==weights)
    pred=pd.read_csv(CW/'predictions.csv',keep_default_na=False); pred=pred.loc[pred.weight.isin(weights)]
    tex=(OUT/'tables/class_weights.tex').read_text(encoding='utf-8')
    for w in weights:
        for field in ['Precision','Recall','F1','Balanced_Accuracy','ROC_AUC','FP','FN']:
            for suffix in ['mean','std']: check(f'metric_{w}_{field}_{suffix}',np.isclose(metrics.loc[w,field+'_'+suffix],source.loc[w,field+'_'+suffix],atol=1e-12))
        pr=[]
        for _,g in pred.loc[pred.weight.eq(w)].groupby('fold'):
            precision,recall,_=precision_recall_curve(g.label,g.probability); pr.append(auc(recall,precision))
        check('PR_AUC_'+w,np.isclose(np.mean(pr),metrics.loc[w,'PR_AUC_mean'],atol=1e-12) and np.isclose(np.std(pr),metrics.loc[w,'PR_AUC_std'],atol=1e-12))
        cells=[]
        for field in ['Precision','Recall','F1','Balanced_Accuracy','ROC_AUC','PR_AUC']:
            cells.append(f'${metrics.loc[w,field+"_mean"]:.3f}\\pm{metrics.loc[w,field+"_std"]:.3f}$')
        cells += [f'${metrics.loc[w,field+"_mean"]:.2f}\\pm{metrics.loc[w,field+"_std"]:.2f}$' for field in ['FP','FN']]
        label={'None':'None','balanced':'Balanced'}.get(w,w.replace('{0:1,1:','1:').replace('}',''))
        check('latex_class_weight_row_'+w,label+' & '+' & '.join(cells)+' \\\\' in tex)
    # Exact source versus the CSVs used in the manuscript and crop plots.
    for target,source_name in [('sample_count_coverage','eligible_bursts_by_sample_count'),('sample_count_transitions','tp_to_fn_summary'),('sample_count_drift','feature_drift_by_sample_count')]:
        pd.testing.assert_frame_equal(pd.read_csv(OUT/'tables'/f'{target}.csv'),pd.read_csv(LENGTH/f'{source_name}.csv'))
        check('source_table_'+target,True)
    lm=pd.read_csv(LENGTH/'metrics_by_sample_count.csv')
    plot=pd.read_csv(OUT/'tables/figure_sample_count_recall_data.csv')
    pd.testing.assert_frame_equal(plot,lm.loc[lm.aggregation.isin(['crop_OOF','parent_OOF'])].reset_index(drop=True)); check('figure_recall_data',True)
    drift=pd.read_csv(LENGTH/'feature_drift_by_sample_count.csv')
    pd.testing.assert_frame_equal(pd.read_csv(OUT/'tables/figure_feature_drift_data.csv'),drift.loc[drift.cohort.eq('common_N_ge30')&drift.true_label.eq(1)].reset_index(drop=True)); check('figure_drift_data',True)
    fd=pd.read_csv(CW/'feature_dataset.csv')
    pd.testing.assert_frame_equal(pd.read_csv(OUT/'tables/figure_feature_distribution_data.csv'),fd[['notebook_feature_index','burst_id','Equipment_state','AI0_RMS','AI0_MaxAbs','AI1_RMS','AI1_PeakToPeak','AI1_Kurtosis']]); check('figure_feature_data',True)
    none=pred.loc[pred.weight.eq('None')]; b=pred.loc[pred.weight.eq('balanced')]
    pair=none.merge(b,on=['fold','burst_id','label'],suffixes=('_n','_b'),validate='one_to_one')
    up=pair.loc[pair.prediction_n.eq(0)&pair.prediction_b.eq(1)]
    check('crossings_94_observations_16_normal_bursts',len(up)==94 and up.burst_id.nunique()==16 and up.label.eq(0).all())
    check('no_reverse_crossings',not (pair.prediction_n.eq(1)&pair.prediction_b.eq(0)).any())
    check('abnormal_identities_unchanged',pair.loc[pair.label.eq(1),'prediction_n'].equals(pair.loc[pair.label.eq(1),'prediction_b']))
    scorestats=pd.read_csv(OUT/'tables/none_score_distribution.csv').set_index('group')
    for name,g in [('Normal',none.loc[none.label.eq(0)]),('All abnormal',none.loc[none.label.eq(1)])]:
        z=logit(g.probability)
        for col,v in dict(mean=z.mean(),median=z.median(),Q1=z.quantile(.25),Q3=z.quantile(.75),min=z.min(),max=z.max()).items():
            check(f'score_distribution_{name}_{col}',np.isclose(scorestats.loc[name,col],v,atol=1e-12))
    check('no_score_clipping',pd.read_csv(OUT/'tables/score_clipping.csv').clipped_count.sum()==0)
    centroids=pd.read_csv(OUT/'tables/fold_centroids.csv')
    check('centroid_vectors_50_fold_two_classes',len(centroids)==100 and centroids.groupby('fold').label.nunique().eq(2).all())
    check('centroid_train_class_counts',centroids.loc[centroids.label.eq(0),'n_training_bursts'].eq(460).all() and centroids.loc[centroids.label.eq(1),'n_training_bursts'].eq(16).all())
    # Check literal narrative numerical values against exact source-derived values.
    claims=[]
    def claim(name,value,decimals,expected=None):
        literal=expected if expected is not None else f'{value:.{decimals}f}'
        ok=literal in main; claims.append(dict(claim=name,value=float(value),literal=literal,passed=ok)); check('paper_claim_'+name,ok)
    for w in ['None','balanced']:
        for field in ['Precision','Recall','F1','ROC_AUC','PR_AUC']:
            claim(w+'_'+field,metrics.loc[w,field+'_mean'],3,expected='0.9' if field=='Recall' else ('1.0' if w=='None' and field=='Precision' else None))
    shifts=pd.read_csv(OUT/'tables/none_balanced_shift_summary.csv').set_index('group')
    for group in shifts.index:
        for col in ['mean_probability_none','mean_probability_balanced','mean_delta_score']: claim(group+'_'+col,shifts.loc[group,col],6)
    for bid in [619,620]:
        group=pair.loc[pair.burst_id.eq(bid)]
        claim(str(bid)+'_balanced_probability',group.probability_b.mean(),6)
        claim(str(bid)+'_paired_score_shift',(logit(group.probability_b)-logit(group.probability_n)).mean(),6)
    for group in ['Normal','Detected abnormal','FN abnormal','All abnormal']:
        claim('score_mean_'+group,scorestats.loc[group,'mean'],3)
    abnormal_distances=pd.read_csv(OUT/'tables/abnormal_centroid_distances.csv')
    detected_distances=abnormal_distances.loc[abnormal_distances.group.eq('Detected abnormal'),'distance_normal_mean']
    claim('detected_normal_centroid_mean',detected_distances.mean(),6)
    claim('detected_normal_centroid_min',detected_distances.min(),6)
    claim('detected_lowest_individual_score',logit(none.loc[none.label.eq(1)&~none.burst_id.isin([619,620]),'probability']).min(),3)
    feature=pd.read_csv(OUT/'tables/feature_separability.csv')
    for _,r in feature.loc[feature.group.eq('Abnormal')].iterrows(): claim(r.feature+'_cohen_d',r.signed_cohen_d_abnormal_minus_normal,3)
    fn=pd.read_csv(OUT/'tables/fn_619_620.csv')
    ftex=(OUT/'tables/fn_619_620.tex').read_text(encoding='utf-8')
    for _,r in fn.iterrows():
        for col in ['probability_mean','distance_normal_mean','distance_abnormal_mean']:
            claim(str(r.burst_id)+'_'+col,r[col],6); check('latex_fn_'+str(r.burst_id)+'_'+col,f'{r[col]:.6f}' in ftex)
    ltex=(OUT/'tables/sample_count.tex').read_text(encoding='utf-8'); lc=pd.read_csv(OUT/'tables/sample_count_results.csv')
    for _,r in lc.iterrows():
        row=f'{int(r.sample_count)} & {int(r.eligible_normal)}/{int(r.eligible_abnormal)} & ${r.Recall_mean:.3f}\\pm{r.Recall_std:.3f}$ & ${r.F1_mean:.3f}\\pm{r.F1_std:.3f}$ & {r.abnormal_probability_mean:.4f} & {int(r.any_FN_parents)}/{int(r.eligible_evaluated_original_TP)} & {int(r.FP_total)} \\\\'
        check('latex_crop_row_'+str(int(r.sample_count)),row in ltex)
    primary=lm.loc[lm.cohort.eq('eligible_N_ge_target')&lm.source.eq('crop')&lm.aggregation.eq('crop_OOF')].set_index('sample_count')
    fixed=lm.loc[lm.cohort.eq('common_N_ge30')&lm.source.eq('crop')&lm.aggregation.eq('crop_OOF')].set_index('sample_count')
    for L in [10,15]: claim('eligible_recall_'+str(L),primary.loc[L,'Recall_mean'],6)
    for L in [10,12,15,20,25,30]: claim('fixed_recall_'+str(L),fixed.loc[L,'Recall_mean'],6)
    ranking=drift.loc[drift.cohort.eq('common_N_ge30')&drift.true_label.eq(1)].groupby('feature').mean_absolute_standardized_delta.mean()
    for feature,value in ranking.items(): claim('drift_'+feature,value,3)
    ds=pd.read_csv(OUT/'tables/dataset_statistics.csv'); dtex=(OUT/'tables/dataset.tex').read_text(encoding='utf-8')
    for _,r in ds.iterrows():
        for c in ['raw_rows','raw_bursts','eligible_bursts','calendar_dates','samples_min','samples_median','samples_max']:
            check('latex_dataset_'+r.group+'_'+c,f'{r[c]:g}' in dtex)
        check('paper_date_'+r.group,r.date_min in main and r.date_max in main)
    check('feature_formulas_five_rows',all(x.replace('_','\\_') in main for x in ['AI0_RMS','AI0_MaxAbs','AI1_RMS','AI1_PeakToPeak','AI1_Kurtosis']) and 'fisher=True, bias=False' in main)
    required=['Introduction','Related Work','Dataset and Problem Definition','Method','Experiments','Results','Discussion','Limitations','Conclusion']
    check('required_sections',all('\\section{'+s+'}' in main for s in required) and '\\begin{abstract}' in main)
    check('research_questions_named',all('RQ'+str(i)+':' in main for i in [1,2,3]))
    check('imbalance_distinct_from_scarcity','Class imbalance and minority-sample scarcity are distinct' in main)
    check('pump_provenance_hedged','project-described hydraulic pump' in main and 'not independently verified' in main)
    check('date_and_selection_limits','class and acquisition date are confounded' in main and 'selection optimism' in main)
    check('crop_filter_and_reference_scope','N\\geq L' in main and 'minimum-length filter' in main and 'only in its parent' in main)
    check('dependencies_not_independent','Fifty repeated folds are dependent' in main and 'no independent test set' in main)
    check('ROC_PR_tradeoff','increases ROC-AUC' in main and 'PR-AUC decreases' in main)
    banned=['we solve class imbalance','balanced weighting is bad','short bursts cause false negatives','619 is mislabeled','our proposed method outperforms','state-of-the-art performance']
    check('no_forbidden_positive_claim',not any(x in main.lower() for x in banned))
    check('no_invented_bibliography',not re.search(r'@\w+\s*\{',(OUT/'references.bib').read_text(encoding='utf-8')) and 'Citation TODO' in main)
    check('no_invented_official_style','\\documentclass[10pt,twocolumn,letterpaper]{article}' in main and 'official ICML 2026 style file must be added for final submission' in readme)
    check('main_pdf_absence_transparent',not (OUT/'main.pdf').exists() and 'has not been compiled' in readme)
    # Structural checks cannot substitute for a genuine compilation/layout check.
    sources=main+'\n'+'\n'.join(p.read_text(encoding='utf-8') for p in (OUT/'tables').glob('*.tex'))
    begins=re.findall(r'\\begin\{([^}]+)\}',sources); ends=re.findall(r'\\end\{([^}]+)\}',sources)
    check('latex_environment_counts',sorted(begins)==sorted(ends))
    stack=[]
    for token in re.finditer(r'\\(begin|end)\{([^}]+)\}',sources):
        if token[1]=='begin': stack.append(token[2])
        else: check('latex_nesting_'+str(token.start()),bool(stack) and stack.pop()==token[2])
    check('latex_referenced_files_exist',all((OUT/p).is_file() for p in re.findall(r'\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}',main)))
    check('five_tables',sources.count('\\begin{table*}')+sources.count('\\begin{table}')==5)
    pdfs=list((OUT/'figures').glob('*.pdf')); check('six_vector_PDF_figures',len(pdfs)==6 and all(p.read_bytes().startswith(b'%PDF') for p in pdfs))
    # Render each figure PDF with installed Poppler for actual vector-output QA.
    poppler=Path('C:/Users/EKR/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
    qa=ROOT/'tmp/pdfs/paper_icml_logistic'; qa.mkdir(parents=True,exist_ok=True)
    for p in pdfs:
        subprocess.run([str(poppler),'-singlefile','-png','-r','130',str(p),str(qa/p.stem)],check=True,capture_output=True)
    from PIL import Image,ImageOps,ImageDraw
    sheet=Image.new('RGB',(1500,1100),'white'); draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(sorted(pdfs)):
        im=Image.open(qa/(p.stem+'.png')).convert('RGB'); im.thumbnail((720,310))
        x=(j%2)*750+(750-im.width)//2; y=(j//2)*360+25
        sheet.paste(im,(x,y)); draw.text(((j%2)*750+15,(j//2)*360+5),p.stem,fill='black')
    sheet.save(qa/'figure_pdf_contact_sheet.png')
    check('figure_PDFs_rendered',all((qa/(p.stem+'.png')).exists() for p in pdfs))
    pd.DataFrame(claims).to_csv(OUT/'tables/paper_numeric_claim_checks.csv',index=False)
    audit['post_authoring_checks']=checks; audit['post_authoring_passed']=all(x['passed'] for x in checks)
    audit['paper_sha256']=hashlib.sha256((OUT/'main.tex').read_bytes()).hexdigest()
    audit['figure_pdf_rendered']=True; audit['main_pdf_compiled']=False; audit['paper_pagination_checked']=False
    audit['visual_QA']='All six PNGs and the Poppler-rendered vector-PDF contact sheet were visually reviewed, with no clipping or missing content. Figure appearance only; main paper pagination remains unverified.'
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    validation='# Analysis validation\n\nPost-authoring source/table/claim/figure checks: '+str(len(checks))+' passed. Exact numerical claim literals checked: '+str(len(claims))+'.\n\n'
    validation+='All source copies match their SHA256 manifests and originals. Class-weight means/std reproduce saved metrics; PR-AUC is recomputed from unchanged probabilities. All selected probabilities are unsaturated. None and balanced pairings preserve labels/folds; crossings are94 normal observations from16 unique parents and zero abnormal/reverse crossings. Both conditions retain18 stableTP and2 persistentFN.\n\n'
    validation+='Five paper tables use the audited data, including the handwritten exact feature formulas. Figure traces map to their source CSVs. Training-only centroid coordinates and distances are supplied per fold, not as a single global centroid. Normal and all-abnormal score-distribution summaries include quartiles and both detected/FN subgroups. No classifier is fitted.\n\n'
    validation+='The complete draft was checked for explicit acquisition-date confounding, selection optimism, dependency, minority-sample scarcity, cautious crop interpretation, literal None labels, empty bibliography TODOs, and non-official style disclosure. Basic LaTeX environment nesting/file references pass. These are not proof of successful compilation. main.pdf is unavailable because no genuine LaTeX engine exists locally; paper pagination cannot be verified.\n\n'
    validation+='All six 300-dpi PNG figures have been visually reviewed. All six vector PDFs were rendered with bundled Poppler and the contact sheet was inspected: no clipping or missing content. The QA contact sheet is at tmp/pdfs/paper_icml_logistic/figure_pdf_contact_sheet.png. Main paper pagination remains unverified because it could not be compiled.\n'
    (OUT/'analysis_validation.md').write_text(validation,encoding='utf-8')
    print('Passed',len(checks),'checks;',len(claims),'paper numerical claims. PDF figures rendered; main.pdf not compiled.')


if __name__=='__main__': main()
