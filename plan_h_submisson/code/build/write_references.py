from pathlib import Path
import csv,json,urllib.request,hashlib
R=Path(__file__).resolve().parents[2];items=[
('jackson1979','Jackson, J. Edward; Mudholkar, Govind S.','1979','Control Procedures for Residuals Associated With Principal Component Analysis','Technometrics 21(3), 341--349','10.1080/00401706.1979.10489779','https://www.stat.cmu.edu/technometrics/70-79/VOL-21-03/v2103341.pdf','PDF pp.1--3, Sections2--3: squared PCA reconstruction residual; known covariance and distribution assumptions; empirical thresholds here do not inherit theoretical guarantees','full text relevant sections accessed'),
('ledoit2004','Ledoit, Olivier; Wolf, Michael','2004','A well-conditioned estimator for large-dimensional covariance matrices','Journal of Multivariate Analysis 88, 365--411','10.1016/S0047-259X(03)00096-4','https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffceb83f14/wellCond.pdf','Sections2--3: covariance shrinkage; iid assumptions differ from dependent residuals; no F1/F2 guarantee','full text relevant sections accessed'),
('jiang2017','Jiang, Benben; Braatz, Richard D.','2017','Fault detection of process correlation structure using canonical variate analysis-based correlation features','Journal of Process Control 58, 131--138','10.1016/j.jprocont.2017.09.003','https://web.mit.edu/braatzgroup/Jiang_JProCon_2017.pdf','Eqs8--14 and Table2; correlation monitoring motivation; this ridge residual is not a reproduction of CVA Rs/Rr or its 500-point window','full text relevant sections accessed'),
('cawley2010','Cawley, Gavin C.; Talbot, Nicola L. C.','2010','On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation','Journal of Machine Learning Research 11, 2079--2107','','https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf','Sections4--6: selection criterion overfitting and optimistic performance estimates; risk, not proof of overfitting here','full text relevant sections; DOI not found on official page'),
('tatbul2018','Tatbul, Nesime; Lee, Tae Jun; Zdonik, Stan; Alam, Mejbah; Gottschlich, Justin','2018','Precision and Recall for Time Series','Advances in Neural Information Processing Systems 31','','https://proceedings.neurips.cc/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html','Sections2--4 arXiv PDF: range-aware evaluation motivation; we retain pointwise metrics and report range descriptions separately','official proceedings metadata and arXiv PDF accessed; arXiv identifier DOI10.48550/arXiv.1803.03639 is not a publication DOI'),
('chicco2020','Chicco, Davide; Jurman, Giuseppe','2020','The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation','BMC Genomics 21, 6','10.1186/s12864-019-6413-7','https://link.springer.com/article/10.1186/s12864-019-6413-7','Metric definitions/discussion; MCC uses all four confusion counts; extra metric does not create independent evidence','publisher full text accessed'),
('saito2015','Saito, Takaya; Rehmsmeier, Marc','2015','The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets','PLOS ONE 10(3), e0118432','10.1371/journal.pone.0118432','https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432','PR evaluation under imbalance; project FP budget and F2 selection are not prescribed by this paper','publisher full text accessed'),
('bergmeir2018','Bergmeir, Christoph; Hyndman, Rob J.; Koo, Bonsoo','2018','A note on the validity of cross-validation for evaluating autoregressive time series prediction','Computational Statistics & Data Analysis 120, 70--83','10.1016/j.csda.2017.11.003','https://robjhyndman.com/papers/cv-wp.pdf','Sections2--4: conditions for autoregressive CV; not a universal ban on random CV; present holdout reuse is separate','author manuscript accessed'),
('nist','NIST/SEMATECH','2012','What are Variables Control Charts?','e-Handbook of Statistical Methods, Section 6.3.2','','https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm','WECO pattern rules illustrate sensitivity/false-alarm tradeoffs; exact G1 gate and quantile are project choices, not a WECO reproduction','official handbook section accessed'),
]
with (R/'provenance/reference_audit.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['key','authors','year','title','venue','publication_doi','url','supported_claim_and_scope','access']);w.writerows(items)
bib=[]
for key,authors,year,title,venue,doi,url,_,_ in items:
 author=authors.replace(';',' and');author='{NIST/SEMATECH}' if key=='nist' else author
 kind='inproceedings' if key=='tatbul2018' else 'misc' if key=='nist' else 'article';vf='booktitle' if kind=='inproceedings' else 'journal' if kind=='article' else 'howpublished'
 venue=venue.replace('&',r'\&')
 b=f'@{kind}{{{key},\n author = {{{author}}},\n title = {{{title}}},\n year = {{{year}}},\n {vf} = {{{venue}}},\n url = {{{url}}}'
 if doi:b+=f',\n doi = {{{doi}}}'
 if key=='tatbul2018':b+=',\n note = {Author manuscript: arXiv:1803.03639; arXiv DOI 10.48550/arXiv.1803.03639}'
 bib.append(b+'\n}\n')
(R/'paper/references.bib').write_text('\n'.join(bib))
# Archive official guidelines as source evidence; no third-party paper PDFs redistributed.
for name,url in [('author','https://icml.cc/Conferences/2026/AuthorInstructions'),('call','https://icml.cc/Conferences/2026/CallForPapers'),('reviewer','https://icml.cc/Conferences/2026/ReviewerInstructions')]:
 try:
  b=urllib.request.urlopen(url,timeout=30).read();(R/'provenance/references'/f'icml2026_{name}.html').write_bytes(b)
 except Exception as e:(R/'provenance/references'/f'{name}_access_error.txt').write_text(str(e))
print('reference audit',len(items))
