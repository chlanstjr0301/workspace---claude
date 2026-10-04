# Frozen-model burst length diagnostic

No classifier was fitted. All 50 saved coefficients and recovered intercepts were reused with original fold scalers; original probabilities reproduced to maximum error 4.218847493575595e-15. Signed notebook features and strict threshold >0.5 are unchanged. No ACF, imputation, threshold tuning or label changes.

Original full dataset: 575 normal + 20 abnormal; recall 0.9 (619/620 always FN), normal FP 0. Common N>=20 has 13 originally detected abnormal parents; the other five originally TP parents (600,604,609,610,614) are ineligible for length20. Original common-cohort recall is 1.0.

Primary metrics (same N>=20 parents; dependent crop OOF observations):

length,cohort,aggregation,n_unique_bursts,n_predictions,n_unique_abnormal_bursts,n_unique_normal_bursts,n_abnormal_predictions,n_normal_predictions,abnormal_detection_rate,abnormal_probability_mean,abnormal_probability_median,normal_FP_rate,normal_probability_mean,Precision,Recall,F1,Balanced_Accuracy

original,common_N_ge20,crop_OOF,465,4650,13,452,130,4520,1.0,0.949737875169731,0.9844904093793891,0.0,0.004135178848025407,1.0,1.0,1.0,1.0

20,common_N_ge20,crop_OOF,465,13710,13,452,390,13320,0.7487179487179487,0.7200920950720118,0.9834593381672205,0.0,0.0044058870193477775,1.0,0.7487179487179487,0.8563049853372434,0.8743589743589744

15,common_N_ge20,crop_OOF,465,13950,13,452,390,13560,0.6666666666666666,0.6457434032899662,0.9699531524032988,0.0,0.004378874564730539,1.0,0.6666666666666666,0.8,0.8333333333333333

10,common_N_ge20,crop_OOF,465,13950,13,452,390,13560,0.5615384615384615,0.5377240262089117,0.8506098356746044,0.0,0.0048177202266234355,1.0,0.5615384615384615,0.7192118226600985,0.7807692307692308



Transition counts (honest eligible-original-TP denominators, not all 18):

cohort,length,eligible_original_TP_parents,affected_any_FN,persistent_all_crop_fold_FN,FN_predictions,n_predictions

actually_shorter,10,15,12,1,191,450

actually_shorter,15,14,9,1,130,420

actually_shorter,20,13,8,0,98,390

common_N_ge20,10,13,11,1,171,390

common_N_ge20,15,13,9,1,130,390

common_N_ge20,20,13,8,0,98,390



Any FN means at least one position/fold FN; persistent FN means every position/fold FN. Normal TN->FP uses identical definitions. Each physical crop contributes once to drift summaries, rather than ten repeated fold observations. Standardized feature sensitivity ranking:

feature
AI0_RMS           7.276877
AI0_MaxAbs        6.268196
AI1_Kurtosis      3.117971
AI1_PeakToPeak    2.262663
AI1_RMS           1.165889

Nearest originally detected abnormal crops to FN619/620 (five-feature Euclidean distance divided by original normal SD):

fn_burst_id,length,crop_id,parent_burst_id,position,crop_distance,parent_distance,crop_probability_mean,parent_probability_mean

620,15,615_15_0,615,start,0.6989282307321292,25.00985745802885,0.004583976333448759,0.9961017372448758

620,20,616_20_15,616,middle,0.9925109146585159,15.702055827289314,0.001576994142846754,0.8040930228690482

619,20,616_20_15,616,middle,1.0774365311209009,15.722221983966193,0.001576994142846754,0.8040930228690482

620,10,615_10_0,615,start,1.1973918820506866,25.00985745802885,0.004124411770071576,0.9961017372448758

619,10,603_10_21,603,end,1.5037624862639707,13.762872363910294,0.001342807174582396,0.8399654693128227

619,15,615_15_0,615,start,1.5052166693847089,25.233108640982014,0.004583976333448759,0.9961017372448758



Parent-repeat results average probabilities across deduplicated positions before applying >0.5. Parent-allrepeat results additionally average repeats. These are diagnostic aggregations, not a change to the official evaluation protocol. Physical crops and repeated validation measurements are dependent; counts are not independent sample counts. Distances describe representation similarity and cannot establish causation or label error.
Among actually shortened originally TP parents, any-FN transitions affect 12/15 eligible at length10 and 9/14 at length15 (out of 18 original TPs, three/four respectively are ineligible). Only burst603 is persistent FN across every position and all 10 repeats at both lengths; many parents lose detection at some positions rather than all positions.

AI0_RMS has the largest mean absolute drift standardized by original normal SD at every length, followed by AI0_MaxAbs. Raw-unit drift cannot fairly rank unlike feature units; AI1_Kurtosis has large raw numerical changes but is not the largest standardized drift.

The short-window explanation for 619/620 gains descriptive support: originally detected long abnormal bursts can yield short low-probability crops close to their five-feature vectors. At matching lengths, nearest crops are 603/end/10 for619 (distance1.504 versus parent13.763, mean P=.00134) and615/start/15 for620 (distance.699 versus parent25.010, mean P=.00458). This supports a length-and-window-content effect, not proof that length alone caused their original FN. Their observed raw content remains distinct and no label error is inferred.
