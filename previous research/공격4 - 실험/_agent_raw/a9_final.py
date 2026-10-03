# -*- coding: utf-8 -*-
import numpy as np, common
Fn,Tn,Bn,Fa,Ta,Ba=common.get(); i=common.FEAT_NAMES.index
n,o=common.load()
j0,jc,j1=i("AI0_std"),i("corr01"),i("AI1_std")

print("### A9-1  '역위상(rocking)' 은 고장에만 나타나는가 — 정상 16000-19000 구간과 비교")
msk=(Tn>=16000)&(Tn<19000)
print("  정상 16000-19000 window (n=%d): AI0_std=%.4f AI1_std=%.4f corr01=%+.3f"%(msk.sum(),Fn[msk,j0].mean(),Fn[msk,j1].mean(),Fn[msk,jc].mean()))
m2=Fa[:,j0]<=np.quantile(Fn[:,j0],0.99)
print("  이상 window 중 진폭 정상범위 (n=%d):        AI0_std=%.4f AI1_std=%.4f corr01=%+.3f"%(m2.sum(),Fa[m2,j0].mean(),Fa[m2,j1].mean(),Fa[m2,jc].mean()))
print("  정상 15000 이전 window:                  AI0_std=%.4f AI1_std=%.4f corr01=%+.3f"%(Fn[Tn<15000,j0].mean(),Fn[Tn<15000,j1].mean(),Fn[Tn<15000,jc].mean()))
# 음의 corr01 window 비율
for nm,F in [("정상 <15000",Fn[Tn<15000]),("정상 16000-19000",Fn[msk]),("정상 19000-20000",Fn[(Tn>=19000)]),("이상",Fa)]:
    print("  %-18s corr01<0 비율 = %.1f%%  / corr01<-0.3 비율 = %.1f%%"%(nm,100*(F[:,jc]<0).mean(),100*(F[:,jc]<-0.3).mean()))

print("\n### A9-2  §2.2 표가 숨긴 것 — '15000-20000' 한 칸이 평균낸 내용")
show=["AI0_acf1","AI0_std","AI1_acf1","AI1_std","corr01"]
ix=[i(s) for s in show]
for lo,hi in [(0,15000),(15000,16000),(16000,19000),(19000,20000)]:
    m=(Tn>=lo)&(Tn<hi)
    print("  %5d-%-6d n=%4d  "%(lo,hi,m.sum())+"".join("%s=%+.3f  "%(s,v) for s,v in zip(show,Fn[m][:,ix].mean(0))))
print("  => 19000-20000 이 학습구간과 동일 수준으로 복귀. '새 운전체제로 바뀐다'가 아니라 16000-19000 의 일시적 이탈이다.")

print("\n### A9-3  §2.3 표의 'AI2_acf1 정상 5구간 평균 0.927' 유효 숫자")
j=i("AI2_acf1")
for lo,hi in [(0,4000),(4000,8000),(8000,12000),(12000,15000),(15000,20000)]:
    m=(Tn>=lo)&(Tn<hi); v=Fn[m,j]
    print("  %5d-%-6d mean=%.6f  sd=%.6f  min=%.4f max=%.4f"%(lo,hi,v.mean(),v.std(),v.min(),v.max()))
print("  => 소수점 3자리에서 '꿈쩍 않는다'는 것은 4~6자리를 버린 결과. window 수준 sd=%.5f, 범위폭=%.4f"%(Fn[:,j].std(),np.ptp(Fn[:,j])))

print("\n### A9-4  candidate_check.py 재현성 — BL-0 행은 계산되는가")
src=open(r"C:\Users\cmsch\Desktop\대회\2026년 제6회 K-인공지능 제조데이터 분석 경진대회\docs\plan\plan A\Candidate Models\candidate_check.py",encoding="utf-8").read()
print("  bench() 안의 BL-0 행:", [l.strip() for l in src.splitlines() if "BL-0" in l])
print("  => §3 표의 BL-0 행(P/R/F1/FPR)은 문자열 리터럴. 이 스크립트에서 계산되지 않는다.")
print("     또한 BL-0 은 검증구간을 전혀 쓰지 않는다(학습 min/max 규칙) -> 표 머리말 '임계값 = 검증 q99.9' 와 불일치.")

print("\n### A9-5  버스트 길이 / 1건 이벤트 확인")
rn=[b-a for a,b in common.runs(n)]; ro=[b-a for a,b in common.runs(o)]
print("  normal  burst %d개, 길이 분포 min=%d med=%.0f max=%d"%(len(rn),min(rn),np.median(rn),max(rn)))
print("  outlier burst %d개, 길이 = %s  (총 %d행, 2.77분, 단일 연속 기록)"%(len(ro),ro,sum(ro)))
print("  outlier 시간범위 = %.1f 분 -> 13개 burst 는 모두 같은 2.8분 안에 있다. 독립 이벤트 1건."%((o.TimeStamp.max()-o.TimeStamp.min()).total_seconds()/60))
