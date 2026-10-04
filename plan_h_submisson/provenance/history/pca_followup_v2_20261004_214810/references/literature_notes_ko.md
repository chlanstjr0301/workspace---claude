# 문헌 확인과 해석 범위

확인일:2026-10-04.

- **Cawley, G. C.; Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR11,2079–2107.** [공식 페이지](https://jmlr.org/papers/v11/cawley10a.html), [원문](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf). 공식 초록과 원문 Section6(인쇄2103쪽/PDF25쪽)의 선택 과적합·평가 편향 설명을 확인했다. 선택 절차도 모델 적합의 일부로 보아 평가와 분리해야 한다는 근거다. 공식 페이지에서 DOI를 확인하지 못했으므로 적지 않는다. 이번에는 후보 수 제한, S 선택 후 설정 잠금, V 실패 시 재선택 금지를 적용했다. 이미 노출된 S/V/H를 새 자료라고 바꾸어 부를 근거는 아니다.
- **Bergmeir, C.; Hyndman, R. J.; Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. Computational Statistics & Data Analysis120,70–83.** [DOI](https://doi.org/10.1016/j.csda.2017.11.003), [공개 원문](https://robjhyndman.com/papers/cv-wp.pdf). 공개본은2017-07-23자 preprint다. Abstract, Sections1–3(정상성·ergodicity·추정 일치성·오차의 MDS 가정), Section7 결론을 확인했다. 순수 자기회귀 모형의 오차가 상관되지 않는 등 조건 아래 표준K-fold가 가능하다는 논의다. 모든 시계열에서 무작위CV를 금지했다는 논문이 아니다. 이번 자료에서 그 가정이 충족됐다고 검증하지 않았으며 원래 시간 분할을 유지하고 원행 의존범위를 감사하는 데 해석상 참고했다. DOI의 출판사 연결은 도구에서500 접근 오류였으나 저자 공개 원문은 확인했다.

FP≤1%, ΔFPR≤.001은 기존 프로젝트 정책을 유지한 것이다. 효과/유지 판정 분리, G0/G1/G2, 분위수 .995/.999, 최대6개, 동점1e-12는 이번 프로젝트 설계다. 문헌이 이 조합의 성능 향상을 보장한다고 주장하지 않는다. F2 가중치를 현장 고장/오경보 비용비로 해석하지 않는다.
