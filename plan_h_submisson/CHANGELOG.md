# Package changes relative to original experiment

- Frozen B0/R5 objects, features, thresholds, split, labels and predictions unchanged; runtime sources copied byte-for-byte.
- Added portable orchestration, fixed-setting two-PCA/one-R5 reconstruction, raw-input mapping validation, common paper metric/figure generator and checksum entrypoint. These are new package utilities, not purported historical originals.
- Retained original complete-data helper and later finite-input rejection wrapper separately; no model-level missing-data interpolation or suppression introduced.
- Regenerated manuscript tables from stored predictions. Corrected draft training-target count from an unverified11292 to actual11296 before final publication in package; no experimental file changed.
- Paper build attempt1 failed in koTeX ICU linebreak; used the prior complete manuscript's xeCJK/Noto settings. Attempt2 exposed a bibliography ampersand escaping error; fixed typesetting only. Logs retained.
- English default anonymous ICML2026 style; Korean research preprint layout retained. No accepted option and no invented author/funding/COI facts.
- Adversarial review is a separate author-side AI-assisted weakness review of final manuscripts, not external review.
- New package reconstruction/inference executions occur after the prior usage audit cutoff; their counts are documented separately and are not silently added to historical counts.

Publication packaging corrections: R5 training pairs 11292→11296 after actual fixed reconstruction; predictor matrix orientation; escaped bibliography ampersand; xeCJK/font engine; official English T1 Times encoding; anonymous correspondence placeholder; indicator glyph; unclipped flow diagram and separated split labels. These are manuscript/build corrections, not model or prediction changes. Historical fit/test counts are left unchanged; this packaging round fits two selected PCA pipelines and one R5 per training reproduction, separately recorded.
