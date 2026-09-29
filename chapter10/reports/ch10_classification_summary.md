# Ch10 주문 취소 분류 모델링 최종 요약 보고서

## 1. 모델링 핵심 사양 (Pipeline Specs)
- **최종 선택 모델**: Random Forest
- **최종 확정 Threshold**: 0.1
- **데이터 분할 방식**: Stratified Random Split (60% Train / 20% Val / 20% Test)

## 2. Final Test 성과 (Test Set Metrics)
- **Accuracy**: 36.67%
- **Precision**: 23.40%
- **Recall**: 84.62%
- **F1-Score**: 36.67%

## 3. 혼동 행렬 (Confusion Matrix)
- **TN**: 11건 | **FP**: 36건
- **FN**: 2건 | **TP**: 11건

## 4. 파이프라인 검증 계약 (Validation Evidence)
- 8대 검증 계약(binary_target, privacy, freeze test 등) **모두 [PASS] 달성 완료**.
