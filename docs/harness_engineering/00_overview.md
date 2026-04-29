# Harness Engineering Overview for `magent`

## 목적
이 문서는 `magent` 프로젝트의 핵심 목표인 아래 파이프라인을 **코드 변경 최소화 + 운영 재현성 극대화** 방식으로 달성하기 위한 Harness Engineering 설계를 정의한다.

- 예측 에이전트 실행 시 학습 루틴 자동 호출
- 하이퍼파라미터 튜닝 기반 반복 학습
- 성능 기준(best score)으로 모델 선택
- 선택 모델을 추론 경로로 배포 후 예측 실행
- 과정/결과를 리포트로 기록

## Harness Engineering이 필요한 이유
기능 구현만으로는 다음 문제가 남는다.

1. 실험이 실행자마다 다르게 수행됨
2. 종료 조건과 탐색 정책이 코드에 하드코딩되어 변경 비용이 큼
3. 리포트 형식이 일관되지 않아 사후 분석이 어려움
4. 운영/논문/재현 실험을 동일 절차로 돌리기 어려움

Harness Engineering은 위 문제를 다음으로 해결한다.

- 실행 규칙을 문서(계약)로 명시
- 탐색 정책을 문서 기반 파라미터로 분리
- 평가 및 중단 기준을 표준화
- 결과 리포트를 구조화

## 권장 디렉토리

```text
docs/harness_engineering/
  00_overview.md
  01_objectives_and_constraints.md
  02_train_eval_contract.md
  03_search_policy.md
  04_execution_playbook.md
  05_report_template.md
  06_agent_profiles_example.md
```

## 적용 범위
- `solar_forecast_agent`
- `wind_forecast_agent`
- `consumption_forecast_agent`

이상탐지 에이전트(`solar_anomaly`, `wind_anomaly`)는 본 harness의 1차 범위 밖으로 두되, 동일 템플릿 확장 가능하도록 설계한다.
