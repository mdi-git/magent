# magent

`magent`는 마이크로그리드 운영을 위한 멀티 에이전트 예측/이상탐지 프로젝트입니다.
핵심 흐름은 **(예측 에이전트 기준) 학습 → 최적 모델 선택 → 예측 실행 → 리포트 기록**입니다.

## 개요

- 오케스트레이터가 여러 MCP 에이전트를 순차 실행합니다.
- 예측 에이전트(`solar`, `wind`, `consumption`)는 실행 시 학습 루틴을 함께 수행합니다.
- 하이퍼파라미터 후보를 시도하고, 점수 기준으로 최적 모델을 선택합니다.
- 각 학습 시도와 최종 선택 결과를 timestamp 로그로 남깁니다.

## 프로젝트 구조 (핵심)

```text
magent/
  main.py
  magent_agents/
    microgrid_balance_orchestrator.py
    solar_forecast_mcp.py
    wind_forecast_mcp.py
    consumption_forecast_mcp.py
    solar_anomaly_mcp.py
    wind_anomaly_mcp.py
  solar_train/
  wind_predict_train/
  powermeter_train/
  gpu/
  docs/harness_engineering/
```

## 빠른 시작

### 1) 환경 준비

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) 단일 에이전트 실행

```bash
python main.py solar_forecast
python main.py wind_forecast
python main.py consumption_forecast
```

번호로도 실행할 수 있습니다.

```bash
python main.py 1
python main.py 2
python main.py 5
```

### 3) 전체 오케스트레이션 실행

```bash
python main.py all
```

## 로그/결과 확인

- 루트 경로에 학습 리포트가 생성됩니다. (`*_training_report_YYYYMMDD_HHMMSS.log`)
- 최근 리포트 확인:

```bash
ls -t *_training_report_*.log | head -n 5
```

## Harness 문서

학습/평가 계약과 실행 정책은 아래 문서를 참고하세요.

- `docs/harness_engineering/00_overview.md`
- `docs/harness_engineering/02_train_eval_contract.md`
- `docs/harness_engineering/03_search_policy.md`
- `docs/harness_engineering/04_execution_playbook.md`

## 라이선스

본 프로젝트는 MIT License를 따릅니다. 자세한 내용은 `LICENSE`를 참고하세요.
