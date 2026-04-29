# Objectives and Constraints

## 1. 시스템 목표 (System Objective)
세 forecast 에이전트(태양광/풍력/소비전력)에 대해 다음 과정을 자동화한다.

1. 학습 스크립트 호출
2. 하이퍼파라미터 후보 탐색
3. 성능지표 기반 best 모델 선택
4. best 모델 추론 경로 배포
5. 예측 실행 및 결과/로그 반환

## 2. 최적화 목표 (Optimization Objective)
- Solar: RMSE 최소화
- Wind: R2 최대화
- Consumption: MAE 최소화

## 3. 반복 제약 (Iteration Constraints)
- 최대 시도 횟수: 4회 (`<5`)
- 최소 시도 횟수: 2회
- 조기 종료(early stop): 성능 임계치 도달 시 즉시 종료

## 4. 운영 제약 (Operational Constraints)
- Python 인터프리터: 프로젝트 `venv/bin/python` 우선
- MCP timeout: 학습+추론 전체를 커버하도록 충분히 크게 설정
- 실패 허용: 일부 시도 실패 가능, 단 best 모델 선택 가능해야 함
- 로깅: 루트 디렉토리에 타임스탬프 기반 리포트 생성 (`YYYYMMDD_HHMM`)

## 5. 재현성 요구사항 (Reproducibility)
각 시도마다 아래를 반드시 기록한다.

- 입력 하이퍼파라미터
- 학습/평가 return code
- metric name, score
- 선택 여부
- 배포 모델 경로

## 6. 산출물 (Artifacts)
- Agent run 결과 JSON
- `{agent}_training_report_YYYYMMDD_HHMM.log`
- 모델 파일 및 feature/scaler 파일(에이전트별 상이)
