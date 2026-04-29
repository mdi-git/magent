# Execution Playbook

이 문서는 실제 운영/실험 시 harness를 어떻게 실행하는지 정의한다.

## 1. 사전 점검
- `venv` 활성화
- 데이터 파일 존재 확인
- 모델 출력 디렉토리 쓰기 권한 확인

## 2. 단일 에이전트 디버그 실행

```bash
python main.py solar_forecast
python main.py wind_forecast
python main.py consumption_forecast
```

또는 번호 기반:

```bash
python main.py 1
python main.py 2
python main.py 5
```

## 3. 전체 오케스트레이션 실행

```bash
python main.py all
```

## 4. 실행 중 확인 포인트
- 각 시도의 학습 return code
- 결과 JSON 생성 여부
- best 모델 선택 여부
- 배포 파일 경로 유효성
- 최종 예측 산출물 생성 여부

## 5. 리포트 검증
루트 디렉토리에서 최신 리포트를 확인한다.

```bash
ls -t *_training_report_*.log | head -n 5
```

리포트 필수 섹션:
- `[rounds]`
- `[best]`
- `[deployed]`

## 6. 장애 대응 가이드
- 전 시도 실패: 데이터 경로/의존성/timeout 확인
- score 비정상: metric 방향(min/max) 계약 확인
- 배포 후 추론 실패: model_path/feature_path 정합성 확인

## 7. 논문용 실험 재현 절차
1. 동일 데이터 스냅샷 고정
2. 동일 후보군/임계치 사용
3. 리포트 파일 원본 보관
4. 결과 표 생성 시 score + params 동시 인용
