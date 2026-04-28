# 기능 명세서 (업데이트)

## 1. GPS 수신 및 소스 추상화
- `GPSProvider` 추상 인터페이스로 GPS 소스를 추상화
- 구현체: `FlaskGPSProvider`(스마트폰), `LTEGPSProvider`(LTE 모듈 내장 GPS), `HardwareGPSProvider`(별도 모듈)
- GPS 데이터 전역 변수 접근 시 Lock으로 스레드 안전성 보장
- GPS 신호 소실 시 극소 해상도 프레임 차이로 정지 여부 보완 판단

## 2. 속도 기반 감지 제어 (신규)
- GPS speed 기반으로 추론 동작을 동적으로 조정
  - speed = 0 → 추론 완전 중단
  - speed < 10km/h → 쿨다운 길게 (저속이라 같은 포트홀 오래 찍힘)
  - 10 ~ 80km/h → 정상 추론
  - speed > 80km/h → 프레임 스킵 늘리기 (블러 심함)
  - speed 급격한 변화 → 추론 일시 중단 (급가속/급감속)

## 3. 포트홀 감지 및 모델 추상화
- `PotholeDetector` 추상 인터페이스로 감지 모델을 추상화 (신규)
  - 구현체: `YOLOv8Detector`(현재), `ONNXDetector`(엣지 최적화), `TensorRTDetector`(라즈베리파이 등), `MockDetector`(테스트용)
  - 반환 타입을 통일된 `Detection` 데이터 클래스로 정의 (x1, y1, x2, y2, confidence)
  - 감지 루프는 `PotholeDetector`만 바라보고 구현체에 무관하게 동작
- 모델 워밍업: 시작 시 더미 이미지로 첫 프레임 지연 방지
- 프레임 스킵: N프레임마다 추론하여 리소스 절감 (신규)
- 쿨다운: 포트홀 감지 후 N초간 추론 중단, 같은 포트홀 중복 처리 방지 (신규)
- confidence, iou threshold를 모델 호출 시 직접 전달
- 웹캠 버퍼 크기를 소폭 증가(예: 2~3)하여 Producer-Consumer 구조와 함께 사용 (신규)
- headless 모드 지원 (모니터 없는 엣지 환경) (신규)
- Producer-Consumer 구조 도입 (신규)
  - Producer 스레드: 웹캠에서 프레임을 지속적으로 캡처해 큐에 적재
  - Consumer 스레드: 큐에서 프레임을 꺼내 추론 수행
  - 프레임 스킵은 유지하되 인위적인 카운터 대신 추론 속도를 자연스러운 rate limiter로 활용
  - 캡처와 추론이 독립적으로 동작해 추론 중에도 최신 프레임 유지

## 4. 중복 감지 방지
- 감지 위치의 지오해시를 백엔드 API로 조회
- 이미 등록된 지오해시면 S3 업로드 및 DB 등록 스킵
- 지오해시 precision 수준 설정값으로 관리 (신규)

## 5. 최적 이미지 선택
- 같은 지오해시 폴더 내 여러 프레임 중 신뢰도 가장 높은 이미지 1장 선택
- 파일명 문자열 정렬 대신 신뢰도 값 파싱으로 정확하게 선택 (버그 수정)
- save_detection_info 반환값 None 체크 후 리스트 추가 (버그 수정)

## 6. S3 업로드
- 백엔드에서 Presigned URL 발급받아 이미지 PUT 업로드
- 업로드 실패 시 DB 등록 차단 (버그 수정)

## 7. 포트홀 등록
- S3 업로드 완료 후 백엔드에 위경도 + 이미지 URL + 지오해시 등록
- geohash 인자 불일치 수정: 업로드 함수에 전달되는 geohash가 실제 처리되는 폴더와 일치하도록 수정 (버그 수정)

## 8. 비동기 업로드
- 장수 + 타임아웃 복합 조건으로 업로드 트리거 (신규)
  - N장 이상 쌓이거나 마지막 감지 후 T초 경과 시 업로드 실행
  - 포트홀 없는 구간에서도 업로드가 누락되지 않음
- 스레드 풀로 동시 업로드 스레드 수 제한 (신규)

## 9. 디바이스 헬스 모니터링 (신규)
- CPU 사용률, 메모리, 카메라 연결 상태, 모델 추론 속도를 주기적으로 서버에 전송
- 엣지 디바이스 정상 동작 여부를 서버에서 모니터링 가능

## 10. 주행 경로 트래킹 (신규)
- 주행 중 위경도 + timestamp를 시계열로 서버에 전송
- 어느 도로 구간이 점검됐는지 파악 가능

## 11. 설정 파일 분리 (신규)
- 하드코딩된 값들을 config.yaml 또는 .env로 분리
  - CONF_THRESHOLD, IOU_THRESHOLD
  - FRAME_SKIP_INTERVAL
  - COOLDOWN_SECONDS
  - UPLOAD_TRIGGER_COUNT, UPLOAD_TRIGGER_TIMEOUT
  - GEOHASH_PRECISION
  - API_BASE_URL
  - MODEL_PATH
