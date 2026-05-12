# Pothole Edge Device

엣지 디바이스(라즈베리파이, 차량용 컴퓨터 등)에서 실시간으로 도로 포트홀을 탐지하고, 감지 결과를 백엔드 서버에 등록하는 시스템이다.

---

## 목차

- [Pothole Edge Device](#pothole-edge-device)
  - [목차](#목차)
  - [시스템 구조](#시스템-구조)
  - [주요 기능](#주요-기능)
  - [요구사항](#요구사항)
  - [설치](#설치)
  - [환경 변수 설정](#환경-변수-설정)
  - [실행](#실행)
    - [웹캠 실시간 탐지](#웹캠-실시간-탐지)
    - [동영상 파일 분석](#동영상-파일-분석)
    - [이미지 파일 단건 분석](#이미지-파일-단건-분석)
    - [GPS 옵션 추가](#gps-옵션-추가)
  - [GPS 설정](#gps-설정)
  - [TensorRT 변환 (선택)](#tensorrt-변환-선택)
  - [프로젝트 구조](#프로젝트-구조)
    - [모듈별 역할](#모듈별-역할)
  - [알려진 한계 및 TODO](#알려진-한계-및-todo)

---

## 시스템 구조

```
[엣지 디바이스]
카메라 (웹캠 / 영상 / 이미지)
        │
        ▼
YOLOv8 포트홀 탐지
        │
        ├── GPS 위치 태깅
        ├── 지오해시 중복 확인
        ├── 감지 프레임 저장
        └── S3 업로드 → 백엔드 API 등록
```

---

## 주요 기능

**포트홀 탐지**
- YOLOv8 기반 실시간 객체 탐지
- GPU 환경에서 FP16 자동 전환으로 속도 최적화
- 모델 시작 시 워밍업으로 첫 프레임 지연 방지
- `PotholeDetector` 추상 인터페이스로 모델 교체 가능 (YOLOv8 → ONNX → TensorRT)

**중복 감지 방지**
- 지오해시(Geohash) 기반 위치 중복 확인
- 같은 구역의 포트홀은 백엔드 API 조회 후 스킵
- 쿨다운 기능으로 동일 포트홀 반복 처리 방지 (웹캠 전용)

**GPS 연동**
- 세 가지 GPS 소스 지원: Flask 서버(스마트폰), LTE 모듈, 하드웨어 GPS
- 백그라운드 폴링 스레드로 감지 루프와 독립적으로 GPS 갱신
- GPS 신호 없을 때 프레임 차이 분석으로 차량 정지 여부 보완 판단

**이미지 선택 및 업로드**
- 같은 지오해시 내 여러 프레임 중 신뢰도 가장 높은 이미지 선택
- 백엔드 Presigned URL 발급 후 S3 업로드
- 업로드 실패 시 DB 등록 차단

**실행 모드**
- 웹캠 실시간 탐지
- 동영상 파일 분석
- 이미지 파일 단건 분석
- Headless 모드 지원 (모니터 없는 환경)

---

## 요구사항

- Python 3.10 이상
- CUDA 지원 GPU (선택, CPU 전용도 동작)
- 카메라 또는 입력 영상 파일

---

## 설치

```bash
git clone <repository-url>
cd pothole-edge
pip install -r requirements.txt
```

GPU(CUDA) 지원이 필요한 경우 PyTorch를 별도로 설치한다.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하거나 기존 파일을 수정한다.


| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_PATH` | `bestv8m.pt` | YOLOv8 가중치 파일 경로 |
| `CONF_THRESHOLD` | `0.3` | confidence 임계값. 낮을수록 오탐 증가 |
| `IOU_THRESHOLD` | `0.45` | NMS IoU 임계값 |
| `FRAME_SKIP` | `3` | 웹캠 모드에서 N프레임마다 추론 |
| `COOLDOWN_SEC` | `3.0` | 감지 후 쿨다운 시간(초), 웹캠 전용 |
| `HEADLESS` | `false` | `true`로 설정 시 화면 출력 없이 실행 |
| `GEOHASH_PRECISION` | `7` | 중복 감지 판단 구역 크기 조정 |
| `API_BASE_URL` | `http://localhost:8080` | 백엔드 서버 주소 |
| `OUTPUT_DIR` | `detections` | 감지 프레임 로컬 저장 경로 |

---

## 실행

### 웹캠 실시간 탐지

```bash
python main.py webcam
```

### 동영상 파일 분석

```bash
python main.py video input/video.mp4
python main.py video input/video.mp4 --output output/
```

### 이미지 파일 단건 분석

```bash
python main.py image pothole.jpg
```

### GPS 옵션 추가

모든 실행 모드에서 `--gps` 옵션을 추가할 수 있다.

```bash
# 스마트폰 Flask GPS 서버
python main.py webcam --gps flask --gps-url http://192.168.0.100:5000/gps

# LTE 모듈 GPS
python main.py webcam --gps lte --gps-port /dev/ttyUSB0

# 하드웨어 GPS 모듈
python main.py webcam --gps hardware --gps-port /dev/ttyAMA0
```

---

## GPS 설정

세 가지 GPS 소스를 지원한다.

| 방식 | 옵션 | 기본값 | 설명 |
|------|------|--------|------|
| Flask GPS | `--gps flask` | `http://192.168.0.100:5000/gps` | 스마트폰에서 Flask 서버 실행 후 HTTP 폴링 |
| LTE 모듈 | `--gps lte` | `/dev/ttyUSB0` | LTE 모듈 내장 GPS를 시리얼 NMEA 파싱 |
| 하드웨어 GPS | `--gps hardware` | `/dev/ttyAMA0` | 별도 GPS 모듈을 시리얼 NMEA 파싱 |

`--gps` 미지정 시 GPS 없이 실행된다. 이 경우 위치 기반 중복 확인 및 백엔드 등록이 동작하지 않는다.

**Flask GPS 서버 응답 형식**

스마트폰에서 아래 형태의 JSON을 제공하는 엔드포인트를 열어야 한다.

```json
{"latitude": 37.123, "longitude": 127.456, "speed": 30.5}
```

---

## TensorRT 변환 (선택)

GPU 환경에서 추론 속도를 높이려면 `.pt` 모델을 TensorRT `.engine` 형식으로 변환한다.

```bash
python export_engine.py
```

변환 완료 후 `.env`의 `MODEL_PATH`를 `bestv8m.engine`으로 변경한다.

> TensorRT 변환은 CUDA GPU 환경에서만 가능하다.

---

## 프로젝트 구조

```
pothole-edge/
├── main.py                   # 진입점. 실행 모드 및 감지 루프 관리
├── export_engine.py          # TensorRT 변환 스크립트
├── requirements.txt          # 의존성 패키지
├── .env                      # 환경 변수 설정
│
├── pothole_edge/
│   ├── __init__.py           # 패키지 공개 인터페이스
│   ├── detector.py           # PotholeDetector 인터페이스 및 YOLOv8Detector 구현
│   ├── gps.py                # GPSProvider 인터페이스 및 구현체 (Flask, LTE, Hardware)
│   ├── record.py             # DetectionRecord 데이터 클래스
│   └── uploader.py           # 지오해시 중복 확인, S3 업로드, 백엔드 등록
│
├── input/                    # 테스트용 입력 파일
├── output/                   # 감지 결과 저장 디렉토리
└── docs/                     # 기능 명세 문서
```

### 모듈별 역할

| 모듈 | 역할 |
|------|------|
| `detector.py` | `PotholeDetector` 추상 인터페이스 정의. `YOLOv8Detector`가 구현체. 모델 교체 시 이 파일만 수정 |
| `gps.py` | `GPSProvider` 추상 인터페이스와 Flask·LTE·Hardware 구현체. 백그라운드 스레드로 GPS 주기적 갱신 |
| `record.py` | 감지 결과(`Detection`)와 GPS 정보(`GPSData`)를 묶은 `DetectionRecord` 정의 |
| `uploader.py` | 지오해시 계산, 백엔드 중복 확인, 로컬 저장, S3 업로드, 포트홀 등록 |

---

## 알려진 한계 및 TODO

**정확도**
- 나무, 하늘 등 도로와 무관한 영역에서 오탐 발생 가능
- 맨홀 뚜껑, 그림자, 물웅덩이 등 포트홀과 외관이 유사한 객체에 대한 오탐 존재
- 야간, 역광, 강수 환경에서 정확도 저하 가능

**구현 예정**
- [ ] 로그 표준화 (현재 print 기반 임시 출력)
- [ ] 예외 처리 강화 (파일 입출력, 카메라 접근 등)
- [ ] 신뢰도 외에 이미지 품질까지 고려한 최적 이미지 선택 로직
- [ ] 최적 이미지 수 조정 (현재 1장 고정)
- [ ] GPS 속도 기반 추론 동적 제어 (정지 시 중단, 고속 시 프레임 스킵 증가)
- [ ] Producer-Consumer 구조 도입으로 캡처와 추론 분리
- [ ] 비동기 업로드 (장수 + 타임아웃 복합 트리거)
- [ ] 디바이스 헬스 모니터링 (CPU, 메모리, 카메라 상태 서버 전송)
- [ ] 주행 경로 트래킹 (위경도 시계열 서버 전송)
- [ ] 매 프레임마다 포트홀 DB 조회하는 방식 개선