import argparse
import time
from typing import Optional

import cv2
import numpy as np

from pothole_edge import (
    Detection,
    DetectionRecord,
    FlaskGPSProvider,
    GPSProvider,
    HardwareGPSProvider,
    LTEGPSProvider,
    PotholeDetector,
    YOLOv8Detector,
)

# ── 설정값 ──────────────────────────────────────────────────────────────────
MODEL_PATH = "bestv8m.pt"
CONF_THRESHOLD = 0.3   # 이 신뢰도 미만의 감지 결과는 버린다
IOU_THRESHOLD = 0.45   # NMS 겹침 판단 기준
FRAME_SKIP = 3         # N프레임마다 한 번 추론 (CPU/GPU 부하 절감, 웹캠 전용)
COOLDOWN_SEC = 3.0     # 감지 후 N초간 추론 중단 (같은 포트홀 중복 처리 방지, 웹캠 전용)
HEADLESS = False       # 모니터 없는 엣지 환경에서 True로 설정
# ────────────────────────────────────────────────────────────────────────────


def _draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """감지 결과를 프레임에 바운딩 박스와 신뢰도로 그려 반환한다."""
    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Pothole {det.confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame


def _log_record(record: DetectionRecord) -> None:
    """감지 레코드를 콘솔에 출력한다. 로그 표준화 전 임시 출력."""
    det = record.detection
    if record.gps:
        gps_str = (
            f"lat={record.gps.latitude:.6f}  lon={record.gps.longitude:.6f}"
            f"  speed={record.gps.speed:.1f}km/h"
        )
    else:
        gps_str = "GPS=없음"
    print(f"[감지] conf={det.confidence:.2f}  bbox=({det.x1:.0f},{det.y1:.0f},{det.x2:.0f},{det.y2:.0f})  {gps_str}")


def _make_records(detections: list[Detection], gps_provider: Optional[GPSProvider]) -> list[DetectionRecord]:
    """감지 결과 목록을 현재 GPS 스냅샷과 묶어 레코드 목록으로 변환한다.

    GPS는 여러 감지가 동시에 발생해도 동일한 시점 스냅샷을 공유한다.
    """
    gps = gps_provider.get() if gps_provider else None
    return [DetectionRecord(detection=det, gps=gps) for det in detections]


def run_webcam(detector: PotholeDetector, gps_provider: Optional[GPSProvider] = None) -> None:
    """웹캠 스트림을 받아 포트홀 감지 루프를 실행한다."""
    cap = cv2.VideoCapture(0)
    # 버퍼를 1로 고정해 항상 최신 프레임을 처리한다.
    # 기본값(4~10)이면 추론 속도보다 캡처가 빨라 오래된 프레임이 쌓인다.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    cooldown_until = 0.0                   # 이 시각 이전에는 추론을 건너뛴다
    last_detections: list[Detection] = []  # 쿨다운 중에도 마지막 결과를 화면에 유지

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.monotonic()
            frame_count += 1

            # 쿨다운이 끝났고 FRAME_SKIP 주기에 해당하는 프레임일 때만 추론
            if now >= cooldown_until and frame_count % FRAME_SKIP == 0:
                last_detections = detector.detect(frame)
                if last_detections:
                    # 포트홀이 감지되면 쿨다운 시작 — 동일 포트홀 반복 처리 방지
                    cooldown_until = now + COOLDOWN_SEC
                    for record in _make_records(last_detections, gps_provider):
                        _log_record(record)

            if not HEADLESS:
                annotated = _draw(frame.copy(), last_detections)
                cv2.imshow("Pothole Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # 예외 발생 시에도 카메라와 윈도우를 반드시 해제한다
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()


def run_image(
    detector: PotholeDetector,
    image_path: str,
    gps_provider: Optional[GPSProvider] = None,
) -> None:
    """이미지 파일 한 장에 대해 감지를 수행하고 결과를 출력한다."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    detections = detector.detect(frame)
    records = _make_records(detections, gps_provider)

    print(f"감지 결과: {len(records)}개")
    for record in records:
        _log_record(record)

    if not HEADLESS:
        cv2.imshow("Pothole Detection - Image", _draw(frame, detections))
        cv2.waitKey(0)  # 아무 키나 누를 때까지 창 유지
        cv2.destroyAllWindows()


def run_video(
    detector: PotholeDetector,
    video_path: str,
    output_path: str = "output.mp4",
    gps_provider: Optional[GPSProvider] = None,
) -> None:
    """동영상 파일을 프레임 단위로 감지해 결과 동영상을 저장한다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"동영상을 불러올 수 없습니다: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    detection_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            detections = detector.detect(frame)

            if detections:
                detection_count += len(detections)
                for record in _make_records(detections, gps_provider):
                    _log_record(record)

            out.write(_draw(frame, detections))

            # 100프레임마다 진행률 출력
            if frame_count % 100 == 0:
                print(f"  {frame_count}/{total_frames} 프레임 처리 중...")

            if not HEADLESS:
                cv2.imshow("Pothole Detection - Video", _draw(frame.copy(), detections))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        if not HEADLESS:
            cv2.destroyAllWindows()

    print(f"완료: {frame_count}프레임 처리, 총 감지 {detection_count}건 → {output_path}")


def _build_gps_provider(args: argparse.Namespace) -> Optional[GPSProvider]:
    """CLI 인자로부터 GPS 프로바이더를 생성한다. --gps 미지정 시 None 반환."""
    if args.gps == "flask":
        return FlaskGPSProvider(args.gps_url)
    elif args.gps == "lte":
        return LTEGPSProvider(args.gps_port or "/dev/ttyUSB0")
    elif args.gps == "hardware":
        return HardwareGPSProvider(args.gps_port or "/dev/ttyAMA0")
    return None


# TODO: 로그 표준화
# TODO: 예외 처리 강화 (파일 입출력, 카메라 접근 등)
# TODO: Producer-Consumer 구조 도입 (웹캠 캡처·추론 스레드 분리)
# TODO: 설정값을 config 파일이나 환경변수로 분리
def main() -> None:
    parser = argparse.ArgumentParser(description="포트홀 감지")

    # GPS 옵션은 모든 모드에서 공통으로 사용
    parser.add_argument("--gps", choices=["flask", "lte", "hardware"], default=None,
                        help="GPS 소스 유형")
    parser.add_argument("--gps-url", default="http://192.168.0.100:5000/gps",
                        help="Flask GPS 서버 URL (--gps flask 시 사용)")
    parser.add_argument("--gps-port", default=None,
                        help="시리얼 포트 (--gps lte/hardware 시 사용)")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("webcam", help="웹캠 실시간 감지")

    image_parser = subparsers.add_parser("image", help="이미지 파일 감지")
    image_parser.add_argument("path", help="이미지 파일 경로")

    video_parser = subparsers.add_parser("video", help="동영상 파일 감지")
    video_parser.add_argument("path", help="동영상 파일 경로")
    video_parser.add_argument("--output", default="output.mp4", help="출력 파일 경로 (기본: output.mp4)")

    args = parser.parse_args()

    gps_provider = _build_gps_provider(args)
    if gps_provider:
        gps_provider.start(interval=1.0)
        print(f"GPS 시작: {type(gps_provider).__name__}")

    detector = YOLOv8Detector(MODEL_PATH, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)
    print("Warming up model...")
    detector.warmup()

    if args.mode == "webcam":
        print("Starting webcam detection. Press 'q' to quit.")
        run_webcam(detector, gps_provider)
    elif args.mode == "image":
        run_image(detector, args.path, gps_provider)
    elif args.mode == "video":
        run_video(detector, args.path, args.output, gps_provider)


if __name__ == "__main__":
    main()

"""
사용 예시:

  python main.py webcam
  python main.py webcam --gps flask --gps-url http://192.168.0.100:5000/gps
  python main.py webcam --gps lte --gps-port /dev/ttyUSB0
  python main.py webcam --gps hardware

  python main.py image pothole.jpg
  python main.py image pothole.jpg --gps flask

  python main.py video video.mp4
  python main.py video video.mp4 --output result.mp4 --gps flask
"""
