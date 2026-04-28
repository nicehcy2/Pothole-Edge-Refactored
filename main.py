import argparse
import time

import cv2
import numpy as np

from pothole_edge import Detection, PotholeDetector, YOLOv8Detector

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


def run_webcam(detector: PotholeDetector) -> None:
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


def run_image(detector: PotholeDetector, image_path: str) -> None:
    """이미지 파일 한 장에 대해 감지를 수행하고 결과 창을 표시한다."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    detections = detector.detect(frame)
    annotated = _draw(frame, detections)

    print(f"감지 결과: {len(detections)}개")
    for det in detections:
        print(f"  confidence={det.confidence:.2f}  bbox=({det.x1:.0f},{det.y1:.0f},{det.x2:.0f},{det.y2:.0f})")

    if not HEADLESS:
        cv2.imshow("Pothole Detection - Image", annotated)
        cv2.waitKey(0)  # 아무 키나 누를 때까지 창 유지
        cv2.destroyAllWindows()


def run_video(detector: PotholeDetector, video_path: str, output_path: str = "output.mp4") -> None:
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

            annotated = _draw(frame, detections)
            out.write(annotated)

            # 100프레임마다 진행률 출력
            if frame_count % 100 == 0:
                print(f"  {frame_count}/{total_frames} 프레임 처리 중...")

            if not HEADLESS:
                cv2.imshow("Pothole Detection - Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        if not HEADLESS:
            cv2.destroyAllWindows()

    print(f"완료: {frame_count}프레임 처리, 총 감지 {detection_count}건 → {output_path}")


# TODO: 로그 표준화
# TODO: main 함수 분할
# TODO: CLI 개선 (예: 감지 모드별로 서브커맨드, 옵션 추가)
# TODO: 예외 처리 강화 (파일 입출력, 카메라 접근 등)
# TODO: 멀티스레딩/멀티프로세싱 도입 (웹캠 프레임 캡처와 감지 병렬화)
# TODO: 객체지향 구조 개선 (예: Detector 인터페이스 확장, 감지 결과 객체화)
# TODO: 설정값을 config 파일이나 환경변수로 분리
def main() -> None:
    parser = argparse.ArgumentParser(description="포트홀 감지")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("webcam", help="웹캠 실시간 감지")

    image_parser = subparsers.add_parser("image", help="이미지 파일 감지")
    image_parser.add_argument("path", help="이미지 파일 경로")

    video_parser = subparsers.add_parser("video", help="동영상 파일 감지")
    video_parser.add_argument("path", help="동영상 파일 경로")
    video_parser.add_argument("--output", default="output.mp4", help="출력 파일 경로 (기본: output.mp4)")

    args = parser.parse_args()

    detector = YOLOv8Detector(MODEL_PATH, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)
    print("Warming up model...")
    detector.warmup()

    if args.mode == "webcam":
        print("Starting webcam detection. Press 'q' to quit.")
        run_webcam(detector)
    elif args.mode == "image":
        run_image(detector, args.path)
    elif args.mode == "video":
        run_video(detector, args.path, args.output)


if __name__ == "__main__":
    main()
