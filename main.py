import cv2
import torch
from ultralytics import YOLO

model = YOLO("bestv8m.pt") # YOLOv8 커스텀 모델 로드

# 사용할 모드 선택 ('webcam', 'image', 'video')
mode = "webcam" # 'image' 또는 'video'로 변경 가능

# 웹캠 사용
if mode == "webcam":
    cap = cv2.VideoCapture(0) # 기본 웹캠(0번) 사용
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # YOLO 감지 수행

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표 변환
                conf = box.conf[0].item()  # 신뢰도
                label = f"Pothole {conf:.2f}"  # 라벨

                if conf > 0.3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Pothole Detection - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # 'q'를 누르면 종료

    cap.release()
    cv2.destroyAllWindows()

# 이미지 감지
elif mode == "image":
    img_path = "pothole.jpg"  # 감지할 이미지 파일
    img = cv2.imread(img_path)

    results = model(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = f"Pothole {conf:.2f}"

            if conf > 0.3:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Pothole Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 동영상 감지
elif mode == "video":
    video_path = "video.mp4"  # 감지할 동영상 파일
    cap = cv2.VideoCapture(video_path)

    # 동영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, int(
        cap.get(5)), (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = f"Pothole {conf:.2f}"

                if conf > 0.3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Pothole Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

else:
    print("Invalid mode selected. Choose 'webcam', 'image', or 'video'.")
