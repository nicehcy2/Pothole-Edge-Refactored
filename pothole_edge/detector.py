from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """추론 결과 하나를 나타내는 값 객체.

    모든 구현체가 동일한 타입을 반환하도록 통일해,
    감지 루프가 특정 모델에 의존하지 않도록 한다.
    """

    x1: float  # 바운딩 박스 좌상단 x
    y1: float  # 바운딩 박스 좌상단 y
    x2: float  # 바운딩 박스 우하단 x
    y2: float  # 바운딩 박스 우하단 y
    confidence: float  # 모델이 예측한 신뢰도 (0.0 ~ 1.0)


class PotholeDetector(ABC):
    """포트홀 감지 모델의 공통 인터페이스.

    감지 루프(main.py)는 이 인터페이스만 바라보므로,
    구현체(YOLOv8, ONNX 등)를 교체해도 루프 코드를 수정할 필요가 없다.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """프레임 한 장을 받아 감지 결과 목록을 반환한다."""
        ...

    def warmup(self, width: int = 640, height: int = 640) -> None:
        """시작 시 더미 이미지로 모델을 한 번 실행해 초기화 지연을 없앤다.

        딥러닝 모델은 첫 추론에서 GPU 커널 컴파일·메모리 할당이 일어나
        수백ms 지연이 발생한다. 실제 웹캠 프레임이 들어오기 전에 미리 실행해둔다.
        구현체별로 워밍업 방식이 다르면 이 메서드를 오버라이드하면 된다.
        """
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self.detect(dummy)


class YOLOv8Detector(PotholeDetector):
    """ultralytics YOLOv8 기반 포트홀 감지 구현체."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
    ) -> None:
        """
        Args:
            model_path: .pt 가중치 파일 경로
            conf_threshold: 이 값 미만의 감지 결과는 버린다
            iou_threshold: NMS(Non-Maximum Suppression)에서 겹침 판단 기준
        """
        from ultralytics import YOLO

        self._device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 있으면 "cuda", 없으면 "cpu"
        self._half = self._device == "cuda"                            # FP16은 GPU에서만 지원
        print(f"[Device] {self._device.upper()} | FP16: {self._half}")
        self._model = YOLO(model_path)                                 # 모델 로드 (기본 CPU/FP32)
        self._model.to(self._device)                                   # 가중치를 VRAM(GPU)으로 이동
        if self._half:
            self._model.half()                                         # 가중치를 FP32 → FP16으로 변환 (속도 2배, VRAM 절반)
        self._conf = conf_threshold
        self._iou = iou_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """프레임을 YOLOv8 모델에 입력해 포트홀 감지 결과를 반환한다.

        conf/iou threshold는 매 호출 시 모델에 직접 전달한다.
        verbose=False로 ultralytics의 콘솔 출력을 억제한다.
        """
        # self._model(frame, ...) : YOLO.__call__ 을 호출해 프레임을 모델에 입력하고 추론 결과를 받는다
        # half=self._half : 가중치가 FP16이면 입력 frame도 FP16으로 변환해 타입 불일치 오류를 방지한다
        results = self._model(frame, conf=self._conf, iou=self._iou, verbose=False, device=self._device, half=self._half)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=box.conf[0].item())
                )
        return detections
