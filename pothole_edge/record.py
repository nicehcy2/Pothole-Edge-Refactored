from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from pothole_edge.detector import Detection
from pothole_edge.gps import GPSData


@dataclass
class DetectionRecord:
    """포트홀 감지 결과와 그 시점의 GPS 정보를 함께 묶은 레코드.

    GPS가 없는 환경(이미지·비디오 오프라인 분석, GPS 미연결)에서는 gps=None.
    """

    detection: Detection
    gps: Optional[GPSData]
    timestamp: float = field(default_factory=time.time)
