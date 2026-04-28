from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class GPSData:
    latitude: float   # 위도 (십진수 도, decimal degrees)
    longitude: float  # 경도 (십진수 도, decimal degrees)
    speed: float      # 속도 (km/h)
    timestamp: float = field(default_factory=time.time)


class GPSProvider(ABC):
    """GPS 소스 추상 인터페이스.

    GPS 데이터는 백그라운드 스레드가 주기적으로 갱신하고,
    감지 루프는 get()으로 최신 값을 읽는다.
    Lock으로 두 스레드 간 경쟁 조건을 방지한다.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[GPSData] = None

    @abstractmethod
    def _fetch(self) -> Optional[GPSData]:
        """GPS 소스에서 최신 데이터를 읽어 반환한다. 실패 시 None."""
        ...

    def update(self) -> None:
        """GPS 소스에서 데이터를 읽어 내부 상태를 갱신한다."""
        data = self._fetch()
        if data is not None:
            with self._lock:
                self._latest = data

    def get(self) -> Optional[GPSData]:
        """최신 GPS 데이터를 스레드 안전하게 반환한다. 수신된 데이터가 없으면 None."""
        with self._lock:
            return self._latest

    def start(self, interval: float = 1.0) -> None:
        """백그라운드 폴링 스레드를 시작한다.

        daemon=True이므로 메인 프로세스 종료 시 자동으로 멈춘다.
        """
        def _loop() -> None:
            while True:
                self.update()
                time.sleep(interval)

        threading.Thread(target=_loop, daemon=True).start()


class FlaskGPSProvider(GPSProvider):
    """스마트폰에서 실행 중인 Flask GPS 서버를 폴링하는 구현체.

    스마트폰이 아래 형태의 JSON을 제공하는 엔드포인트를 열어야 한다.
    {"latitude": 37.123, "longitude": 127.456, "speed": 30.5}
    """

    def __init__(self, url: str) -> None:
        """
        Args:
            url: 스마트폰 Flask 서버의 GPS 엔드포인트 (예: "http://192.168.0.100:5000/gps")
        """
        super().__init__()
        self._url = url

    def _fetch(self) -> Optional[GPSData]:
        # 선택적 임포트 — requests 미설치 환경에서 다른 Provider는 영향받지 않도록
        import requests
        try:
            resp = requests.get(self._url, timeout=2)
            resp.raise_for_status()
            body = resp.json()
            return GPSData(
                latitude=float(body["latitude"]),
                longitude=float(body["longitude"]),
                speed=float(body["speed"]),
            )
        except Exception:
            return None


class _NMEASerialProvider(GPSProvider):
    """시리얼 포트에서 NMEA 문장을 읽는 공통 베이스.

    LTEGPSProvider, HardwareGPSProvider가 이 클래스를 상속한다.
    포트 오류 발생 시 다음 폴링 주기에 자동으로 재연결을 시도한다.
    """

    def __init__(self, port: str, baudrate: int) -> None:
        super().__init__()
        self._port = port
        self._baudrate = baudrate
        self._ser = None

    def _open(self) -> None:
        import serial
        if self._ser is None or not self._ser.is_open:
            self._ser = serial.Serial(self._port, self._baudrate, timeout=1)

    def _fetch(self) -> Optional[GPSData]:
        try:
            self._open()
            line = self._ser.readline().decode("ascii", errors="ignore").strip()
            return _parse_gprmc(line)
        except Exception:
            self._ser = None  # 다음 호출에서 재연결 시도
            return None


def _parse_gprmc(sentence: str) -> Optional[GPSData]:
    """GPRMC NMEA 문장에서 위경도·속도를 추출한다.

    형식: $GPRMC,HHMMSS,A,LLLL.LL,N,YYYYY.YY,E,SSS.S,...
    두 번째 필드가 'A'(Active)일 때만 유효한 데이터로 처리한다.
    """
    if not sentence.startswith("$GPRMC"):
        return None
    parts = sentence.split(",")
    if len(parts) < 8 or parts[2] != "A":
        return None
    try:
        lat = _nmea_to_decimal(parts[3], parts[4])
        lon = _nmea_to_decimal(parts[5], parts[6])
        speed_kmh = float(parts[7]) * 1.852  # knots → km/h
        return GPSData(latitude=lat, longitude=lon, speed=speed_kmh)
    except (ValueError, IndexError):
        return None


def _nmea_to_decimal(value: str, direction: str) -> float:
    """NMEA 좌표 형식(DDDMM.MMMM)을 십진수 도(decimal degrees)로 변환한다."""
    dot = value.index(".")
    degrees = float(value[: dot - 2])
    minutes = float(value[dot - 2:])
    decimal = degrees + minutes / 60.0
    if direction in ("S", "W"):
        decimal = -decimal
    return decimal


class LTEGPSProvider(_NMEASerialProvider):
    """LTE 모듈 내장 GPS를 시리얼로 읽는 구현체."""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600) -> None:
        super().__init__(port, baudrate)


class HardwareGPSProvider(_NMEASerialProvider):
    """별도 GPS 하드웨어 모듈을 시리얼로 읽는 구현체.

    라즈베리파이의 UART 기본 포트(/dev/ttyAMA0)를 기본값으로 사용한다.
    """

    def __init__(self, port: str = "/dev/ttyAMA0", baudrate: int = 9600) -> None:
        super().__init__(port, baudrate)


class FrameMotionDetector:
    """GPS 신호 소실 시 프레임 차이로 차량 정지 여부를 보완 판단한다.

    32×32 썸네일로 다운샘플링 후 비교해 연산 비용을 최소화한다.
    """

    _SIZE = (32, 32)
    _THRESHOLD = 2.0  # 평균 픽셀 차이가 이 값 미만이면 정지로 판단

    def __init__(self) -> None:
        self._prev: Optional[np.ndarray] = None

    def is_still(self, frame: np.ndarray) -> bool:
        """현재 프레임과 이전 프레임을 비교해 차량이 정지 중이면 True를 반환한다."""
        thumb = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self._SIZE
        ).astype(float)
        if self._prev is None:
            self._prev = thumb
            return False
        diff = float(np.mean(np.abs(thumb - self._prev)))
        self._prev = thumb
        return diff < self._THRESHOLD
