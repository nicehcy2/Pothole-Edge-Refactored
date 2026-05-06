import os
import re
import time
from typing import Optional

import cv2
import numpy as np
import pygeohash
import requests


def compute_geohash(lat: float, lon: float, precision: int) -> str:
    return pygeohash.encode(lat, lon, precision=precision)


def is_geohash_registered(geohash_str: str, api_base_url: str) -> bool:
    """백엔드 API로 지오해시 등록 여부를 조회한다. 오류 시 미등록으로 간주한다."""
    try:
        resp = requests.get(
            f"{api_base_url}/api/potholes/geohash/{geohash_str}",
            timeout=3,
        )
        return resp.status_code == 200
    except Exception:
        return False


def save_detection_info(
    frame: np.ndarray,
    confidence: float,
    geohash_str: str,
    output_dir: str,
) -> Optional[str]:
    """감지 프레임을 지오해시 폴더에 저장하고 경로를 반환한다. 실패 시 None."""
    folder = os.path.join(output_dir, geohash_str)
    os.makedirs(folder, exist_ok=True)
    filename = f"conf_{confidence:.4f}_{int(time.time() * 1000)}.jpg"
    path = os.path.join(folder, filename)
    return path if cv2.imwrite(path, frame) else None


def select_best_image(geohash_str: str, output_dir: str) -> Optional[str]:
    """지오해시 폴더에서 신뢰도 값 파싱으로 최고 신뢰도 이미지 경로를 반환한다."""
    folder = os.path.join(output_dir, geohash_str)
    if not os.path.isdir(folder):
        return None

    pattern = re.compile(r"conf_(\d+\.\d+)_")
    best_path: Optional[str] = None
    best_conf = -1.0

    for fname in os.listdir(folder):
        m = pattern.search(fname)
        if m:
            conf = float(m.group(1))
            if conf > best_conf:
                best_conf = conf
                best_path = os.path.join(folder, fname)

    return best_path


def upload_to_s3(image_path: str, api_base_url: str) -> Optional[str]:
    """백엔드에서 Presigned URL을 발급받아 S3에 업로드하고 이미지 URL을 반환한다. 실패 시 None."""
    try:
        resp = requests.post(f"{api_base_url}/api/upload/presigned", timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        with open(image_path, "rb") as f:
            put_resp = requests.put(data["presigned_url"], data=f, timeout=30)
        if put_resp.status_code not in (200, 204):
            return None
        return data.get("image_url")
    except Exception:
        return None


def register_pothole(
    lat: float,
    lon: float,
    image_url: str,
    geohash_str: str,
    api_base_url: str,
) -> bool:
    """백엔드에 포트홀 위경도, 이미지 URL, 지오해시를 등록한다."""
    try:
        resp = requests.post(
            f"{api_base_url}/api/potholes",
            json={"latitude": lat, "longitude": lon, "imageUrl": image_url, "geohash": geohash_str},
            timeout=5,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False
