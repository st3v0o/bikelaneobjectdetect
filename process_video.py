from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from inference_sdk import InferenceHTTPClient
from shapely.geometry import Polygon, box
from azure.storage.blob import BlobServiceClient, ContentSettings


# Load Render/host environment variables at startup.
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "1500")) * 1024 * 1024


def load_polygon(polygon_file: Path) -> Polygon:
    with open(polygon_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data["bike_lane_polygon"]
    return Polygon(points)


def ensure_dirs(base_dir: Path) -> None:
    (base_dir / "output" / "detections").mkdir(parents=True, exist_ok=True)
    (base_dir / "output" / "results").mkdir(parents=True, exist_ok=True)
    (base_dir / "videos").mkdir(parents=True, exist_ok=True)


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def upload_file_to_blob(
    blob_service_client: BlobServiceClient,
    container_name: str,
    local_path: Path,
    blob_name: str,
) -> str:
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    with open(local_path, "rb") as data:
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type="image/jpeg"),
        )

    return blob_client.url


def draw_overlay(frame, polygon_points, detections_to_draw):
    annotated = frame.copy()

    pts = polygon_points[:]
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    for det in detections_to_draw:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class"]} {det["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return annotated


def parse_prediction(pred: dict):
    x = pred["x"]
    y = pred["y"]
    w = pred["width"]
    h = pred["height"]

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    return x1, y1, x2, y2


def bbox_intersection_ratio(bike_lane_poly: Polygon, bbox_coords) -> float:
    x1, y1, x2, y2 = bbox_coords
    det_box = box(x1, y1, x2, y2)

    if det_box.area == 0:
        return 0.0

    intersection_area = bike_lane_poly.intersection(det_box).area
    return intersection_area / det_box.area


def is_vehicle_class(class_name: str) -> bool:
    class_name = class_name.lower().strip()
    vehicle_classes = {
        "car",
        "vehicle",
        "truck",
        "bus",
        "van",
        "pickup",
        "suv",
    }
    return class_name in vehicle_classes


def estimate_event_datetime(video_start_dt: datetime | None, frame_index: int, fps: float) -> datetime | None:
    if video_start_dt is None or fps <= 0:
        return None

    seconds = frame_index / fps
    return video_start_dt + timedelta(seconds=seconds)


def load_video_start_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def load_gpx_track(gpx_path: Path):
    if not gpx_path.exists():
        return []

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    tree = ET.parse(gpx_path)
    root = tree.getroot()

    track = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = trkpt.attrib.get("lat")
        lon = trkpt.attrib.get("lon")
        time_elem = trkpt.find("gpx:time", ns)

        if lat is None or lon is None or time_elem is None or not time_elem.text:
            continue

        timestamp = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00"))
        track.append(
            {
                "timestamp": timestamp,
                "latitude": float(lat),
                "longitude": float(lon),
            }
        )

    track.sort(key=lambda x: x["timestamp"])
    return track


def get_gps_for_datetime(event_dt: datetime | None, gps_track):
    if event_dt is None or not gps_track:
        return None, None

    times = [p["timestamp"] for p in gps_track]
    pos = bisect_left(times, event_dt)

    if pos == 0:
        nearest = gps_track[0]
    elif pos == len(times):
        nearest = gps_track[-1]
    else:
        before = gps_track[pos - 1]
        after = gps_track[pos]

        before_diff = abs((event_dt - before["timestamp"]).total_seconds())
        after_diff = abs((after["timestamp"] - event_dt).total_seconds())
        nearest = before if before_diff <= after_diff else after

    return nearest["latitude"], nearest["longitude"]


def validate_required_env() -> None:
    missing = []
    for key in ["ROBOFLOW_API_KEY", "ROBOFLOW_MODEL_ID", "AZURE_STORAGE_CONNECTION_STRING"]:
        if not os.getenv(key):
            missing.append(key)

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def process_video_job(
    working_dir: Path,
    video_path: Path,
    polygon_path: Path,
    gpx_path: Path | None = None,
    video_start_iso: str | None = None,
) -> dict:
    validate_required_env()
    ensure_dirs(working_dir)

    frame_interval = get_env_int("FRAME_SAMPLE_INTERVAL", 15)
    min_confidence = get_env_float("MIN_CONFIDENCE", 0.4)
    min_intersection_ratio = get_env_float("MIN_INTERSECTION_RATIO", 0.1)

    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    roboflow_model_id = os.getenv("ROBOFLOW_MODEL_ID")
    azure_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    detections_container = os.getenv("AZURE_DETECTIONS_CONTAINER", "detections")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=roboflow_api_key,
    )

    bike_lane_poly = load_polygon(polygon_path)
    polygon_points = list(bike_lane_poly.exterior.coords)[:-1]
    blob_service_client = BlobServiceClient.from_connection_string(azure_conn_str)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    video_start_dt = load_video_start_datetime(video_start_iso)
    gps_track = load_gpx_track(gpx_path) if gpx_path else []

    results = []
    frame_index = 0
    event_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        temp_frame_path = working_dir / "output" / f"frame_{frame_index}.jpg"
        cv2.imwrite(str(temp_frame_path), frame)

        try:
            inference = client.infer(str(temp_frame_path), model_id=roboflow_model_id)
            predictions = inference.get("predictions", [])
        finally:
            if temp_frame_path.exists():
                temp_frame_path.unlink()

        matched_detections = []
        for pred in predictions:
            class_name = pred.get("class", "")
            confidence = float(pred.get("confidence", 0.0))

            if confidence < min_confidence:
                continue
            if not is_vehicle_class(class_name):
                continue

            bbox_coords = parse_prediction(pred)
            overlap_ratio = bbox_intersection_ratio(bike_lane_poly, bbox_coords)

            if overlap_ratio >= min_intersection_ratio:
                matched_detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox_coords,
                        "overlap_ratio": overlap_ratio,
                    }
                )

        if matched_detections:
            event_counter += 1
            event_id = str(uuid.uuid4())

            annotated = draw_overlay(frame, polygon_points, matched_detections)
            image_filename = f"event_{event_counter:05d}_frame_{frame_index}.jpg"
            local_detection_path = working_dir / "output" / "detections" / image_filename
            cv2.imwrite(str(local_detection_path), annotated)

            image_url = upload_file_to_blob(
                blob_service_client=blob_service_client,
                container_name=detections_container,
                local_path=local_detection_path,
                blob_name=image_filename,
            )

            event_dt = estimate_event_datetime(video_start_dt, frame_index, fps)
            lat, lon = get_gps_for_datetime(event_dt, gps_track)

            for det in matched_detections:
                x1, y1, x2, y2 = det["bbox"]
                results.append(
                    {
                        "event_id": event_id,
                        "video_file": video_path.name,
                        "frame_index": frame_index,
                        "frame_time_sec": round(frame_index / fps, 3) if fps > 0 else None,
                        "datetime_utc": event_dt.isoformat() if event_dt else None,
                        "latitude": lat,
                        "longitude": lon,
                        "image_url": image_url,
                        "class": det["class"],
                        "confidence": det["confidence"],
                        "overlap_ratio": det["overlap_ratio"],
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                    }
                )

        frame_index += 1

    cap.release()

    df = pd.DataFrame(results)
    csv_path = working_dir / "output" / "results" / "detections.csv"
    df.to_csv(csv_path, index=False)

    return {
        "csv_path": str(csv_path),
        "rows": len(df),
        "events": event_counter,
        "fps": fps,
        "total_frames": total_frames,
    }


@app.get("/")
def index():
    return jsonify(
        {
            "service": "video-processing-api",
            "status": "ok",
            "routes": {
                "health": "/health",
                "process": "POST /process",
            },
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/process")
def process_video_endpoint():
    if "video" not in request.files:
        return jsonify({"error": "Missing required file field: video"}), 400
    if "polygon" not in request.files:
        return jsonify({"error": "Missing required file field: polygon"}), 400

    temp_dir = Path(tempfile.mkdtemp(prefix="video-job-"))

    try:
        ensure_dirs(temp_dir)

        video_file = request.files["video"]
        polygon_file = request.files["polygon"]
        gpx_file = request.files.get("gpx")
        video_start_iso = request.form.get("video_start_iso")
        download_csv = request.form.get("download_csv", "false").lower() == "true"

        video_path = temp_dir / "videos" / "input_video.mp4"
        polygon_path = temp_dir / "polygon.json"
        gpx_path = temp_dir / "track.gpx"

        video_file.save(video_path)
        polygon_file.save(polygon_path)
        if gpx_file and gpx_file.filename:
            gpx_file.save(gpx_path)
        else:
            gpx_path = None

        result = process_video_job(
            working_dir=temp_dir,
            video_path=video_path,
            polygon_path=polygon_path,
            gpx_path=gpx_path,
            video_start_iso=video_start_iso,
        )

        if download_csv:
            return send_file(
                result["csv_path"],
                as_attachment=True,
                download_name="detections.csv",
                mimetype="text/csv",
            )

        csv_url_note = (
            "CSV returned directly only when download_csv=true; otherwise it is stored temporarily on the server during request execution."
        )
        return jsonify({**result, "note": csv_url_note})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
