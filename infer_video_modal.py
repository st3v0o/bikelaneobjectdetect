import json
import os
import subprocess
import xml.etree.ElementTree as ET
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve

import modal

app = modal.App("bike-lane-video-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.4.21",
        "opencv-python-headless",
        "numpy",
        "supabase",
        "httpx",
        "shapely",
    )
)

volume = modal.Volume.from_name("bike-lane-data", create_if_missing=True)
supabase_secret = modal.Secret.from_name("supabase-creds")


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 30,
    volumes={"/root/models": volume},
    secrets=[supabase_secret],
)
def run_video(
    video_url: str,
    gpx_url: str,
    job_name: str,
    video_start_iso: str | None = None,
):
    import cv2
    import numpy as np
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from supabase import create_client
    from ultralytics import YOLO

    # ----------------------------
    # Config
    # ----------------------------
    BIKE_LANE_CLASS_NAMES = {"bike-lane"}

    # Classes you want treated as hazards.
    # Leave empty set() to allow any non-bike-lane class.
    HAZARD_CLASS_ALLOWLIST = {
    "vehicle",
    "leaves",
    "sediment",
    "advance warning sign",
    }

    FRAME_FPS_EXTRACT = 1
    MIN_CONFIDENCE = 0.25
    MIN_INTERSECTION_RATIO = 0.05  # fraction of hazard polygon area intersecting bike lane
    UPSERT_TO_DB = True
    DETECTIONS_TABLE = "detections"
    OVERLAYS_BUCKET = "overlays"
    RESULTS_BUCKET = "results"

    # ----------------------------
    # Supabase
    # ----------------------------
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase = create_client(supabase_url, supabase_key)

    # ----------------------------
    # Working dirs
    # ----------------------------
    work = Path("/tmp/video_job")
    video_dir = work / "video"
    gpx_dir = work / "gpx"
    frames_dir = work / "frames"
    out_dir = work / "out"

    video_dir.mkdir(parents=True, exist_ok=True)
    gpx_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / "input.mp4"
    gpx_path = gpx_dir / "track.gpx"

    # ----------------------------
    # Helpers
    # ----------------------------
    def parse_iso(dt_str: str) -> datetime:
        # supports Z
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

    def load_gpx_track(path: Path):
        ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
        tree = ET.parse(path)
        root = tree.getroot()

        track = []
        for trkpt in root.findall(".//gpx:trkpt", ns):
            lat = trkpt.attrib.get("lat")
            lon = trkpt.attrib.get("lon")
            time_elem = trkpt.find("gpx:time", ns)

            if lat is None or lon is None or time_elem is None or not time_elem.text:
                continue

            track.append(
                {
                    "timestamp": parse_iso(time_elem.text),
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            )

        track.sort(key=lambda p: p["timestamp"])
        return track

    def interpolate_gps_for_datetime(event_dt: datetime | None, gps_track):
        if event_dt is None or not gps_track:
            return None, None, None

        times = [p["timestamp"] for p in gps_track]
        pos = bisect_left(times, event_dt)

        if pos == 0:
            p = gps_track[0]
            return p["latitude"], p["longitude"], p["timestamp"].isoformat()

        if pos >= len(gps_track):
            p = gps_track[-1]
            return p["latitude"], p["longitude"], p["timestamp"].isoformat()

        before = gps_track[pos - 1]
        after = gps_track[pos]

        total = (after["timestamp"] - before["timestamp"]).total_seconds()
        if total <= 0:
            return before["latitude"], before["longitude"], before["timestamp"].isoformat()

        elapsed = (event_dt - before["timestamp"]).total_seconds()
        ratio = max(0.0, min(1.0, elapsed / total))

        lat = before["latitude"] + (after["latitude"] - before["latitude"]) * ratio
        lon = before["longitude"] + (after["longitude"] - before["longitude"]) * ratio
        return lat, lon, event_dt.isoformat()

    def polygon_from_mask(mask_xy):
        if mask_xy is None or len(mask_xy) < 3:
            return None
        try:
            poly = Polygon(mask_xy)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                return None
            return poly
        except Exception:
            return None

    def overlap_ratio(hazard_poly, bike_lane_poly):
        if hazard_poly is None or bike_lane_poly is None or hazard_poly.area == 0:
            return 0.0
        inter_area = hazard_poly.intersection(bike_lane_poly).area
        return inter_area / hazard_poly.area

    def should_treat_as_bike_lane(class_name: str) -> bool:
        return class_name.lower().strip() in BIKE_LANE_CLASS_NAMES

    def should_treat_as_hazard(class_name: str) -> bool:
        class_name = class_name.lower().strip()
        if should_treat_as_bike_lane(class_name):
            return False
        if HAZARD_CLASS_ALLOWLIST:
            return class_name in HAZARD_CLASS_ALLOWLIST
        return True

    def upload_file_and_get_url(bucket: str, object_path: str, local_path: Path) -> str:
        with open(local_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                object_path,
                f,
                {"content-type": "image/jpeg", "x-upsert": "true"},
            )

        # Public bucket version:
        return supabase.storage.from_(bucket).get_public_url(object_path)

        # If bucket is private, replace with something like:
        # signed = supabase.storage.from_(bucket).create_signed_url(object_path, 60 * 60 * 24 * 7)
        # return signed["signedURL"]

    # ----------------------------
    # Download inputs
    # ----------------------------
    print(f"Downloading video from: {video_url}")
    urlretrieve(video_url, video_path)

    print(f"Downloading GPX from: {gpx_url}")
    urlretrieve(gpx_url, gpx_path)

    gps_track = load_gpx_track(gpx_path)
    if not gps_track:
        raise RuntimeError("No timestamped GPX track points found in GPX file")

    # If video_start_iso is not supplied, assume the video starts at the first GPX timestamp.
    # This is okay if recording started together. If not, pass video_start_iso explicitly.
    video_start_dt = parse_iso(video_start_iso) if video_start_iso else gps_track[0]["timestamp"]

    # ----------------------------
    # Extract 1 fps frames
    # ----------------------------
    print("Extracting 1 fps frames...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"fps={FRAME_FPS_EXTRACT}",
            str(frames_dir / "frame_%05d.jpg"),
        ],
        check=True,
    )

    frame_files = sorted(frames_dir.glob("*.jpg"))
    print(f"Extracted {len(frame_files)} frames")
    if not frame_files:
        raise RuntimeError("No frames extracted from video")

    # ----------------------------
    # Load model
    # ----------------------------
    model = YOLO("/root/models/models/best.pt")

    # ----------------------------
    # Run inference
    # ----------------------------
    print("Running inference...")
    results = model.predict(
        source=str(frames_dir),
        save=True,
        project=str(out_dir),
        name="predictions",
        conf=MIN_CONFIDENCE,
    )

    prediction_dir = out_dir / "predictions"
    summary = []
    db_rows = []
    uploaded_overlay_count = 0

    for i, r in enumerate(results):
        frame_name = frame_files[i].name
        frame_path = frame_files[i]
        overlay_path = prediction_dir / frame_name

        names = r.names
        boxes = r.boxes
        masks = r.masks

        n = len(boxes) if boxes is not None else 0
        if n == 0:
            summary.append(
                {
                    "frame_index": i,
                    "timestamp_seconds": i / FRAME_FPS_EXTRACT,
                    "frame_file": frame_name,
                    "kept": False,
                    "reason": "no_detections",
                    "detections": [],
                }
            )
            continue

        all_detections = []
        bike_lane_polys = []

        for j in range(n):
            cls_id = int(boxes.cls[j].item())
            conf = float(boxes.conf[j].item())
            class_name = names.get(cls_id, str(cls_id))

            poly = None
            polygon_coords = None
            if masks is not None and hasattr(masks, "xy") and j < len(masks.xy):
                polygon_coords = masks.xy[j].tolist()
                poly = polygon_from_mask(polygon_coords)

            det = {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "polygon": polygon_coords,
                "shape": poly,
            }
            all_detections.append(det)

            if should_treat_as_bike_lane(class_name) and poly is not None:
                bike_lane_polys.append(poly)

        if not bike_lane_polys:
            summary.append(
                {
                    "frame_index": i,
                    "timestamp_seconds": i / FRAME_FPS_EXTRACT,
                    "frame_file": frame_name,
                    "kept": False,
                    "reason": "no_bike_lane_detected",
                    "detections": [
                        {
                            "class_name": d["class_name"],
                            "confidence": d["confidence"],
                        }
                        for d in all_detections
                    ],
                }
            )
            continue

        bike_lane_union = unary_union(bike_lane_polys)
        if isinstance(bike_lane_union, MultiPolygon):
            bike_lane_union = unary_union(list(bike_lane_union.geoms))

        intersecting_hazards = []
        for det in all_detections:
            if not should_treat_as_hazard(det["class_name"]):
                continue
            if det["shape"] is None:
                continue

            ratio = overlap_ratio(det["shape"], bike_lane_union)
            if ratio >= MIN_INTERSECTION_RATIO:
                intersecting_hazards.append(
                    {
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "bike_lane_overlap": ratio,
                        "polygon": det["polygon"],
                    }
                )

        if not intersecting_hazards:
            summary.append(
                {
                    "frame_index": i,
                    "timestamp_seconds": i / FRAME_FPS_EXTRACT,
                    "frame_file": frame_name,
                    "kept": False,
                    "reason": "no_hazard_intersecting_bike_lane",
                    "detections": [
                        {
                            "class_name": d["class_name"],
                            "confidence": d["confidence"],
                        }
                        for d in all_detections
                    ],
                }
            )
            continue

        timestamp_seconds = i / FRAME_FPS_EXTRACT
        event_dt = video_start_dt + timedelta(seconds=timestamp_seconds)
        latitude, longitude, gpx_time = interpolate_gps_for_datetime(event_dt, gps_track)

        if overlay_path.exists():
            overlay_object_path = f"{job_name}/{frame_name}"
            image_url = upload_file_and_get_url(OVERLAYS_BUCKET, overlay_object_path, overlay_path)
            uploaded_overlay_count += 1
        else:
            image_url = None

        summary.append(
            {
                "frame_index": i,
                "timestamp_seconds": timestamp_seconds,
                "frame_file": frame_name,
                "kept": True,
                "image_url": image_url,
                "latitude": latitude,
                "longitude": longitude,
                "gpx_time": gpx_time,
                "intersecting_hazards": intersecting_hazards,
            }
        )

        for hazard in intersecting_hazards:
            db_rows.append(
                {
                    "job_name": job_name,
                    "frame_index": i,
                    "timestamp_seconds": timestamp_seconds,
                    "latitude": latitude,
                    "longitude": longitude,
                    "image_url": image_url,
                    "confidence": hazard["confidence"],
                    "class_name": hazard["class_name"],
                    "bike_lane_overlap": hazard["bike_lane_overlap"],
                    "gpx_time": gpx_time,
                }
            )

    # ----------------------------
    # Upload summary.json
    # ----------------------------
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_object_path = f"{job_name}/summary.json"
    with open(summary_path, "rb") as f:
        supabase.storage.from_(RESULTS_BUCKET).upload(
            summary_object_path,
            f,
            {"content-type": "application/json", "x-upsert": "true"},
        )

    summary_url = supabase.storage.from_(RESULTS_BUCKET).get_public_url(summary_object_path)

    # ----------------------------
    # Insert rows into database
    # ----------------------------
    inserted_count = 0
    if UPSERT_TO_DB and db_rows:
        # chunk inserts to be safe
        chunk_size = 500
        for start in range(0, len(db_rows), chunk_size):
            chunk = db_rows[start : start + chunk_size]
            supabase.table(DETECTIONS_TABLE).insert(chunk).execute()
            inserted_count += len(chunk)

    return {
        "job_name": job_name,
        "frames_extracted": len(frame_files),
        "frames_kept": len([x for x in summary if x.get("kept")]),
        "overlay_images_uploaded": uploaded_overlay_count,
        "db_rows_inserted": inserted_count,
        "summary_url": summary_url,
    }


@app.local_entrypoint()
def main(
    video_url: str,
    gpx_url: str,
    job_name: str,
    video_start_iso: str = "",
):
    result = run_video.remote(
        video_url=video_url,
        gpx_url=gpx_url,
        job_name=job_name,
        video_start_iso=video_start_iso or None,
    )
    print(result)