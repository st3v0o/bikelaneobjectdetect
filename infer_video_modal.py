import json
import os
import shutil
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

    HAZARD_CLASS_ALLOWLIST = {
        "vehicle",
        "leaves",
        "sediment",
        "advance warning sign",
    }

    FRAME_FPS_EXTRACT = 1
    MIN_CONFIDENCE = 0.25

    # Shrink the bike lane inward a little so tiny edge touches don't count.
    BIKE_LANE_SHRINK_PIXELS = 6

    # Generic thresholds for non-vehicle hazards.
    DEFAULT_MIN_INTERSECTION_AREA = 1500.0
    DEFAULT_MIN_HAZARD_OVERLAP_RATIO = 0.12  # fraction of hazard area inside lane
    DEFAULT_MIN_BIKE_LANE_OVERLAP_RATIO = 0.01  # fraction of lane area covered
    DEFAULT_REQUIRE_CENTROID_IN_LANE = False

    # Stricter rule for vehicles so a tire edge doesn't count.
    CLASS_RULES = {
        "vehicle": {
            "min_intersection_area": 2500.0,
            "min_hazard_overlap_ratio": 0.20,
            "min_bike_lane_overlap_ratio": 0.015,
            "require_centroid_in_lane": True,
        },
        "leaves": {
            "min_intersection_area": 1200.0,
            "min_hazard_overlap_ratio": 0.10,
            "min_bike_lane_overlap_ratio": 0.005,
            "require_centroid_in_lane": False,
        },
        "sediment": {
            "min_intersection_area": 1200.0,
            "min_hazard_overlap_ratio": 0.10,
            "min_bike_lane_overlap_ratio": 0.005,
            "require_centroid_in_lane": False,
        },
        "advance warning sign": {
            "min_intersection_area": 1200.0,
            "min_hazard_overlap_ratio": 0.10,
            "min_bike_lane_overlap_ratio": 0.005,
            "require_centroid_in_lane": False,
        },
    }

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
    if work.exists():
        shutil.rmtree(work)

    video_dir = work / "video"
    gpx_dir = work / "gpx"
    frames_dir = work / "frames"
    out_dir = work / "out"
    overlay_dir = out_dir / "overlays"

    video_dir.mkdir(parents=True, exist_ok=True)
    gpx_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / "input.mp4"
    gpx_path = gpx_dir / "track.gpx"

    # ----------------------------
    # Helpers
    # ----------------------------
    def parse_iso(dt_str: str) -> datetime:
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

    def should_treat_as_bike_lane(class_name: str) -> bool:
        return class_name.lower().strip() in BIKE_LANE_CLASS_NAMES

    def should_treat_as_hazard(class_name: str) -> bool:
        class_name = class_name.lower().strip()
        if should_treat_as_bike_lane(class_name):
            return False
        if HAZARD_CLASS_ALLOWLIST:
            return class_name in HAZARD_CLASS_ALLOWLIST
        return True

    def polygon_from_mask(mask_xy):
        if mask_xy is None or len(mask_xy) < 3:
            return None
        try:
            poly = Polygon(mask_xy)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 0:
                return None
            return poly
        except Exception:
            return None

    def clean_polygon(geom):
        if geom is None:
            return None
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty or geom.area <= 0:
                return None
            return geom
        except Exception:
            return None

    def shrink_geometry(geom, pixels: float):
        geom = clean_polygon(geom)
        if geom is None:
            return None
        if pixels == 0:
            return geom
        try:
            shrunk = geom.buffer(-pixels)
            shrunk = clean_polygon(shrunk)
            return shrunk
        except Exception:
            return geom

    def get_rule_for_class(class_name: str):
        class_name = class_name.lower().strip()
        return CLASS_RULES.get(
            class_name,
            {
                "min_intersection_area": DEFAULT_MIN_INTERSECTION_AREA,
                "min_hazard_overlap_ratio": DEFAULT_MIN_HAZARD_OVERLAP_RATIO,
                "min_bike_lane_overlap_ratio": DEFAULT_MIN_BIKE_LANE_OVERLAP_RATIO,
                "require_centroid_in_lane": DEFAULT_REQUIRE_CENTROID_IN_LANE,
            },
        )

    def evaluate_substantial_overlap(hazard_poly, bike_lane_poly, class_name: str):
        hazard_poly = clean_polygon(hazard_poly)
        bike_lane_poly = clean_polygon(bike_lane_poly)

        if hazard_poly is None or bike_lane_poly is None:
            return {
                "is_overlap": False,
                "intersection_area": 0.0,
                "hazard_overlap_ratio": 0.0,
                "bike_lane_overlap_ratio": 0.0,
                "centroid_in_lane": False,
            }

        if hazard_poly.area <= 0 or bike_lane_poly.area <= 0:
            return {
                "is_overlap": False,
                "intersection_area": 0.0,
                "hazard_overlap_ratio": 0.0,
                "bike_lane_overlap_ratio": 0.0,
                "centroid_in_lane": False,
            }

        inter = hazard_poly.intersection(bike_lane_poly)
        inter = clean_polygon(inter)

        if inter is None:
            return {
                "is_overlap": False,
                "intersection_area": 0.0,
                "hazard_overlap_ratio": 0.0,
                "bike_lane_overlap_ratio": 0.0,
                "centroid_in_lane": False,
            }

        intersection_area = float(inter.area)
        hazard_overlap_ratio = intersection_area / float(hazard_poly.area)
        bike_lane_overlap_ratio = intersection_area / float(bike_lane_poly.area)

        centroid_in_lane = False
        try:
            centroid_in_lane = bool(bike_lane_poly.contains(hazard_poly.centroid))
        except Exception:
            centroid_in_lane = False

        rule = get_rule_for_class(class_name)

        passes = (
            intersection_area >= rule["min_intersection_area"]
            and hazard_overlap_ratio >= rule["min_hazard_overlap_ratio"]
            and bike_lane_overlap_ratio >= rule["min_bike_lane_overlap_ratio"]
        )

        if rule["require_centroid_in_lane"]:
            passes = passes and centroid_in_lane

        return {
            "is_overlap": passes,
            "intersection_area": intersection_area,
            "hazard_overlap_ratio": hazard_overlap_ratio,
            "bike_lane_overlap_ratio": bike_lane_overlap_ratio,
            "centroid_in_lane": centroid_in_lane,
        }

    def shapely_to_cv_polylines(geom):
        """
        Returns a list of np.int32 arrays shaped for cv2.polylines.
        Supports Polygon and MultiPolygon.
        """
        lines = []

        geom = clean_polygon(geom)
        if geom is None:
            return lines

        def exterior_to_pts(poly):
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            if len(coords) >= 2:
                return coords.reshape((-1, 1, 2))
            return None

        if isinstance(geom, Polygon):
            pts = exterior_to_pts(geom)
            if pts is not None:
                lines.append(pts)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                pts = exterior_to_pts(poly)
                if pts is not None:
                    lines.append(pts)

        return lines

    def draw_outline(image, geom, color, thickness=2):
        for pts in shapely_to_cv_polylines(geom):
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_label(image, geom, text, color):
        geom = clean_polygon(geom)
        if geom is None:
            return
        try:
            x = int(geom.representative_point().x)
            y = int(geom.representative_point().y)
            cv2.putText(
                image,
                text,
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        except Exception:
            return

    def render_overlay(frame_path: Path, bike_lane_geom, kept_hazards, save_path: Path):
        image = cv2.imread(str(frame_path))
        if image is None:
            return False

        # Bike lane outline in green
        draw_outline(image, bike_lane_geom, color=(0, 255, 0), thickness=2)

        # Hazard outlines in red, labels in red
        for hazard in kept_hazards:
            draw_outline(image, hazard["shape"], color=(0, 0, 255), thickness=3)
            label = (
                f'{hazard["class_name"]} '
                f'ov={hazard["hazard_overlap_ratio"]:.2f} '
                f'area={int(hazard["intersection_area"])}'
            )
            draw_label(image, hazard["shape"], label, color=(0, 0, 255))

        ok = cv2.imwrite(str(save_path), image)
        return bool(ok)

    def upload_file_and_get_url(bucket: str, object_path: str, local_path: Path, content_type: str) -> str:
        with open(local_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                object_path,
                f,
                {"content-type": content_type, "x-upsert": "true"},
            )
        return supabase.storage.from_(bucket).get_public_url(object_path)

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

    video_start_dt = parse_iso(video_start_iso) if video_start_iso else gps_track[0]["timestamp"]

    # ----------------------------
    # Extract frames
    # ----------------------------
    print(f"Extracting {FRAME_FPS_EXTRACT} fps frames...")
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
        save=False,   # important: do not use default boxed overlays
        conf=MIN_CONFIDENCE,
        verbose=True,
    )

    summary = []
    db_rows = []
    uploaded_overlay_count = 0

    for i, r in enumerate(results):
        frame_name = frame_files[i].name
        frame_path = frame_files[i]
        overlay_path = overlay_dir / frame_name

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

        if masks is None or not hasattr(masks, "xy") or len(masks.xy) == 0:
            summary.append(
                {
                    "frame_index": i,
                    "timestamp_seconds": i / FRAME_FPS_EXTRACT,
                    "frame_file": frame_name,
                    "kept": False,
                    "reason": "no_masks_available",
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

            if j < len(masks.xy):
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
        bike_lane_union = clean_polygon(bike_lane_union)
        if bike_lane_union is None:
            summary.append(
                {
                    "frame_index": i,
                    "timestamp_seconds": i / FRAME_FPS_EXTRACT,
                    "frame_file": frame_name,
                    "kept": False,
                    "reason": "invalid_bike_lane_geometry",
                    "detections": [],
                }
            )
            continue

        # Shrink bike lane so tiny boundary touches do not count.
        bike_lane_for_overlap = shrink_geometry(bike_lane_union, BIKE_LANE_SHRINK_PIXELS)
        if bike_lane_for_overlap is None:
            bike_lane_for_overlap = bike_lane_union

        kept_hazards = []

        for det in all_detections:
            if not should_treat_as_hazard(det["class_name"]):
                continue
            if det["shape"] is None:
                continue

            overlap = evaluate_substantial_overlap(
                hazard_poly=det["shape"],
                bike_lane_poly=bike_lane_for_overlap,
                class_name=det["class_name"],
            )

            if overlap["is_overlap"]:
                kept_hazards.append(
                    {
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "polygon": det["polygon"],
                        "shape": det["shape"],
                        "intersection_area": overlap["intersection_area"],
                        "hazard_overlap_ratio": overlap["hazard_overlap_ratio"],
                        "bike_lane_overlap_ratio": overlap["bike_lane_overlap_ratio"],
                        "centroid_in_lane": overlap["centroid_in_lane"],
                    }
                )

        if not kept_hazards:
            summary.append(
                {
                    "frame_index": i,
                    "timestamp_seconds": i / FRAME_FPS_EXTRACT,
                    "frame_file": frame_name,
                    "kept": False,
                    "reason": "no_substantial_hazard_overlap",
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

        overlay_written = render_overlay(
            frame_path=frame_path,
            bike_lane_geom=bike_lane_union,   # draw full lane outline
            kept_hazards=kept_hazards,
            save_path=overlay_path,
        )

        if overlay_written and overlay_path.exists():
            overlay_object_path = f"{job_name}/{frame_name}"
            image_url = upload_file_and_get_url(
                OVERLAYS_BUCKET,
                overlay_object_path,
                overlay_path,
                "image/jpeg",
            )
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
                "intersecting_hazards": [
                    {
                        "class_name": h["class_name"],
                        "confidence": h["confidence"],
                        "intersection_area": h["intersection_area"],
                        "hazard_overlap_ratio": h["hazard_overlap_ratio"],
                        "bike_lane_overlap_ratio": h["bike_lane_overlap_ratio"],
                        "centroid_in_lane": h["centroid_in_lane"],
                        "polygon": h["polygon"],
                    }
                    for h in kept_hazards
                ],
            }
        )

        for hazard in kept_hazards:
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
                    "bike_lane_overlap": hazard["hazard_overlap_ratio"],
                    "intersection_area": hazard["intersection_area"],
                    "bike_lane_overlap_ratio": hazard["bike_lane_overlap_ratio"],
                    "centroid_in_lane": hazard["centroid_in_lane"],
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
        "config": {
            "bike_lane_shrink_pixels": BIKE_LANE_SHRINK_PIXELS,
            "vehicle_rule": CLASS_RULES["vehicle"],
            "default_rule": {
                "min_intersection_area": DEFAULT_MIN_INTERSECTION_AREA,
                "min_hazard_overlap_ratio": DEFAULT_MIN_HAZARD_OVERLAP_RATIO,
                "min_bike_lane_overlap_ratio": DEFAULT_MIN_BIKE_LANE_OVERLAP_RATIO,
                "require_centroid_in_lane": DEFAULT_REQUIRE_CENTROID_IN_LANE,
            },
        },
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
