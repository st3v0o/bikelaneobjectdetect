import json
import os
import shutil
import subprocess
import uuid
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
    run_id: str,
    video_start_iso: str | None = None,
):
    import cv2
    import numpy as np
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from supabase import create_client
    from ultralytics import YOLO

    SCRIPT_VERSION = "debug-conf06-status-v1"

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
    MIN_CONFIDENCE = 0.6

    BIKE_LANE_SHRINK_PIXELS = 8
    MIN_INTERSECTION_AREA = 3000.0
    MIN_HAZARD_OVERLAP_RATIO = 0.25
    MIN_BIKE_LANE_OVERLAP_RATIO = 0.015
    REQUIRE_CENTROID_IN_LANE = False

    UPSERT_TO_DB = True
    DETECTIONS_TABLE = "detections"
    JOB_RUNS_TABLE = "job_runs"
    OVERLAYS_BUCKET = "overlays"
    RESULTS_BUCKET = "results"

    MODEL_PATH = "/root/models/models/best.pt"

    # Debug styling so you instantly know this renderer is being used
    BIKE_LANE_COLOR = (255, 255, 255)  # white
    HAZARD_COLOR = (0, 255, 255)       # yellow
    BIKE_LANE_THICKNESS = 8
    HAZARD_THICKNESS = 10

    # ----------------------------
    # Supabase
    # ----------------------------
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase = create_client(supabase_url, supabase_key)

    def update_job_status(status: str, step: str, message: str = "", extra: dict | None = None):
        payload = {
            "job_name": job_name,
            "run_id": run_id,
            "status": status,
            "step": step,
            "message": message,
            "script_version": SCRIPT_VERSION,
            "min_confidence": MIN_CONFIDENCE,
        }
        if extra:
            payload.update(extra)

        try:
            supabase.table(JOB_RUNS_TABLE).upsert(payload).execute()
        except Exception as e:
            print(f"Warning: failed to update job status: {e}")

    print(f"SCRIPT_VERSION={SCRIPT_VERSION}")
    print(f"JOB_NAME={job_name}")
    print(f"RUN_ID={run_id}")
    print(f"MIN_CONFIDENCE={MIN_CONFIDENCE}")

    assert abs(MIN_CONFIDENCE - 0.6) < 1e-9, f"Expected 0.6, got {MIN_CONFIDENCE}"

    update_job_status("running", "starting", "Job started")

    # Delete old DB rows for this job so results are not mixed
    if UPSERT_TO_DB:
        try:
            supabase.table(DETECTIONS_TABLE).delete().eq("job_name", job_name).execute()
            print(f"Deleted old rows for job_name={job_name}")
        except Exception as e:
            print(f"Warning: failed deleting old rows for {job_name}: {e}")

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

    def clean_geometry(geom):
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
        geom = clean_geometry(geom)
        if geom is None:
            return None
        if pixels == 0:
            return geom
        try:
            shrunk = geom.buffer(-pixels)
            shrunk = clean_geometry(shrunk)
            return shrunk
        except Exception:
            return geom

    def evaluate_substantial_overlap(hazard_poly, bike_lane_poly):
        hazard_poly = clean_geometry(hazard_poly)
        bike_lane_poly = clean_geometry(bike_lane_poly)

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
        inter = clean_geometry(inter)

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

        passes = (
            intersection_area >= MIN_INTERSECTION_AREA
            and hazard_overlap_ratio >= MIN_HAZARD_OVERLAP_RATIO
            and bike_lane_overlap_ratio >= MIN_BIKE_LANE_OVERLAP_RATIO
        )

        if REQUIRE_CENTROID_IN_LANE:
            passes = passes and centroid_in_lane

        return {
            "is_overlap": passes,
            "intersection_area": intersection_area,
            "hazard_overlap_ratio": hazard_overlap_ratio,
            "bike_lane_overlap_ratio": bike_lane_overlap_ratio,
            "centroid_in_lane": centroid_in_lane,
        }

    def geometry_to_polylines(geom):
        geom = clean_geometry(geom)
        if geom is None:
            return []

        lines = []

        def polygon_exterior_to_pts(poly):
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            if len(coords) < 2:
                return None
            return coords.reshape((-1, 1, 2))

        if isinstance(geom, Polygon):
            pts = polygon_exterior_to_pts(geom)
            if pts is not None:
                lines.append(pts)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                pts = polygon_exterior_to_pts(poly)
                if pts is not None:
                    lines.append(pts)

        return lines

    def draw_outline(image, geom, color, thickness):
        for pts in geometry_to_polylines(geom):
            cv2.polylines(
                image,
                [pts],
                isClosed=True,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    def draw_label(image, geom, text, color):
        geom = clean_geometry(geom)
        if geom is None:
            return
        try:
            rp = geom.representative_point()
            x = int(rp.x)
            y = int(rp.y)
            cv2.putText(
                image,
                text,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    def render_overlay(frame_path: Path, bike_lane_geom, kept_hazards, save_path: Path) -> bool:
        image = cv2.imread(str(frame_path))
        if image is None:
            return False

        draw_outline(image, bike_lane_geom, BIKE_LANE_COLOR, BIKE_LANE_THICKNESS)

        for hazard in kept_hazards:
            draw_outline(image, hazard["shape"], HAZARD_COLOR, HAZARD_THICKNESS)
            label = f'{hazard["class_name"]} conf={hazard["confidence"]:.2f}'
            draw_label(image, hazard["shape"], label, HAZARD_COLOR)

        cv2.putText(
            image,
            f"DEBUG {SCRIPT_VERSION} conf={MIN_CONFIDENCE} run={run_id}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )

        return bool(cv2.imwrite(str(save_path), image))

    def upload_file_and_get_url(bucket: str, object_path: str, local_path: Path, content_type: str) -> str:
        with open(local_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                object_path,
                f,
                {"content-type": content_type, "x-upsert": "true"},
            )
        return supabase.storage.from_(bucket).get_public_url(object_path)

    try:
        # ----------------------------
        # Download inputs
        # ----------------------------
        update_job_status("running", "downloading_inputs", "Downloading video and GPX")
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
        update_job_status("running", "extracting_frames", "Extracting frames from video")
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
        update_job_status("running", "loading_model", "Loading YOLO model")
        print(f"Loading model from {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

        # ----------------------------
        # Run inference
        # ----------------------------
        update_job_status("running", "running_inference", "Running inference")
        print("Running inference...")
        results = model.predict(
            source=str(frames_dir),
            save=False,
            conf=MIN_CONFIDENCE,
            verbose=True,
        )

        for frame_idx, r in enumerate(results):
            if r.boxes is None or len(r.boxes) == 0:
                continue

            confs = [float(x.item()) for x in r.boxes.conf]
            if confs:
                min_seen = min(confs)
                max_seen = max(confs)
                print(f"Frame {frame_idx}: confidence range {min_seen:.4f} - {max_seen:.4f}")

                if min_seen < MIN_CONFIDENCE:
                    raise RuntimeError(
                        f"Returned confidence {min_seen:.4f} below threshold {MIN_CONFIDENCE}"
                    )

        summary = []
        db_rows = []
        uploaded_overlay_count = 0

        update_job_status("running", "processing_detections", "Processing detections")

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
            bike_lane_union = clean_geometry(bike_lane_union)

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
                bike_lane_geom=bike_lane_union,
                kept_hazards=kept_hazards,
                save_path=overlay_path,
            )

            if overlay_written and overlay_path.exists():
                update_job_status("running", "uploading_overlays", f"Uploading overlay for {frame_name}")
                overlay_object_path = f"{job_name}/{run_id}/{frame_name}"
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
                        "run_id": run_id,
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
                        "script_version": SCRIPT_VERSION,
                    }
                )

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        update_job_status("running", "saving_summary", "Uploading summary and debug files")

        summary_object_path = f"{job_name}/{run_id}/summary.json"
        with open(summary_path, "rb") as f:
            supabase.storage.from_(RESULTS_BUCKET).upload(
                summary_object_path,
                f,
                {"content-type": "application/json", "x-upsert": "true"},
            )

        summary_url = supabase.storage.from_(RESULTS_BUCKET).get_public_url(summary_object_path)

        debug_payload = {
            "script_version": SCRIPT_VERSION,
            "job_name": job_name,
            "run_id": run_id,
            "min_confidence": MIN_CONFIDENCE,
            "frame_fps_extract": FRAME_FPS_EXTRACT,
            "model_path": MODEL_PATH,
            "config": {
                "bike_lane_shrink_pixels": BIKE_LANE_SHRINK_PIXELS,
                "min_intersection_area": MIN_INTERSECTION_AREA,
                "min_hazard_overlap_ratio": MIN_HAZARD_OVERLAP_RATIO,
                "min_bike_lane_overlap_ratio": MIN_BIKE_LANE_OVERLAP_RATIO,
                "require_centroid_in_lane": REQUIRE_CENTROID_IN_LANE,
            },
            "overlay_style": {
                "bike_lane_color": BIKE_LANE_COLOR,
                "hazard_color": HAZARD_COLOR,
                "bike_lane_thickness": BIKE_LANE_THICKNESS,
                "hazard_thickness": HAZARD_THICKNESS,
            },
        }

        debug_path = out_dir / "debug.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, indent=2)

        debug_object_path = f"{job_name}/{run_id}/debug.json"
        with open(debug_path, "rb") as f:
            supabase.storage.from_(RESULTS_BUCKET).upload(
                debug_object_path,
                f,
                {"content-type": "application/json", "x-upsert": "true"},
            )

        debug_url = supabase.storage.from_(RESULTS_BUCKET).get_public_url(debug_object_path)

        inserted_count = 0
        if UPSERT_TO_DB and db_rows:
            update_job_status("running", "writing_database", "Writing detections to database")
            chunk_size = 500
            for start in range(0, len(db_rows), chunk_size):
                chunk = db_rows[start:start + chunk_size]
                supabase.table(DETECTIONS_TABLE).insert(chunk).execute()
                inserted_count += len(chunk)

        update_job_status(
            "complete",
            "done",
            "Processing complete",
            {
                "summary_url": summary_url,
                "debug_url": debug_url,
                "frames_extracted": len(frame_files),
                "frames_kept": len([x for x in summary if x.get("kept")]),
                "overlay_images_uploaded": uploaded_overlay_count,
                "db_rows_inserted": inserted_count,
            },
        )

        return {
            "job_name": job_name,
            "run_id": run_id,
            "script_version": SCRIPT_VERSION,
            "min_confidence": MIN_CONFIDENCE,
            "frames_extracted": len(frame_files),
            "frames_kept": len([x for x in summary if x.get("kept")]),
            "overlay_images_uploaded": uploaded_overlay_count,
            "db_rows_inserted": inserted_count,
            "summary_url": summary_url,
            "debug_url": debug_url,
            "config": {
                "bike_lane_shrink_pixels": BIKE_LANE_SHRINK_PIXELS,
                "min_intersection_area": MIN_INTERSECTION_AREA,
                "min_hazard_overlap_ratio": MIN_HAZARD_OVERLAP_RATIO,
                "min_bike_lane_overlap_ratio": MIN_BIKE_LANE_OVERLAP_RATIO,
                "require_centroid_in_lane": REQUIRE_CENTROID_IN_LANE,
            },
        }

    except Exception as e:
        update_job_status("failed", "error", str(e))
        raise


@app.local_entrypoint()
def main(
    video_url: str,
    gpx_url: str,
    job_name: str,
    run_id: str = "",
    video_start_iso: str = "",
):
    result = run_video.remote(
        video_url=video_url,
        gpx_url=gpx_url,
        job_name=job_name,
        run_id=run_id or uuid.uuid4().hex[:10],
        video_start_iso=video_start_iso or None,
    )
    print(result)
