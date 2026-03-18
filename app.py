import os
import uuid
import traceback
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException

from supabase import create_client
import modal


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))  # 2 GB default

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

SOURCE_VIDEO_BUCKET = os.getenv("SOURCE_VIDEO_BUCKET", "source-videos")
SOURCE_GPX_BUCKET = os.getenv("SOURCE_GPX_BUCKET", "source-gpx")

MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "bike-lane-video-inference")
MODAL_FUNCTION_NAME = os.getenv("MODAL_FUNCTION_NAME", "run_video")

SIGNED_URL_TTL_SECONDS = int(os.getenv("SIGNED_URL_TTL_SECONDS", "3600"))

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mpeg", ".mpg")
GPX_EXTENSIONS = (".gpx",)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else None


def create_signed_download_url(bucket: str, object_path: str) -> str:
    signed = supabase.storage.from_(bucket).create_signed_url(
        object_path,
        SIGNED_URL_TTL_SECONDS,
    )
    signed_url = signed.get("signedURL") or signed.get("signedUrl")
    if not signed_url:
        raise RuntimeError(f"Could not create signed URL for {bucket}/{object_path}")
    return signed_url


def list_bucket_files_one_level(bucket_name: str) -> List[str]:
    """
    Lists files assuming a structure like:
      bucket/
        job-name/
          file.ext

    Returns paths relative to bucket root, e.g.:
      richmond-ride-001/input.mp4
      richmond-ride-001/track.gpx
    """
    results: List[str] = []

    top_level = supabase.storage.from_(bucket_name).list(
        path="",
        options={"limit": 1000, "offset": 0},
    )

    if not top_level:
        return results

    for folder in top_level:
        folder_name = folder.get("name")
        if not folder_name:
            continue

        children = supabase.storage.from_(bucket_name).list(
            path=folder_name,
            options={"limit": 1000, "offset": 0},
        )

        if not children:
            continue

        for child in children:
            child_name = child.get("name")
            if not child_name:
                continue

            full_path = f"{folder_name}/{child_name}"
            results.append(full_path)

    return sorted(results)


def filter_paths_by_extensions(paths: List[str], allowed_extensions: tuple) -> List[str]:
    return [p for p in paths if p.lower().endswith(allowed_extensions)]


def build_file_options(paths: List[str]) -> List[Dict[str, Any]]:
    """
    Converts raw storage paths into frontend-friendly dropdown options.
    """
    options: List[Dict[str, Any]] = []

    for path in paths:
        parts = path.split("/")
        folder = parts[0] if len(parts) > 1 else ""
        filename = parts[-1]

        label = f"{folder} / {filename}" if folder else filename

        options.append(
            {
                "path": path,
                "label": label,
                "folder": folder,
                "filename": filename,
            }
        )

    return options


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "Upload too large"}), 413


@app.errorhandler(404)
def handle_404(e):
    return jsonify(
        {
            "error": "Route not found",
            "path": request.path,
            "method": request.method,
        }
    ), 404


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    if isinstance(e, HTTPException):
        return jsonify(
            {
                "error": e.description,
                "type": e.__class__.__name__,
            }
        ), e.code

    app.logger.exception("Unhandled exception")
    return jsonify(
        {
            "error": str(e),
            "type": e.__class__.__name__,
            "traceback": traceback.format_exc(),
        }
    ), 500


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/map")
def map_page():
    return render_template(
        "map.html",
        supabase_url=SUPABASE_URL,
        supabase_anon_key=SUPABASE_ANON_KEY,
    )


@app.get("/list-storage-files")
def list_storage_files():
    """
    Returns available files from the source video and GPX buckets
    so the frontend can populate dropdowns.
    """
    app.logger.info("Entered /list-storage-files route")

    try:
        video_paths = list_bucket_files_one_level(SOURCE_VIDEO_BUCKET)
        gpx_paths = list_bucket_files_one_level(SOURCE_GPX_BUCKET)

        video_paths = filter_paths_by_extensions(video_paths, VIDEO_EXTENSIONS)
        gpx_paths = filter_paths_by_extensions(gpx_paths, GPX_EXTENSIONS)

        return jsonify(
            {
                "videos": build_file_options(video_paths),
                "gpx": build_file_options(gpx_paths),
                "video_bucket": SOURCE_VIDEO_BUCKET,
                "gpx_bucket": SOURCE_GPX_BUCKET,
                "video_count": len(video_paths),
                "gpx_count": len(gpx_paths),
            }
        )

    except Exception as e:
        app.logger.exception("Exception inside /list-storage-files")
        return jsonify(
            {
                "error": str(e),
                "type": e.__class__.__name__,
                "traceback": traceback.format_exc(),
            }
        ), 500


@app.get("/job-status/<job_name>/<run_id>")
def job_status(job_name, run_id):
    app.logger.info(f"Entered /job-status for job_name={job_name}, run_id={run_id}")

    try:
        res = (
            supabase.table("job_runs")
            .select("*")
            .eq("job_name", job_name)
            .eq("run_id", run_id)
            .limit(1)
            .execute()
        )

        rows = res.data or []
        if not rows:
            return jsonify(
                {
                    "status": "unknown",
                    "step": "not_found",
                    "message": "No job status found yet",
                    "job_name": job_name,
                    "run_id": run_id,
                }
            ), 404

        return jsonify(rows[0])

    except Exception as e:
        app.logger.exception("Exception inside /job-status")
        return jsonify(
            {
                "status": "error",
                "step": "status_lookup_failed",
                "message": str(e),
                "job_name": job_name,
                "run_id": run_id,
            }
        ), 500


@app.post("/submit-job")
def submit_job():
    app.logger.info("Entered /submit-job route")

    job_name = (request.form.get("job_name") or "").strip()
    video_object_path = (request.form.get("video_object_path") or "").strip()
    gpx_object_path = (request.form.get("gpx_object_path") or "").strip()
    video_start_iso = (request.form.get("video_start_iso") or "").strip()

    if not job_name:
        if "/" in video_object_path:
            job_name = video_object_path.split("/", 1)[0]
        else:
            job_name = f"job-{uuid.uuid4().hex[:12]}"

    if not video_object_path:
        return jsonify({"error": "Missing video_object_path"}), 400

    if not gpx_object_path:
        return jsonify({"error": "Missing gpx_object_path"}), 400

    run_id = uuid.uuid4().hex[:10]

    try:
        app.logger.info(f"Using Modal app: {MODAL_APP_NAME}, function: {MODAL_FUNCTION_NAME}")
        app.logger.info(f"job_name={job_name}, run_id={run_id}")

        video_url = create_signed_download_url(SOURCE_VIDEO_BUCKET, video_object_path)
        gpx_url = create_signed_download_url(SOURCE_GPX_BUCKET, gpx_object_path)

        fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)

        function_call = fn.spawn(
            video_url=video_url,
            gpx_url=gpx_url,
            job_name=job_name,
            run_id=run_id,
            video_start_iso=video_start_iso or None,
        )

        return jsonify(
            {
                "status": "submitted",
                "job_name": job_name,
                "run_id": run_id,
                "modal_call_id": function_call.object_id,
                "video_bucket": SOURCE_VIDEO_BUCKET,
                "video_object_path": video_object_path,
                "gpx_bucket": SOURCE_GPX_BUCKET,
                "gpx_object_path": gpx_object_path,
            }
        )

    except Exception as e:
        app.logger.exception("Exception inside /submit-job")
        return jsonify(
            {
                "error": str(e),
                "type": e.__class__.__name__,
                "traceback": traceback.format_exc(),
            }
        ), 500


@app.get("/upload")
def upload_page():
    return render_template("upload.html")


@app.post("/upload")
def handle_upload():
    if supabase is None:
        return jsonify({"error": "Supabase is not configured on this server"}), 503

    video_file = request.files.get("video")
    gpx_file = request.files.get("gpx")
    job_name = (request.form.get("job_name") or "").strip()
    video_start_iso = (request.form.get("video_start_iso") or "").strip()
    auto_submit = request.form.get("auto_submit") == "true"

    if not video_file or not video_file.filename:
        return jsonify({"error": "No video file provided"}), 400
    if not gpx_file or not gpx_file.filename:
        return jsonify({"error": "No GPX file provided"}), 400

    video_filename = video_file.filename
    gpx_filename = gpx_file.filename

    if not video_filename.lower().endswith(VIDEO_EXTENSIONS):
        return jsonify({"error": f"Unsupported video format. Allowed: {VIDEO_EXTENSIONS}"}), 400
    if not gpx_filename.lower().endswith(GPX_EXTENSIONS):
        return jsonify({"error": "File must be a .gpx file"}), 400

    if not job_name:
        job_name = f"upload-{uuid.uuid4().hex[:10]}"

    video_object_path = f"{job_name}/{video_filename}"
    gpx_object_path = f"{job_name}/{gpx_filename}"

    try:
        app.logger.info(f"Uploading video to {SOURCE_VIDEO_BUCKET}/{video_object_path}")
        supabase.storage.from_(SOURCE_VIDEO_BUCKET).upload(
            video_object_path,
            video_file.read(),
            {"content-type": video_file.content_type or "video/mp4", "x-upsert": "true"},
        )

        app.logger.info(f"Uploading GPX to {SOURCE_GPX_BUCKET}/{gpx_object_path}")
        supabase.storage.from_(SOURCE_GPX_BUCKET).upload(
            gpx_object_path,
            gpx_file.read(),
            {"content-type": "application/gpx+xml", "x-upsert": "true"},
        )

        result = {
            "status": "uploaded",
            "job_name": job_name,
            "video_object_path": video_object_path,
            "gpx_object_path": gpx_object_path,
        }

        if auto_submit:
            run_id = uuid.uuid4().hex[:10]
            video_url = create_signed_download_url(SOURCE_VIDEO_BUCKET, video_object_path)
            gpx_url = create_signed_download_url(SOURCE_GPX_BUCKET, gpx_object_path)

            fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
            function_call = fn.spawn(
                video_url=video_url,
                gpx_url=gpx_url,
                job_name=job_name,
                run_id=run_id,
                video_start_iso=video_start_iso or None,
            )

            result.update({
                "status": "submitted",
                "run_id": run_id,
                "modal_call_id": function_call.object_id,
            })

        return jsonify(result)

    except Exception as e:
        app.logger.exception("Exception inside /upload")
        return jsonify({
            "error": str(e),
            "type": e.__class__.__name__,
            "traceback": traceback.format_exc(),
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
