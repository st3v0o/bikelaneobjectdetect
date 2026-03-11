import os
import uuid
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from supabase import create_client
import modal


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "2000")) * 1024 * 1024

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# These are for browser-side Leaflet map access. Use your public anon key here.
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]

# Source upload buckets (private is fine here)
SOURCE_VIDEO_BUCKET = os.getenv("SOURCE_VIDEO_BUCKET", "source-videos")
SOURCE_GPX_BUCKET = os.getenv("SOURCE_GPX_BUCKET", "source-gpx")

# Your Modal app/function names
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "bike-lane-video-inference")
MODAL_FUNCTION_NAME = os.getenv("MODAL_FUNCTION_NAME", "run_video")

# Signed URL lifetime for Modal to download source files
SIGNED_URL_TTL_SECONDS = int(os.getenv("SIGNED_URL_TTL_SECONDS", "3600"))

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def upload_local_file_to_supabase(bucket: str, object_path: str, local_path: Path, content_type: str) -> str:
    """
    Upload a local file to Supabase Storage and return a signed URL.
    Intended for private source buckets so Modal can download the file.
    """
    with open(local_path, "rb") as f:
        supabase.storage.from_(bucket).upload(
            object_path,
            f,
            {
                "content-type": content_type,
                "x-upsert": "true",
            },
        )

    signed = supabase.storage.from_(bucket).create_signed_url(object_path, SIGNED_URL_TTL_SECONDS)
    signed_url = signed.get("signedURL") or signed.get("signedUrl")

    if not signed_url:
        raise RuntimeError(f"Could not create signed URL for {bucket}/{object_path}")

    return signed_url


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify(
        {
            "error": "Upload too large",
            "max_upload_mb": int(os.getenv("MAX_UPLOAD_MB", "2000")),
        }
    ), 413


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/map")
def map_page():
    """
    Serves the embeddable Leaflet map page.
    The actual map JS lives in templates/map.html.
    """
    return render_template(
        "map.html",
        supabase_url=SUPABASE_URL,
        supabase_anon_key=SUPABASE_ANON_KEY,
    )


@app.post("/upload")
def upload():
    video = request.files.get("video")
    gpx = request.files.get("gpx")
    job_name = (request.form.get("job_name") or "").strip()
    video_start_iso = (request.form.get("video_start_iso") or "").strip()

    if not video or not video.filename:
        return jsonify({"error": "Missing video file"}), 400

    if not gpx or not gpx.filename:
        return jsonify({"error": "Missing GPX file"}), 400

    if not job_name:
        job_name = f"job-{uuid.uuid4().hex[:12]}"

    safe_video_name = secure_filename(video.filename)
    safe_gpx_name = secure_filename(gpx.filename)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            video_path = tmpdir / safe_video_name
            gpx_path = tmpdir / safe_gpx_name

            video.save(video_path)
            gpx.save(gpx_path)

            video_object_path = f"{job_name}/{safe_video_name}"
            gpx_object_path = f"{job_name}/{safe_gpx_name}"

            video_url = upload_local_file_to_supabase(
                SOURCE_VIDEO_BUCKET,
                video_object_path,
                video_path,
                video.mimetype or "video/mp4",
            )

            gpx_url = upload_local_file_to_supabase(
                SOURCE_GPX_BUCKET,
                gpx_object_path,
                gpx_path,
                gpx.mimetype or "application/gpx+xml",
            )

        # Trigger deployed Modal function asynchronously
        fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
        function_call = fn.spawn(
            video_url=video_url,
            gpx_url=gpx_url,
            job_name=job_name,
            video_start_iso=video_start_iso or None,
        )

        return jsonify(
            {
                "status": "submitted",
                "job_name": job_name,
                "modal_call_id": function_call.object_id,
                "video_object_path": video_object_path,
                "gpx_object_path": gpx_object_path,
            }
        )

    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "type": e.__class__.__name__,
            }
        ), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
