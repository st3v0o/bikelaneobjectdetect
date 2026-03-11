import os
import uuid
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from supabase import create_client
import modal

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "2000")) * 1024 * 1024

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Buckets for source uploads
SOURCE_VIDEO_BUCKET = os.getenv("SOURCE_VIDEO_BUCKET", "source-videos")
SOURCE_GPX_BUCKET = os.getenv("SOURCE_GPX_BUCKET", "source-gpx")

# Your deployed Modal app + function names
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "bike-lane-video-inference")
MODAL_FUNCTION_NAME = os.getenv("MODAL_FUNCTION_NAME", "run_video")

# Signed URL validity for Modal to download inputs
SIGNED_URL_TTL_SECONDS = int(os.getenv("SIGNED_URL_TTL_SECONDS", "3600"))

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def upload_local_file_to_supabase(bucket: str, object_path: str, local_path: Path, content_type: str) -> str:
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


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


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

    # Look up deployed Modal function and submit asynchronously
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
