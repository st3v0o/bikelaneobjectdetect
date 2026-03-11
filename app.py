import os
import uuid
import traceback

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from supabase import create_client
import modal


app = Flask(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]

SOURCE_VIDEO_BUCKET = os.getenv("SOURCE_VIDEO_BUCKET", "source-videos")
SOURCE_GPX_BUCKET = os.getenv("SOURCE_GPX_BUCKET", "source-gpx")

MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "bike-lane-video-inference")
MODAL_FUNCTION_NAME = os.getenv("MODAL_FUNCTION_NAME", "run_video")

SIGNED_URL_TTL_SECONDS = int(os.getenv("SIGNED_URL_TTL_SECONDS", "3600"))

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def create_signed_download_url(bucket: str, object_path: str) -> str:
    signed = supabase.storage.from_(bucket).create_signed_url(
        object_path,
        SIGNED_URL_TTL_SECONDS,
    )
    signed_url = signed.get("signedURL") or signed.get("signedUrl")
    if not signed_url:
        raise RuntimeError(f"Could not create signed URL for {bucket}/{object_path}")
    return signed_url


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify(
        {
            "error": "Upload too large",
        }
    ), 413


@app.errorhandler(Exception)
def handle_unexpected_error(e):
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


@app.post("/submit-job")
def submit_job():
    app.logger.info("Entered /submit-job route")

    job_name = (request.form.get("job_name") or "").strip()
    video_object_path = (request.form.get("video_object_path") or "").strip()
    gpx_object_path = (request.form.get("gpx_object_path") or "").strip()
    video_start_iso = (request.form.get("video_start_iso") or "").strip()

    if not job_name:
        job_name = f"job-{uuid.uuid4().hex[:12]}"

    if not video_object_path:
        return jsonify({"error": "Missing video_object_path"}), 400

    if not gpx_object_path:
        return jsonify({"error": "Missing gpx_object_path"}), 400

    try:
        app.logger.info("Creating signed video URL")
        video_url = create_signed_download_url(SOURCE_VIDEO_BUCKET, video_object_path)

        app.logger.info("Creating signed GPX URL")
        gpx_url = create_signed_download_url(SOURCE_GPX_BUCKET, gpx_object_path)

        app.logger.info("Looking up Modal function")
        fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)

        app.logger.info("Spawning Modal job")
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
