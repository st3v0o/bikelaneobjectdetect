from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route("/")
def home():
    return {"status": "running"}

@app.route("/process", methods=["POST"])
def process_video():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]

    video_path = "input_video.mp4"
    video.save(video_path)

    # run your script
    subprocess.run(["python", "process_video.py", video_path])

    return {"status": "processing complete"}