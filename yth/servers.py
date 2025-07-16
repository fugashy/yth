from flask import Flask, render_template, Response
import click
import cv2
import numpy as np




@click.group()
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=50280)
@click.pass_context
def servers(ctx, host, port):
    ctx.obj = dict()
    ctx.obj["host"] = host
    ctx.obj["port"] = port



def _fugashy(camera_id):
    cap = cv2.VideoCapture(camera_id)
    kernel = np.ones((3, 3), np.uint8)


    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.Canny(frame, threshold1=50, threshold2=150)

        core = cv2.dilate(frame, kernel, iterations=4)
        out = cv2.dilate(core, kernel, iterations=4)

        outline = cv2.subtract(out, core)

        h, w = out.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        rgba[out > 0] = (0, 0, 0, 255)
        rgba[core > 0] = (255, 255, 255, 255)

#       alpha = np.zeros_like(frame)
#       alpha[frame > 0] = 255

#       frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#       frame = cv2.merge([frame, alpha])

        _, buffer = cv2.imencode(".png", rgba)
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@servers.command()
@click.option("--camera_id", type=int, default=0)
@click.pass_context
def fugashy(ctx, camera_id):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
                _fugashy(camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame')
    app.run(host=ctx.obj["host"], port=ctx.obj["port"])
