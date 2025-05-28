from flask import Flask, render_template, Response
import click
import cv2




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

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode(".jpg", frame)
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
                fugashy(camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame')
    app.run(host=ctx.obj["host"], port=ctx.obj["port"])
