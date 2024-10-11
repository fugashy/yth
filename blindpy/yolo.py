from collections import namedtuple

import click
import pandas as pd

from ultralytics import YOLO
import torch


_COLUMN_NAMES = [
    "frame_id", "tracking_id", "cls", "conf", "x1", "y1", "x2", "y2"]


@click.group()
def yolo():
    pass


@yolo.command()
@click.argument("input_filepath", type=str)
@click.option("--output_filepath", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--model-name", type=str, default="yolov8s.pt")
def predict(
        input_filepath,
        output_filepath,
        model_name):
    if not torch.backends.mps.is_available():
        print("MPS is not available...")
        return

    model = YOLO(model_name)

    results = model(source=input_filepath, show=False, save=False, device="mps")
    tuples = list()
    Bbx = namedtuple("bbx", _COLUMN_NAMES)
    bbxs = list()
    for i, r in enumerate(results):
        frame_id = i
        for bbx in r.boxes:
            tracking_id = bbx.id
            cls = int(bbx.cls)
            conf = float(bbx.conf)
            xyxy = bbx.xyxy
            x1 = float(xyxy[0,0])
            y1 = float(xyxy[0,1])
            x2 = float(xyxy[0,2])
            y2 = float(xyxy[0,3])

            bbxs.append(Bbx(frame_id, tracking_id, cls, conf, x1, y1, x2, y2))

    df = pd.DataFrame(bbxs)
    df.to_csv(output_filepath)
    print(f"output result to {output_filepath}")
