from collections import namedtuple
from dataclasses import dataclass

import click
from ultralytics import YOLO
import torch
import pandas as pd
import cv2
import moviepy.editor as mp
from tqdm import tqdm

from . import draw_utils



_COLUMN_NAMES = [
    "frame_id", "tracking_id", "cls", "conf", "x1", "y1", "x2", "y2"]


@click.group()
def blindpy():
    pass


@blindpy.command()
@click.argument("input_filepath", type=str)
@click.option("--output_filepath", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--model-name", type=str, default="yolov8s.pt")
def yolo(
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


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_num: int


def get_video_info(video):
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return VideoInfo(width, height, fps, frame_num)



@blindpy.command()
@click.argument("video_path", type=str)
@click.option("--result_path", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--output_video_path", type=str, default="/tmp/blindpy.mp4")
@click.option("--style", type=click.Choice(["rect", "image"]), default="rect")
@click.option("--draw_image_path", type=str, default="")
@click.option("--show-once", is_flag=True, default=False)
def seal(video_path, result_path, output_video_path, style, draw_image_path, show_once):
    video = cv2.VideoCapture(video_path)
    info = get_video_info(video)
    print(f"size: ({info.width}, {info.height}), fps: {info.fps:1.2f}, num: {info.frame_num}")

    results = pd.read_csv(result_path, header=0)

    fmt = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_video_path = "/tmp/tmp.mp4"
    writer = cv2.VideoWriter(
            tmp_video_path,
            fmt,
            info.fps, (info.width, info.height))
    if not writer.isOpened():
        print("failed to create a writer")
        return

    print("try to process all frames...")
    for frame_id in tqdm(range(info.frame_num)):
        ret, img = video.read()
        if not ret:
            print(f"failed to read image[{frame_id}]")
            break
        result = results[results["frame_id"] == frame_id]
        if not result.empty:
            img = draw_utils.call(
                    style,
                    img,
                    result,
                    {
                        "draw_image": draw_image_path,
                    })
            if show_once:
                import matplotlib.pyplot as plt
                plt.imshow(img)
                plt.show()
                return

        writer.write(img)

    video.release()
    writer.release()

    # 音声をつける
    print("try to set audio...")
    clip_input = mp.VideoFileClip(video_path)
    clip_input.audio.write_audiofile("/tmp/audio.mp3")
    clip = mp.VideoFileClip(tmp_video_path)
    audio = mp.AudioFileClip("/tmp/audio.mp3")
    clip = clip.set_audio(audio)
    clip.write_videofile(
            output_video_path,codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True)



def entry_point():
    blindpy()
