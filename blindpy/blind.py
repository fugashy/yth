from dataclasses import dataclass

import click
import cv2
import moviepy.editor as mp
from tqdm import tqdm
import pandas as pd


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


class Drawer():

    @dataclass
    class Param():
        video_path: str
        result_path: str
        output_video_path: str
        targets: list
        show_once: bool
        with_audio: bool
        tmp_video_path: str

    def __init__(
            self,
            param: Param):
        self.p = param

        self._video = cv2.VideoCapture(self.p.video_path)
        self._info = get_video_info(self._video)
        print(
            f"size: ({self._info.width}, {self._info.height}), "
            f"fps: {self._info.fps:1.2f}, num: {self._info.frame_num}")
        self._results = pd.read_csv(self.p.result_path, header=0)

        fmt = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
                self.p.tmp_video_path,
                fmt,
                self._info.fps,
                (self._info.width, self._info.height))
        if not self._writer.isOpened():
            print("failed to create a writer")
            return

    def __del__(self):
        pass


    def run(self):
        print("try to process all frames...")
        for frame_id in tqdm(range(self._info.frame_num)):
            ret, img = self._video.read()
            if not ret:
                print(f"failed to read image[{frame_id}]")
                break
            result = self._results[self._results["frame_id"] == frame_id]
            if not result.empty:
                img = self._draw_impl(img, result)
                if self.p.show_once:
                    import matplotlib.pyplot as plt
                    plt.imshow(img)
                    plt.show()
                    return

            self._writer.write(img)

        self._video.release()
        self._writer.release()

        if self.p.with_audio:
            clip_input = mp.VideoFileClip(self.p.video_path)
            clip_input.audio.write_audiofile("/tmp/audio.mp3")
            clip = mp.VideoFileClip(self.p.tmp_video_path)
            audio = mp.AudioFileClip("/tmp/audio.mp3")
            clip = clip.set_audio(audio)
            clip.write_videofile(
                    self.p.output_video_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True)
        else:
            clip = mp.VideoFileClip(self.p.tmp_video_path)
            clip.write_videofile(
                    self.p.output_video_path,
                    codec='libx264',
                    remove_temp=True)

    def _draw_impl(self, img, result):
        raise NotImplementedError("not implemented")


class RectDrawer(Drawer):
    def __init__(
            self,
            param: Drawer.Param):
        super().__init__(param)

    def _draw_impl(self, img, result):
        for r in result.itertuples():
            if r.cls not in self.p.targets:
                continue
            p1 = (round(r.x1), round(r.y1))
            p2 = (round(r.x2), round(r.y2))
            color = (255, 0, 0)
            cv2.rectangle(img, p1, p2, color, 2)
        return img


class ImageDrawer(Drawer):
    def __init__(
            self,
            param: Drawer.Param,
            image_path):
        super().__init__(param)

        # アルファ値込みで使いたい
        self._overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    def _draw_impl(self, img, result):
        widths = result["x2"] - result["x1"]
        heights = result["y2"] - result["y1"]
        rs = result.copy()
        rs["largeness"] = widths * heights
        rs = rs.sort_values(by="largeness", ascending=True)

        for r in rs.itertuples():
            if r.cls not in self.p.targets:
                continue
            # 描画領域の左上と右下座標
            rect_width = round(r.x2) - round(r.x1)
            rect_height = round(r.y2) - round(r.y1)

            resized_overlay = cv2.resize(
                    self._overlay_img,
                    (rect_width, rect_height))

            overlay_bgr = resized_overlay[:, :, :3]  # BGRチャンネル
            overlay_alpha = resized_overlay[:, :, 3]  # アルファチャンネル
            # アルファ値を0-1に正規化
            overlay_alpha = overlay_alpha / 255.0

            # 矩形範囲内の背景画像部分を取り出す
            background_region = img[round(r.y1):round(r.y2), round(r.x1):round(r.x2)]

            # アルファチャンネルを使って背景とオーバーレイ画像をブレンド
            for c in range(0, 3):  # 各カラー（BGR）チャンネルで合成
                background_region[:, :, c] = overlay_alpha * overlay_bgr[:, :, c] + (1 - overlay_alpha) * background_region[:, :, c]

                # 合成結果を元の背景画像に適用
                img[round(r.y1):round(r.y2), round(r.x1):round(r.x2)] = background_region

        return img


class GaussianDrawer(Drawer):
    def __init__(
            self,
            param: Drawer.Param,
            kernel_size):
        super().__init__(param)
        self._kernel_size = kernel_size


    def _draw_impl(self, img, results):
        for r in results.itertuples():
            if r.cls not in self.p.targets:
                continue
            x = round((r.x1 + r.x2) / 2.)
            y = round((r.y1 + r.y2) / 2.)
            hw = round((r.x2 - r.x1) / 2.)
            hh = round((r.y2 - r.y1) / 2.)

            roi = img[y-hh:y+hh, x-hw:x+hw]
            blurred_roi = cv2.GaussianBlur(roi, [self._kernel_size, self._kernel_size], 0)
            img[y-hh:y+hh, x-hw:x+hw] = blurred_roi
        return img


class MosaicDrawer(Drawer):
    def __init__(
            self,
            param: Drawer.Param,
            scale):
        super().__init__(param)
        self._scale = scale


    def _draw_impl(self, img, results):
        for r in results.itertuples():
            if r.cls not in self.p.targets:
                continue
            x = round((r.x1 + r.x2) / 2.)
            y = round((r.y1 + r.y2) / 2.)
            hw = round((r.x2 - r.x1) / 2.)
            hh = round((r.y2 - r.y1) / 2.)

            roi = img[y-hh:y+hh, x-hw:x+hw]

            w = roi.shape[1]
            h = roi.shape[0]
            roi = cv2.resize(
                    roi,
                    (int(w / self._scale), int(h / self._scale)))
            roi = cv2.resize(
                    roi,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST)
            img[y-hh:y+hh, x-hw:x+hw] = roi
        return img


@click.group()
def blind():
    pass


@blind.command()
@click.argument("video_path", type=str)
@click.option("--result_path", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--output_video_path", type=str, default="/tmp/blindpy.mp4")
@click.option("--targets", type=list, default=[0, 1, 2, 3])
@click.option("--show-once", is_flag=True, default=False)
@click.option("--with-audio", is_flag=True, default=True)
def rect(
        video_path,
        result_path,
        output_video_path,
        targets,
        show_once,
        with_audio):
    drawer = RectDrawer(
            Drawer.Param(
                video_path=video_path,
                result_path=result_path,
                output_video_path=output_video_path,
                targets=targets,
                show_once=show_once,
                with_audio=with_audio,
                tmp_video_path="/tmp/tmp.mp4"))
    drawer.run()


@blind.command()
@click.argument("video_path", type=str)
@click.argument("image_path", type=str)
@click.option("--result_path", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--output_video_path", type=str, default="/tmp/blindpy.mp4")
@click.option("--targets", type=list, default=[0, 1, 2, 3])
@click.option("--show-once", is_flag=True, default=False)
@click.option("--with-audio", is_flag=True, default=True)
def image(
        video_path,
        image_path,
        result_path,
        output_video_path,
        targets,
        show_once,
        with_audio):
    drawer = ImageDrawer(
            Drawer.Param(
                video_path=video_path,
                result_path=result_path,
                output_video_path=output_video_path,
                targets=targets,
                show_once=show_once,
                with_audio=with_audio,
                tmp_video_path="/tmp/tmp.mp4"),
            image_path)
    drawer.run()


@blind.command()
@click.argument("video_path", type=str)
@click.option("--result_path", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--output_video_path", type=str, default="/tmp/blindpy.mp4")
@click.option("--targets", type=list, default=[0, 1, 2, 3])
@click.option("--show-once", is_flag=True, default=False)
@click.option("--with-audio", is_flag=True, default=True)
@click.option("--kernel-size", type=int, default=51)
def blur(
        video_path,
        result_path,
        output_video_path,
        targets,
        show_once,
        with_audio,
        kernel_size):
    drawer = GaussianDrawer(
            Drawer.Param(
                video_path=video_path,
                result_path=result_path,
                output_video_path=output_video_path,
                targets=targets,
                show_once=show_once,
                with_audio=with_audio,
                tmp_video_path="/tmp/tmp.mp4"),
            kernel_size)
    drawer.run()


@blind.command()
@click.argument("video_path", type=str)
@click.option("--result_path", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--output_video_path", type=str, default="/tmp/blindpy.mp4")
@click.option("--targets", type=list, default=[0, 1, 2, 3])
@click.option("--show-once", is_flag=True, default=False)
@click.option("--with-audio", is_flag=True, default=True)
@click.option("--scale", type=int, default=50)
def mosaic(
        video_path,
        result_path,
        output_video_path,
        targets,
        show_once,
        with_audio,
        scale):
    drawer = MosaicDrawer(
            Drawer.Param(
                video_path=video_path,
                result_path=result_path,
                output_video_path=output_video_path,
                targets=targets,
                show_once=show_once,
                with_audio=with_audio,
                tmp_video_path="/tmp/tmp.mp4"),
            scale)
    drawer.run()
