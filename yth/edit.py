import click
import ffmpeg
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pydub import AudioSegment, silence
import os
import tqdm


@click.group()
def edit():
    pass


def _extract_audio_track(
        input_video_path,
        audio_track_num,
        output_audio_path):
    # 指定した音声トラックを抽出（ここで無音判定を行うトラック）
    (
        ffmpeg
        .input(input_video_path)
        .output(
            output_audio_path,
            map=f'0:a:{audio_track_num}',
            acodec='pcm_s16le')
        .run()
    )


def _get_silence_intervals(
        audio_path,
        min_silence_len=1000,
        silence_thresh=-40):
    # pydubで音声を読み込み、無音区間を取得
    sound = AudioSegment.from_wav(audio_path)
    silent_ranges = silence.detect_silence(
            sound,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh)

    # 無音でない区間を取得（ミリ秒から秒に変換）
    non_silent_ranges = []
    if silent_ranges:
        non_silent_ranges = [
                (silent_ranges[i-1][1] / 1000, silent_ranges[i][0] / 1000)
                for i in range(1, len(silent_ranges))]
        # 最初と最後の音声区間を追加
        if silent_ranges[0][0] > 0:
            non_silent_ranges.insert(0, (0, silent_ranges[0][0] / 1000))
        if silent_ranges[-1][1] < sound.duration_seconds * 1000:
            non_silent_ranges.append(
                    (silent_ranges[-1][1] / 1000, sound.duration_seconds))
    else:
        non_silent_ranges = [(0, sound.duration_seconds)]  # 無音区間がない場合

    return non_silent_ranges


def _trim_video_with_all_tracks(input_video_path, output_video_path, non_silent_ranges):
    # 入力ファイルを取得
    input_stream = ffmpeg.input(input_video_path)
    
    # トリミングされたクリップを保持するリスト
    video_clips = []
    audio_clips = [[] for _ in range(len(ffmpeg.probe(input_video_path)['streams']) - 1)]  # 各オーディオトラックごとにリストを保持
    
    for start, end in non_silent_ranges:
        # 映像トラックをトリミング
        video_trimmed = input_stream.trim(start=start, end=end).setpts('PTS-STARTPTS')
        video_clips.append(video_trimmed)

        # 各オーディオトラックをトリミング
        for i in range(len(audio_clips)):
            audio_trimmed = input_stream[f'a:{i}'].filter_('atrim', start=start, end=end).filter_('asetpts', 'PTS-STARTPTS')
            audio_clips[i].append(audio_trimmed)

    # 映像クリップを結合
    video_concat = ffmpeg.concat(*video_clips, v=1, a=0).node

    # 各オーディオトラックを結合
    audio_concats = []
    for audio_clip in audio_clips:
        audio_concats.append(ffmpeg.concat(*audio_clip, v=0, a=1).node)

    # 出力ファイルに全てのトラックを反映
    output = ffmpeg.output(
            video_concat[0],
            *[
                audio_concat[1]
                for audio_concat in audio_concats],
            output_video_path,
            vcodec='h264_videotoolbox')
    ffmpeg.run(output)



@edit.command()
@click.argument("input_video_path", type=str)
@click.option("--output-video-path", type=str, default="/tmp/video.mp4")
@click.option("--audio-track-num", type=int, default=1)
@click.option("--min-silence-len", type=int, default=1000)
@click.option("--silence-thresh", type=int, default=-40)
def filter_silence(
        input_video_path,
        output_video_path,
        audio_track_num=0,
        min_silence_len=1000,
        silence_thresh=-40):
    # 一時的に保存する音声ファイルのパス
    audio_path = f"/tmp/temp_audio_track_{audio_track_num}.wav"

    # 指定されたトラックの音声を抽出
    _extract_audio_track(input_video_path, audio_track_num, audio_path)

    # 無音判定に基づいて無音でない区間を取得
    non_silent_ranges = _get_silence_intervals(audio_path, min_silence_len, silence_thresh)
    print(non_silent_ranges)

    # 無音でない区間で、全てのトラックをトリミング
    _trim_video_with_all_tracks(input_video_path, output_video_path, non_silent_ranges)

    # 一時ファイルの削除
    os.remove(audio_path)
