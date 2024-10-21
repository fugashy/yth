import whisper as whisp
import json
import click
import ffmpeg


def _extract_audio_as_file(input_videopath, track_id, audio_filepath):
    probe = ffmpeg.probe(input_videopath)
    audio_streams = [
            stream for stream in probe["streams"]
            if stream["codec_type"] == "audio"
            ]
    (
        ffmpeg
        .input(input_videopath)
        .output(audio_filepath, map=f"0:a:{track_id}")
        .run()
    )



@click.group()
def whisper():
    pass


@whisper.command()
@click.argument("input_videopath", type=str)
@click.option("--track-id", type=int, default=1)
@click.option("--model-name", type=str, default="turbo")
@click.option("--use-previous", is_flag=True, default=False)
def transcribe(input_videopath, track_id, model_name, use_previous):
    audio_filepath = "/tmp/audio.mp3"
    _extract_audio_as_file(
            input_videopath, track_id, audio_filepath)

    model = whisp.load_model(model_name)
    result = model.transcribe(
            audio_filepath,
            verbose=True,
            fp16=False,
            language="ja")

    f = open('/tmp/transcription.txt', 'w', encoding='UTF-8')
    f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))
    f.close()





