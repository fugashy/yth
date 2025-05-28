import click

from .yolo import yolo
from .results import results
from .blind import blind
from .whisper import whisper
from .edit import edit
from .servers import servers


@click.group()
def blindpy():
    pass


def entry_point():
    commands = [
            yolo,
            results,
            blind,
            whisper,
            edit,
            servers,
            ]
    [
        blindpy.add_command(c)
        for c in commands
        ]
    blindpy()
