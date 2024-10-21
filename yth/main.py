import click

from .yolo import yolo
from .results import results
from .blind import blind
from .whisper import whisper


@click.group()
def blindpy():
    pass


def entry_point():
    commands = [
            yolo,
            results,
            blind,
            whisper,
            ]
    [
        blindpy.add_command(c)
        for c in commands
        ]
    blindpy()
