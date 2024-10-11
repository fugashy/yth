import click
import cv2

from .yolo import yolo
from .results import results
from .blind import blind


@click.group()
def blindpy():
    pass


def entry_point():
    commands = [
            yolo,
            results,
            blind,
            ]
    [
        blindpy.add_command(c)
        for c in commands
        ]
    blindpy()
