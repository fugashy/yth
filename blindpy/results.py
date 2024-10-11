import click


@click.group()
def results():
    pass


@results.command()
@click.option("--input_filepath", type=str, default="/tmp/blindpy-yolo-results.txt")
def inspect(input_filepath):
    pass


@results.command()
@click.option("--input_filepath", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--output_filepath", type=str, default="/tmp/blindpy-yolo-results-modified.txt")
def clean(input_filepath, output_filepath):
    pass
