"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Napari Travali2."""


if __name__ == "__main__":
    main(prog_name="napari-travali2")  # pragma: no cover
