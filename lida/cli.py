import typer
import uvicorn
import os
from typing_extensions import Annotated

# from lida.web.backend.app import launch

app = typer.Typer()


@app.command()
def ui(host: Annotated[str, typer.Argument(help="The host to run the server on.")] = "127.0.0.1",
       port: Annotated[int, typer.Argument(help="The port to run the server on.")] = 8081,
       workers: Annotated[int, typer.Argument(help="Number of worker processes.")] = 1,
       reload: Annotated[bool, typer.Argument(help="Enable auto-reload.")] = True,
       docs: Annotated[bool, typer.Argument(help="Enable docs.")] = False,
       code_eval: Annotated[str, typer.Argument(help="Enable code evaluation.")] = "0"
       ):
    """
    Launch the lida UI.Pass in parameters host, port, workers, and reload to override the default values.
    """
# def ui(host: str = "127.0.0.1", port: int = 8081, workers: int = 1,
#         reload: bool = True, docs: bool = False, code_eval: str = 0):
#     """
# Launch the lida UI.Pass in parameters host, port, workers, and reload to
# override the default values.

#     """

    os.environ["LIDA_ALLOW_CODE_EVAL"] = code_eval
    os.environ["LIDA_API_DOCS"] = str(docs)

    uvicorn.run(
        "lida.web.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def list():
    print("list")


def run():
    app()


if __name__ == "__main__":
    app()
