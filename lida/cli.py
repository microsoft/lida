import typer
import uvicorn
import os
from typing_extensions import Annotated
from llmx import providers

# from lida.web.backend.app import launch

app = typer.Typer()


@app.command()
def ui(host: str = "127.0.0.1",
       port: int = 8081,
       workers: int = 1,
       reload: Annotated[bool, typer.Option("--reload")] = True,
       docs: bool = False):
    """
    Launch the lida .Pass in parameters host, port, workers, and reload to override the default values.
    """

    os.environ["LIDA_API_DOCS"] = str(docs)

    uvicorn.run(
        "lida.web.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def models():
    print("A list of supported providers:")
    for provider in providers.items():
        print(f"Provider: {provider[1]['name']}")
        for model in provider[1]["models"]:
            print(f"  - {model['name']}")


def run():
    app()


if __name__ == "__main__":
    app()
