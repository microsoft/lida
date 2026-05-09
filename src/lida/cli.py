import typer
import uvicorn
import os
from typing_extensions import Annotated

from lida.components.litellm_generator import DEFAULT_BASE_URL, DEFAULT_MODEL

app = typer.Typer()


@app.command()
def ui(
    host: str = "127.0.0.1",
    port: int = 8081,
    workers: int = 1,
    reload: Annotated[bool, typer.Option("--reload")] = True,
    docs: bool = False,
):
    """
    Launch the lida UI. The LLM endpoint is a LiteLLM proxy configured via
    LITELLM_BASE_URL, LITELLM_API_KEY, and LITELLM_MODEL environment variables.
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
    """Print the active LiteLLM endpoint and default model."""
    base_url = os.environ.get("LITELLM_BASE_URL") or DEFAULT_BASE_URL
    model = os.environ.get("LITELLM_MODEL") or DEFAULT_MODEL
    print(f"Provider: litellm ({base_url})")
    print(f"  - {model}")


def run():
    app()


if __name__ == "__main__":
    app()
