import base64
import logging
import sqlite3
from typing import List, Tuple, Union, Dict
import os
import io
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import tiktoken

logger = logging.getLogger("lida")

SQLITE_MAGIC = b"SQLite format 3\x00"
SQLITE_EXTENSIONS = (".db", ".sqlite", ".sqlite3")
GZIP_MAGIC = b"\x1f\x8b"
TAR_MAGIC_OFFSET = 257
TAR_MAGIC = b"ustar"


def adapt_messages_for_provider(messages: List[Dict[str, str]], provider: str) -> List[Dict[str, str]]:
    """Pass-through retained for adapter call sites; LiteLLM proxy speaks OpenAI format."""
    return messages


def get_dirs(path: str) -> List[str]:
    return next(os.walk(path))[1]


def clean_column_name(col_name: str) -> str:
    """
    Clean a single column name by replacing special characters and spaces with underscores.

    :param col_name: The name of the column to be cleaned.
    :return: A sanitized string valid as a column name.
    """
    return re.sub(r"[^0-9a-zA-Z_]", "_", col_name)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all column names in the given DataFrame.

    :param df: The DataFrame with possibly dirty column names.
    :return: A copy of the DataFrame with clean column names.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = [clean_column_name(col) for col in cleaned_df.columns]
    return cleaned_df


def read_dataframe(file_location: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Read a dataframe from a given file location.
    It samples down to 4500 rows if the data exceeds that limit and cleans column names.

    :param file_location: The path to the file containing the data.
    :param encoding: Encoding to use for the file reading.
    :return: A cleaned and sampled DataFrame.
    """
    file_extension = file_location.split(".")[-1]

    # Handle GZIP compressed files
    if file_extension == "gz":
        file_extension = file_location.split(".")[-2]

    read_funcs = {
        "json": lambda: pd.read_json(file_location, orient="records", encoding=encoding, compression="infer"),
        "csv": lambda: pd.read_csv(file_location, encoding=encoding, compression="infer"),
        "xls": lambda: pd.read_excel(file_location),
        "xlsx": lambda: pd.read_excel(file_location),
        "parquet": pd.read_parquet,
        "feather": pd.read_feather,
        "tsv": lambda: pd.read_csv(file_location, sep="	", encoding=encoding, compression="infer"),
    }

    if file_extension not in read_funcs:
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        df = read_funcs[file_extension]()
    except Exception as e:
        logger.error(f"Failed to read file: {file_location}. Error: {e}")
        raise

    # Clean column names
    df = clean_column_names(df)

    # Sample down to 4500 rows if necessary
    if len(df) > 4500:
        logger.info(f"Dataframe has more than 4500 rows ({len(df)}). Sampling down to 4500 rows.")
        df = df.sample(4500, random_state=42)

    return df


def is_sqlite_file(file_location: str) -> bool:
    """Detect SQLite databases via file magic; falls back to extension if unreadable."""
    try:
        with open(file_location, "rb") as f:
            return f.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return file_location.lower().endswith(SQLITE_EXTENSIONS)


def is_gzip_file(file_location: str) -> bool:
    """Detect gzip-wrapped uploads via the 0x1f 0x8b magic bytes."""
    try:
        with open(file_location, "rb") as f:
            return f.read(len(GZIP_MAGIC)) == GZIP_MAGIC
    except OSError:
        return False


def is_tar_archive(file_location: str) -> bool:
    """Detect POSIX tar archives via the 'ustar' magic at offset 257."""
    try:
        with open(file_location, "rb") as f:
            f.seek(TAR_MAGIC_OFFSET)
            return f.read(len(TAR_MAGIC)) == TAR_MAGIC
    except OSError:
        return False


def list_sqlite_tables(file_location: str) -> List[str]:
    """Return the user-defined table names in a SQLite database (sqlite_* internals excluded)."""
    with sqlite3.connect(f"file:{file_location}?mode=ro", uri=True) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [row[0] for row in cur.fetchall()]


def read_sqlite_tables(file_location: str) -> Dict[str, pd.DataFrame]:
    """Load every user table in a SQLite database into a dict of DataFrames (cleaned & sampled)."""
    tables: Dict[str, pd.DataFrame] = {}
    with sqlite3.connect(f"file:{file_location}?mode=ro", uri=True) as conn:
        for name in list_sqlite_tables(file_location):
            df = pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
            df = clean_column_names(df)
            if len(df) > 4500:
                logger.info(f"Sampling table '{name}' down from {len(df)} to 4500 rows.")
                df = df.sample(4500, random_state=42)
            tables[name] = df
    return tables


def file_to_df(file_location: str):
    """Get summary of data from file location"""
    file_name = file_location.split("/")[-1]
    df = None
    if "csv" in file_name:
        df = pd.read_csv(file_location)
    elif "xlsx" in file_name:
        df = pd.read_excel(file_location)
    elif "json" in file_name:
        df = pd.read_json(file_location, orient="records")
    elif "parquet" in file_name:
        df = pd.read_parquet(file_location)
    elif "feather" in file_name:
        df = pd.read_feather(file_location)

    return df


def plot_raster(rasters: Union[str, List[str]], figsize: Tuple[int, int] = (10, 10)):
    """
    Plot a series of base64-encoded raster images in a horizontal layout.

    Args:
        rasters: A single base64 string or a list of base64-encoded strings representing the images.
        figsize: A tuple indicating the size of the figure to display.
    """
    plt.figure(figsize=figsize)

    if isinstance(rasters, str):
        rasters = [rasters]

    images = []

    # Find the max height for resizing
    max_height = 0
    for raster in rasters:
        decoded_image = base64.b64decode(raster)
        image = plt.imread(io.BytesIO(decoded_image), format="PNG")

        max_height = max(max_height, image.shape[0])

    # Resize images to max_height while preserving the aspect ratio and alpha channel if it exists
    for raster in rasters:
        decoded_image = base64.b64decode(raster)
        image = plt.imread(io.BytesIO(decoded_image), format="PNG")

        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(max_height * aspect_ratio)
        image_resized = np.array([np.interp(np.linspace(0, len(row), new_width), np.arange(0, len(row)), row) for row in image])

        if image_resized.shape[2] == 4:  # If RGBA, preserve alpha channel
            alpha_channel = image_resized[:, :, 3:]
            # Drop the alpha for visualization
            image_resized = image_resized[:, :, :3]
            image_resized = np.clip(image_resized, 0, 1)
            image_resized = np.concatenate((image_resized, alpha_channel), axis=2)

        images.append(image_resized)

    # Concatenate images along the width
    concatenated_image = np.concatenate(images, axis=1)

    plt.imshow(concatenated_image)
    plt.axis("off")
    plt.show()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


def clean_code_snippet(code_string):
    # Extract code snippet using regex
    cleaned_snippet = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", code_string)

    if cleaned_snippet:
        cleaned_snippet = cleaned_snippet.group(1)
    else:
        cleaned_snippet = code_string

    # remove non-printable characters
    # cleaned_snippet = re.sub(r'[ - ]+', ' ', cleaned_snippet)

    return cleaned_snippet
