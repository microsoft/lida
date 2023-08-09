import json
import logging
from typing import Union
import pandas as pd
from lida.utils import clean_code_snippet, read_dataframe
from lida.datamodel import TextGenerationConfig


system_prompt = """
You are a an experienced data analyst that can annotate datasets. Your instructions are as follows:
i) ALWAYS generate the name of the dataset and the dataset_description
ii) ALWAYS generate a field description.
iv.) ALWAYS generate a semantic_type (a single word) for each field given its values e.g. company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc
You return an updated JSON dictionary without any preamble or explanation.
"""

logger = logging.getLogger(__name__)


class Summarizer():
    def __init__(self, text_gen) -> None:
        self.summary = None
        self.text_gen = text_gen

    def check_type(self, dtype: str, value):
        """Cast value to right type to ensure it is JSON serializable"""
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    def get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
        """Get properties of each column in a pandas DataFrame"""
        properties_list = []
        for column in df.columns:
            dtype = df[column].dtype
            properties = {}
            if dtype in [int, float, complex]:
                properties["dtype"] = "number"
                properties["std"] = self.check_type(dtype, df[column].std())
                properties["min"] = self.check_type(dtype, df[column].min())
                properties["max"] = self.check_type(dtype, df[column].max())

            elif dtype == bool:
                properties["dtype"] = "boolean"
            elif dtype == object:
                # Check if the string column can be cast to a valid datetime
                try:
                    pd.to_datetime(df[column], errors='raise')
                    properties["dtype"] = "date"
                except ValueError:
                    # Check if the string column has a limited number of values
                    if df[column].nunique() / len(df[column]) < 0.5:
                        properties["dtype"] = "category"
                    else:
                        properties["dtype"] = "string"
            elif pd.api.types.is_categorical_dtype(df[column]):
                properties["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "date"
            else:
                properties["dtype"] = str(dtype)

            # add min max if dtype is date
            if properties["dtype"] == "date":
                try:
                    properties["min"] = df[column].min()
                    properties["max"] = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors='coerce')
                    properties["min"] = cast_date_col.min()
                    properties["max"] = cast_date_col.max()
            # Add additional properties to the output dictionary
            nunique = df[column].nunique()
            if "samples" not in properties:
                non_null_values = df[column][df[column].notnull()].unique()
                n_samples = min(n_samples, len(non_null_values))
                samples = pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
                properties["samples"] = samples
            properties["num_unique_values"] = nunique
            properties["semantic_type"] = ""
            properties["description"] = ""
            properties_list.append({"column": column, "properties": properties})

        return properties_list

    def encrich(self, result: dict, textgen_config: TextGenerationConfig) -> dict:
        """Enrich the data summary with descriptions"""
        logger.info(f"Enriching the data summary with descriptions")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"""
        Annotate the dictionary below. Only return a JSON object.
        {result}
        """},
        ]

        response = self.text_gen.generate(messages=messages, config=textgen_config)
        try:
            json_string = clean_code_snippet(response.text[0]["content"])
            result = json.loads(json_string)
        except json.decoder.JSONDecodeError:
            logger.info(f"Error decoding JSON: {response.text[0]['content']}")
            print(response.text[0]["content"])
            raise ValueError(
                f"The model did not return a valid JSON object while attempting to summarize the data. Please try again.",
                response.usage)
        return result

    def summarize(self, data: Union[pd.DataFrame, str],
                  file_name="", n_samples: int = 3, enrich: bool = True,
                  textgen_config=TextGenerationConfig(n=1)) -> dict:
        """Summarize data from a pandas DataFrame or a file location"""

        # if data is a file path, read it into a pandas DataFrame, set file_name to the file name
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data)
        data_properties = self.get_column_properties(data, n_samples)
        result = {
            "name": file_name,
            "file_name": file_name,
            "dataset_description": "",
            "fields": data_properties,
        }
        if enrich:
            result = self.encrich(result, textgen_config)
        result["field_names"] = data.columns.tolist()
        result["file_name"] = file_name

        return result
