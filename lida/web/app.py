import json
import os
import logging
import requests
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import traceback

from llmx import llm, providers
from ..datamodel import GoalWebRequest, TextGenerationConfig, UploadUrl, VisualizeEditWebRequest, VisualizeEvalWebRequest, VisualizeExplainWebRequest, VisualizeRecommendRequest, VisualizeRepairWebRequest, VisualizeWebRequest
from ..components import Manager


# instantiate model and generator
textgen = llm()
logger = logging.getLogger(__name__)
api_docs = os.environ.get("LIDA_API_DOCS", "False") == "True"

print("Api docs ... ", api_docs)

lida = Manager(text_gen=textgen)
app = FastAPI()
# allow cross origin requests for testing on localhost:800* ports only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api = FastAPI(root_path="/api", docs_url="/docs" if api_docs else None, redoc_url=None)
app.mount("/api", api)


root_file_path = os.path.dirname(os.path.abspath(__file__))
static_folder_root = os.path.join(root_file_path, "ui")
files_static_root = os.path.join(root_file_path, "files/")
data_folder = os.path.join(root_file_path, "files/data")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(files_static_root, exist_ok=True)
os.makedirs(static_folder_root, exist_ok=True)


# mount lida front end UI files
app.mount("/", StaticFiles(directory=static_folder_root, html=True), name="ui")
api.mount("/files", StaticFiles(directory=files_static_root, html=True), name="files")


# def check_model

@api.post("/visualize")
async def visualize_data(req: VisualizeWebRequest) -> dict:
    """Generate goals given a dataset summary"""
    try:
        # print(req.textgen_config)
        code_specs = lida.visualize(
            summary=req.summary,
            goal=req.goal,
            textgen_config=req.textgen_config if req.textgen_config else TextGenerationConfig(),
            library=req.library)
        charts = lida.execute(
            code_specs=code_specs,
            data=lida.data,
            summary=req.summary,
            library=req.library,
            return_error=True)
        print("found charts: ", len(charts), " for goal: ")
        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {"status": True, "charts": charts,
                "message": "Successfully generated charts."}

    except Exception as exception_error:
        logger.error(f"Error generating visualization goals: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization goals. {str(exception_error)}"}


@api.post("/visualize/edit")
async def edit_visualization(req: VisualizeEditWebRequest) -> dict:
    """Given a visualization code, and a goal, generate a new visualization"""
    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        code_specs = lida.edit(
            code=req.code,
            summary=req.summary,
            instructions=req.instructions,
            textgen_config=textgen_config,
            library=req.library)

        charts = lida.execute(
            code_specs=code_specs,
            data=lida.data,
            summary=req.summary,
            library=req.library,
            return_error=True)
        # charts = [asdict(chart) for chart in charts]
        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {"status": True, "charts": charts,
                "message": f"Successfully edited charts."}

    except Exception as exception_error:
        logger.error(f"Error generating visualization edits: {str(exception_error)}")
        print(traceback.print_exc())
        return {"status": False,
                "message": f"Error generating visualization edits."}


@api.post("/visualize/repair")
async def repair_visualization(req: VisualizeRepairWebRequest) -> dict:
    """ Given a visualization goal and some feedback, generate a new visualization that addresses the feedback"""

    try:

        code_specs = lida.repair(
            code=req.code,
            feedback=req.feedback,
            goal=req.goal,
            summary=req.summary,
            textgen_config=req.textgen_config if req.textgen_config else TextGenerationConfig(),
            library=req.library)

        charts = lida.execute(
            code_specs=code_specs,
            data=lida.data,
            summary=req.summary,
            library=req.library,
            return_error=True)
        # charts = [asdict(chart) for chart in charts]
        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {"status": True, "charts": charts,
                "message": "Successfully generated chart repairs"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization repairs: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization repairs."}


@api.post("/visualize/explain")
async def explain_visualization(req: VisualizeExplainWebRequest) -> dict:
    """Given a visualization code, provide an explanation of the code"""
    textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig(
        n=1,
        temperature=0)
    print("textgen_config: ", req.textgen_config)
    try:
        explanations = lida.explain(
            code=req.code,
            textgen_config=textgen_config,
            library=req.library)
        return {"status": True, "explanations": explanations[0],
                "message": "Successfully generated explanations"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization explanation: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization explanation."}


@api.post("/visualize/evaluate")
async def evaluate_visualization(req: VisualizeEvalWebRequest) -> dict:
    """Given a visualization code, provide an evaluation of the code"""

    try:
        evaluations = lida.evaluate(
            code=req.code,
            goal=req.goal,
            textgen_config=req.textgen_config if req.textgen_config else TextGenerationConfig(
                n=1,
                temperature=0),
            library=req.library)
        try:
            evaluations = json.loads(evaluations[0])
        except BaseException:
            print("Error parsing evaluation JSON data", evaluations)
            return {"status": False, "message": "Error parsing evaluation JSON data"}
        return {"status": True, "evaluations": evaluations,
                "message": "Successfully generated evaluation"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization evaluation: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization evaluation."}


@api.post("/visualize/recommend")
async def recommend_visualization(req: VisualizeRecommendRequest) -> dict:
    """Given a dataset summary, generate a visualization recommendations"""

    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        code_specs = lida.recommend(
            summary=req.summary,
            code=req.code,
            textgen_config=textgen_config,
            library=req.library)
        charts = lida.execute(
            code_specs=code_specs,
            data=lida.data,
            summary=req.summary,
            library=req.library,
            return_error=True)
        # charts = [asdict(chart) for chart in charts]
        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {"status": True, "charts": charts,
                "message": "Successfully generated chart recommendation"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization recommendation: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization recommendation."}


@api.post("/text/generate")
async def generate_text(textgen_config: TextGenerationConfig) -> dict:
    """Generate text given some prompt"""

    try:
        completions = textgen.generate(textgen_config)
        return {"status": True, "completions": completions.text}
    except Exception as exception_error:
        logger.error(f"Error generating text: {str(exception_error)}")
        return {"status": False, "message": f"Error generating text."}


@api.post("/goal")
async def generate_goal(req: GoalWebRequest) -> dict:
    """Generate goals given a dataset summary"""
    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        goals = lida.goals(req.summary, n=req.n, textgen_config=textgen_config)
        return {"status": True, "data": goals,
                "message": f"Successfully generated {len(goals)} goals"}
    except Exception as exception_error:
        logger.error(f"Error generating goals: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization goals. {exception_error}"}


@api.post("/summarize")
async def upload_file(file: UploadFile):
    """ Upload a file and return a summary of the data """
    # allow csv, excel, json
    allowed_types = ["text/csv", "application/vnd.ms-excel", "application/json"]

    # print("file: ", file)
    # check file type
    if file.content_type not in allowed_types:
        return {"status": False,
                "message": f"Uploaded file type ({file.content_type}) not allowed. Allowed types are: csv, excel, json"}

    try:

        # save file to files folder
        file_location = os.path.join(data_folder, file.filename)
        # open file without deleting existing contents
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # summarize
        textgen_config = TextGenerationConfig(n=1, temperature=0)
        summary = lida.summarize(
            data=file_location,
            file_name=file.filename,
            enrich=True,
            textgen_config=textgen_config)
        return {"status": True, "summary": summary, "data_filename": file.filename}
    except Exception as exception_error:
        logger.error(f"Error processing file: {str(exception_error)}")
        return {"status": False, "message": f"Error processing file."}


# upload via url
@api.post("/summarize/url")
def upload_file_via_url(payload: UploadUrl) -> dict:
    """ Upload a file from a url and return a summary of the data """
    url = payload.url
    file_name = url.split("/")[-1]
    file_location = os.path.join(data_folder, file_name)

    # download file
    url_response = requests.get(url, allow_redirects=True, timeout=1000)
    open(file_location, "wb").write(url_response.content)
    try:
        textgen_config = TextGenerationConfig(n=1, temperature=0)
        summary = lida.summarize(
            data=file_location,
            file_name=file_name,
            textgen_config=textgen_config)
        return {"status": True, "summary": summary, "data_filename": file_name}
    except Exception as exception_error:
        # traceback.print_exc()
        logger.error(f"Error processing file: {str(exception_error)}")
        return {"status": False, "message": f"Error processing file."}

# list supported models


@api.get("/models")
def list_models() -> dict:
    return {"status": True, "data": providers, "message": "Successfully listed models"}
