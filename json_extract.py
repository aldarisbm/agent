import os
import json
from dotenv import load_dotenv

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader

load_dotenv()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_name = os.getenv("MODEL_PATH")
grammar_file = os.getenv("GRAMMAR_FILE")

llm = LlamaCpp(
    model_path=model_name,
    temperature=0,
    use_mlock=True,
    grammar_path=grammar_file,
    n_batch=512,
    n_ctx=4096,
    n_gpu_layers=20,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

db = Chroma(collection_name="store", persist_directory="./chroma_db")



with open("./prompt_file") as prompt:
    pr = prompt.read()
    json_result = llm(prompt=pr)
    res = json_result.strip()
    res = json.loads(res)
    print(res)