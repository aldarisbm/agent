import os
import json
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain

load_dotenv()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_name = os.getenv("MODEL_PATH")
grammar_file = os.getenv("GRAMMAR_FILE")
pdf_file = "./pdf_samples/A-Game-Of-Thrones-Book.pdf"

# -n 256 -c 2048  --mlock -ngl 20 --temp 0.8 --batch_size 512 \

llm = LlamaCpp(
    model_path=model_name,
    temperature=0,
    use_mlock=True,
    # grammar_path=grammar_file,
    n_batch=512,
    n_ctx=4096,
    n_gpu_layers=20,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

loader = PyPDFLoader(pdf_file)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64)
docs = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    collection_name="store",
    documents=docs,
    persist_directory="./chroma_db",
    embedding=embedding_function
)

with open("./prompt_samples/pf_rag") as prompt_file:
    t = prompt_file.read()
    prompt = PromptTemplate(
        template=t,
        input_variables=["context", "question"]
    )

    chain_type_kwargs = {
        "prompt": prompt
    }

    search_kwargs = {
        "top_k": 5,
    }

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs=search_kwargs,
        ),
        chain_type_kwargs=chain_type_kwargs,
    )

    query = "What's the name of Cersei Lannister's wolf?"
    result = qa(query)
    print("res", result)
    print(result["result"])