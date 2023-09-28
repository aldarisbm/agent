import os
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

load_dotenv()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_name = os.getenv("MODEL_PATH")
grammar_file = os.getenv("GRAMMAR_FILE")
pdf_file = "doc_samples/2304.12244.pdf"
store_name = os.path.basename(pdf_file)

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
text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=32)
docs = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="nq-distilbert-base-v1")

# db = Chroma.from_documents(
#     collection_name="store",
#     documents=docs,
#     persist_directory="./chroma_db",
#     embedding=embedding_function
# )
db = Chroma(collection_name="store", persist_directory="./chroma_db", embedding_function=embedding_function)

query = "This paper is about"
returned_docs = db.similarity_search(query, k=5)
for index, doc in enumerate(returned_docs):
    print(index+1, ".-", doc.page_content)
    print()
    print()


# with open("./prompt_samples/pf_rag") as prompt_file:
#     t = prompt_file.read()
#     prompt = PromptTemplate(
#         template=t,
#         input_variables=["context", "question"]
#     )
#
#     chain_type_kwargs = {
#         "prompt": prompt
#     }
#
#     search_kwargs = {
#         "k": 10,
#     }
#
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=db.as_retriever(
#             search_kwargs=search_kwargs,
#         ),
#         chain_type_kwargs=chain_type_kwargs,
#     )
#
#     query = "What is the name of the half-blood prince?"
#     result = qa(query)
#     print("res", result)