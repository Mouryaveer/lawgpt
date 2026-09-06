import langchain_groq, langchain_community, langchainhub, langchain, os
import json
import logging
import time
from typing import Callable
from functools import wraps
from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
load_dotenv()

# Get configuration from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in .env file.")

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
USER_AGENT = os.getenv("USER_AGENT", "MyLangChainScraper/mk2")

# Set environment variables for LangChain
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["USER_AGENT"] = USER_AGENT
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Get Pinecone and model configuration
index_name = os.getenv("PINECONE_INDEX_NAME", "caus-legal-vdb")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "new-data-v2")

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-large-v2")
embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
embedding_device = os.getenv("EMBEDDING_DEVICE", "cuda")
deepinfra_token = os.getenv("DEEPINFRA_TOKEN")
groq_model_name = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
retriever_top_k = int(os.getenv("RETRIEVER_TOP_K", "6"))
embedding_provider = "unknown"


class E5Embeddings:
    def __init__(self, model_name: str, batch_size: int):
        try:
            self.base = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': embedding_device},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': batch_size},
            )
            logger.info("Loaded embedding model on device=%s", embedding_device)
        except Exception as exc:
            if embedding_device == 'cpu':
                raise
            logger.warning("Embedding init failed on device=%s (%s). Falling back to cpu.", embedding_device, exc)
            self.base = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': batch_size},
            )
            logger.info("Loaded embedding model on device=cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # e5 models expect passages to be prefixed for best retrieval quality.
        return self.base.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(f"query: {text}")


class DeepInfraE5Embeddings:
    def __init__(self, model_name: str, api_token: str, local_factory):
        from deepinfra_embeddings import DeepInfraEmbeddings

        self.base = DeepInfraEmbeddings(model_name=model_name, api_token=api_token)
        self.local_factory = local_factory
        self.local_embedder = None

    def _fallback(self, exc: Exception):
        if self.local_embedder is None:
            logger.warning("DeepInfra embedding request failed (%s). Falling back to local model.", exc)
            self.local_embedder = self.local_factory()
        return self.local_embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Keep prefixes aligned with local e5 usage for consistent retrieval behavior.
        try:
            return self.base.embed_documents([f"passage: {text}" for text in texts])
        except Exception as exc:
            return self._fallback(exc).embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.base.embed_query(f"query: {text}")
        except Exception as exc:
            return self._fallback(exc).embed_query(text)


def init_embeddings():
    global embedding_provider
    if deepinfra_token:
        try:
            logger.info("DEEPINFRA_TOKEN detected. Validating DeepInfra embeddings API with model=%s", embedding_model_name)
            from deepinfra_embeddings import DeepInfraEmbeddings
            deep_base = DeepInfraEmbeddings(model_name=embedding_model_name, api_token=deepinfra_token)
            deep_base.embed_query("query: test connection")
            logger.info("DeepInfra API authentication successful!")
            embedding_provider = "deepinfra"
            return DeepInfraE5Embeddings(
                model_name=embedding_model_name,
                api_token=deepinfra_token,
                local_factory=lambda: E5Embeddings(
                    model_name=embedding_model_name,
                    batch_size=embedding_batch_size,
                ),
            )
        except Exception as exc:
            logger.warning("DeepInfra API check failed (%s). Pre-loading local HuggingFace embeddings at startup.", exc)

    logger.info("Using local HuggingFace embeddings with model=%s", embedding_model_name)
    local_emb = E5Embeddings(model_name=embedding_model_name, batch_size=embedding_batch_size)
    embedding_provider = "local-e5"
    return local_emb

embedz = init_embeddings()



# # Load documents from JSONL file
# docs = []
# with open("chunked_documents.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         item = json.loads(line)
#         docs.append(
#             Document(
#                 page_content=item["text"], 
#                 metadata={"id": item["id"]}
#             )
#         )

# print(f"Loaded {len(docs)} documents")

# Initialize Pinecone (if needed)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Connect to existing vector store (no uploading)
print("Connecting to existing vector store...")
try:
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embedz,
        text_key="text",
        namespace=pinecone_namespace or None,
    )
    logger.info(
        "Successfully connected to Pinecone index=%s namespace=%s",
        index_name,
        pinecone_namespace or "<default>",
    )
except Exception as e:
    logger.error(f"Failed to connect to Pinecone: {str(e)}", exc_info=True)
    raise

retriever = vector_store.as_retriever(search_kwargs={"k": retriever_top_k})

from langchain import hub
prompt=hub.pull("therager4000/legal_llm_prompt_indian")


from langchain_groq import ChatGroq

def get_groq_llm(model_name: str):
    return ChatGroq(model=model_name, verbose=True)

try:
    llm = get_groq_llm(groq_model_name)
except Exception as exc:
    logger.warning("Groq init failed for model %s (%s). Falling back to openai/gpt-oss-20b", groq_model_name, exc)
    groq_model_name = "openai/gpt-oss-20b"
    llm = get_groq_llm(groq_model_name)



from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain=({"context":  retriever | format_docs, "question": RunnablePassthrough()}
          | prompt
          | llm
          | StrOutputParser())


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator that retries a function with exponential backoff on exception.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"RAG invocation attempt {attempt + 1}/{max_retries + 1}")
                    result = func(*args, **kwargs)
                    logger.info("RAG invocation successful")
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"RAG invocation failed (attempt {attempt + 1}): {str(e)}")
                        logger.info(f"Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"RAG invocation failed after {max_retries + 1} attempts: {str(e)}")
            
            # If all retries exhausted, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


@retry_on_exception(max_retries=3, delay=1.0, backoff=1.5)
def ragu(query: str) -> str:
    """
    RAG Generation - Process a query through the RAG chain.
    """
    global rag_chain, llm, groq_model_name
    try:
        logger.info(f"Starting RAG processing for query: {query[:100]}...")
        response = rag_chain.invoke(query)
        logger.info("RAG processing completed successfully")
        return response
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Error in RAG processing: {err_msg}", exc_info=True)
        if "404" in err_msg or "model_not_found" in err_msg or "does not exist" in err_msg:
            fallback_model = "openai/gpt-oss-20b"
            if groq_model_name != fallback_model:
                logger.warning("Switching Groq model from %s to fallback %s", groq_model_name, fallback_model)
                groq_model_name = fallback_model
                llm = ChatGroq(model=groq_model_name, verbose=True)
                rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
                             | prompt
                             | llm
                             | StrOutputParser())
                return rag_chain.invoke(query)
        raise



