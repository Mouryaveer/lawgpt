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
embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
deepinfra_token = os.getenv("DEEPINFRA_TOKEN")
groq_model_name = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
retriever_top_k = int(os.getenv("RETRIEVER_TOP_K", "6"))
embedding_provider = "unknown"
allow_local_embedding_fallback = os.getenv("ALLOW_LOCAL_EMBEDDING_FALLBACK", "true").lower() in {"1", "true", "yes"}
rag_fallback_enabled = os.getenv("RAG_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes"}

# ── Lazy globals ────────────────────────────────────────────────────────────
# Nothing heavy is initialised at import time. Everything is created on the
# first call to get_rag_chain(), which is invoked from ragu() on the first
# real request — by which time uvicorn has already bound the port.
_embedz = None
_vector_store = None
_retriever = None
_llm = None
_prompt = None
_rag_chain = None
_initialized = False
# ────────────────────────────────────────────────────────────────────────────


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
        if not allow_local_embedding_fallback:
            raise RuntimeError("DeepInfra embeddings are unavailable and local fallback is disabled.") from exc
        if self.local_embedder is None:
            logger.warning("DeepInfra embedding request failed (%s). Falling back to local model.", exc)
            self.local_embedder = self.local_factory()
        return self.local_embedder
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.base.embed_documents([f"passage: {text}" for text in texts])
        except Exception as exc:
            return self._fallback(exc).embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.base.embed_query(f"query: {text}")
        except Exception as exc:
            return self._fallback(exc).embed_query(text)


def _init_embeddings():
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
                )
            )
        except Exception as exc:
            logger.warning("DeepInfra API check failed (%s).", exc)
            if not allow_local_embedding_fallback:
                raise RuntimeError("DeepInfra embeddings are unavailable and local fallback is disabled.") from exc

    if not allow_local_embedding_fallback:
        raise RuntimeError("No usable embedding provider is configured.")

    logger.info("Using local HuggingFace embeddings with model=%s", embedding_model_name)
    local_emb = E5Embeddings(model_name=embedding_model_name, batch_size=embedding_batch_size)
    embedding_provider = "local-e5"
    return local_emb


def _build_llm_only_chain():
    """Build a degraded but responsive chain when retrieval is unavailable."""
    global _prompt, _llm, _rag_chain, _initialized, groq_model_name, embedding_provider
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_groq import ChatGroq

    _prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Turn2Law, an Indian legal information assistant. "
         "Answer clearly and cautiously. State that this is general legal "
         "information, not legal advice, and do not invent citations."),
        ("human", "Question: {question}"),
    ])
    try:
        _llm = ChatGroq(model=groq_model_name, verbose=True)
    except Exception as exc:
        logger.warning("Groq init failed for model %s (%s). Using fallback model.", groq_model_name, exc)
        groq_model_name = "openai/gpt-oss-20b"
        _llm = ChatGroq(model=groq_model_name, verbose=True)

    _rag_chain = (
        {"question": RunnablePassthrough()}
        | _prompt
        | _llm
        | StrOutputParser()
    )
    embedding_provider = "llm-only-fallback"
    _initialized = True
    logger.warning("Serving with LLM-only fallback; Pinecone retrieval is unavailable.")
    return _rag_chain


def get_rag_chain():
    """Lazily initialise the RAG stack, with a fast degraded fallback."""
    global _embedz, _vector_store, _retriever, _llm, _prompt, _rag_chain
    global _initialized, groq_model_name

    if _initialized:
        return _rag_chain

    logger.info("First request received — initialising RAG stack...")
    try:
        _embedz = _init_embeddings()
        Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        logger.info("Connecting to existing vector store...")
        _vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=_embedz,
            text_key="text",
            namespace=pinecone_namespace or None,
        )
        logger.info(
            "Successfully connected to Pinecone index=%s namespace=%s",
            index_name,
            pinecone_namespace or "<default>",
        )

        _retriever = _vector_store.as_retriever(search_kwargs={"k": retriever_top_k})
        from langchain import hub
        _prompt = hub.pull("therager4000/legal_llm_prompt_indian")
        from langchain_groq import ChatGroq
        try:
            _llm = ChatGroq(model=groq_model_name, verbose=True)
        except Exception as exc:
            logger.warning("Groq init failed for model %s (%s). Falling back to openai/gpt-oss-20b", groq_model_name, exc)
            groq_model_name = "openai/gpt-oss-20b"
            _llm = ChatGroq(model=groq_model_name, verbose=True)

        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)
        _rag_chain = (
            {"context": _retriever | format_docs, "question": RunnablePassthrough()}
            | _prompt
            | _llm
            | StrOutputParser()
        )
        _initialized = True
        logger.info("RAG stack initialised successfully.")
        return _rag_chain
    except Exception as exc:
        logger.error("RAG stack unavailable: %s", exc, exc_info=True)
        if rag_fallback_enabled:
            return _build_llm_only_chain()
        raise

def retry_on_exception(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
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
            raise last_exception
        return wrapper
    return decorator


@retry_on_exception(max_retries=1, delay=0.5, backoff=1.5)
def ragu(query: str) -> str:
    """
    RAG Generation — process a query through the RAG chain.
    Initialises the stack on first call if not already done.
    """
    global _rag_chain, _llm, groq_model_name

    chain = get_rag_chain()

    try:
        logger.info(f"Starting RAG processing for query: {query[:100]}...")
        response = chain.invoke(query)
        logger.info("RAG processing completed successfully")
        return response
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Error in RAG processing: {err_msg}", exc_info=True)
        if "404" in err_msg or "model_not_found" in err_msg or "does not exist" in err_msg:
            fallback_model = "openai/gpt-oss-20b"
            if groq_model_name != fallback_model:
                from langchain_groq import ChatGroq
                from langchain_core.runnables import RunnablePassthrough
                from langchain_core.output_parsers import StrOutputParser

                def format_docs(docs):
                    return "\n".join(doc.page_content for doc in docs)

                logger.warning("Switching Groq model from %s to fallback %s", groq_model_name, fallback_model)
                groq_model_name = fallback_model
                _llm = ChatGroq(model=groq_model_name, verbose=True)
                _rag_chain = (
                    {"context": _retriever | format_docs, "question": RunnablePassthrough()}
                    | _prompt
                    | _llm
                    | StrOutputParser()
                )
                return _rag_chain.invoke(query)
        raise
