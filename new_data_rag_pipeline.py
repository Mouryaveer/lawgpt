import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


class UnifiedLegalChunkBuilder:
    """Build a unified chunked corpus from acts, constitution, and landmark cases."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, tokenizer: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = None
        try:
            import tiktoken

            self.encoding = tiktoken.get_encoding(tokenizer)
        except ImportError:
            logger.warning(
                "tiktoken is not installed. Falling back to word-based chunking. "
                "Install project dependencies for token-accurate chunks."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.strip() for line in text.split("\n"))
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        chunks: List[str] = []
        step = self.chunk_size - self.chunk_overlap

        if self.encoding is not None:
            tokens = self.encoding.encode(text)
            if not tokens:
                return []

            if len(tokens) <= self.chunk_size:
                return [text]

            for start in range(0, len(tokens), step):
                end = start + self.chunk_size
                chunk_text = self.encoding.decode(tokens[start:end]).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                if end >= len(tokens):
                    break
            return chunks

        words = text.split()
        if not words:
            return []

        if len(words) <= self.chunk_size:
            return [text]

        for start in range(0, len(words), step):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(words):
                break

        return chunks

    def _build_chunk_records(self, raw_docs: Iterable[RawDocument]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        for raw_doc in raw_docs:
            chunks = self._chunk_text(raw_doc.text)
            total = len(chunks)
            for idx, chunk in enumerate(chunks, start=1):
                record = {
                    "id": f"{raw_doc.doc_id}_c{idx}",
                    "text": chunk,
                    "metadata": {
                        **raw_doc.metadata,
                        "chunk_index": idx,
                        "total_chunks_in_document": total,
                    },
                }
                records.append(record)

        return records

    def load_acts(self, file_path: str) -> List[RawDocument]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs: List[RawDocument] = []

        for act_id, act in data.items():
            act_title = (act.get("title") or "Untitled Act").strip()
            year = str(act.get("year") or "")
            act_url = act.get("url") or ""
            pdf_url = act.get("pdf_url") or ""
            sections = act.get("sections") or []

            for section_idx, section in enumerate(sections, start=1):
                section_number = str(section.get("section_number") or section_idx)
                section_title = (section.get("title") or f"Section {section_number}").strip()
                section_text = (section.get("text") or "").strip()

                if len(section_text) < 40:
                    continue

                text = (
                    f"Act: {act_title}\n"
                    f"Year: {year}\n"
                    f"Section: {section_number} - {section_title}\n\n"
                    f"{section_text}"
                )

                docs.append(
                    RawDocument(
                        doc_id=f"act_{act_id}_sec_{section_idx}",
                        text=text,
                        metadata={
                            "dataset": "acts_sections",
                            "source_type": "act_section",
                            "act_id": act_id,
                            "year": year,
                            "act_title": act_title,
                            "section_number": section_number,
                            "section_title": section_title,
                            "act_url": act_url,
                            "pdf_url": pdf_url,
                        },
                    )
                )

        logger.info("Loaded %s raw documents from acts_sections", len(docs))
        return docs

    def load_constitution(self, file_path: str) -> List[RawDocument]:
        with open(file_path, "r", encoding="utf-8") as f:
            parts = json.load(f)

        docs: List[RawDocument] = []

        for part_idx, part in enumerate(parts, start=1):
            part_name = (part.get("part") or f"Part {part_idx}").strip()
            part_title = (part.get("title") or "").strip()
            part_about = (part.get("about") or "").strip()

            for article_idx, article in enumerate(part.get("articles") or [], start=1):
                article_number = str(article.get("article_number") or article_idx)
                article_title = (article.get("title") or f"Article {article_number}").strip()
                article_url = article.get("url") or ""
                law = article.get("Law") or {}

                content = (law.get("content") or "").strip()
                summary = (law.get("summary") or "").strip()
                versions = law.get("versions") or []

                base_meta = {
                    "dataset": "constitution_parts_enriched",
                    "source_type": "constitution_article",
                    "part": part_name,
                    "part_title": part_title,
                    "article_number": article_number,
                    "article_title": article_title,
                    "article_url": article_url,
                }

                if content:
                    text = (
                        f"Constitution Part: {part_name} - {part_title}\n"
                        f"About Part: {part_about}\n"
                        f"Article {article_number}: {article_title}\n\n"
                        f"Law Text:\n{content}"
                    )
                    docs.append(
                        RawDocument(
                            doc_id=f"const_{part_idx}_art_{article_number}_law",
                            text=text,
                            metadata={**base_meta, "content_variant": "law"},
                        )
                    )

                if summary:
                    text = (
                        f"Constitution Part: {part_name} - {part_title}\n"
                        f"Article {article_number}: {article_title}\n\n"
                        f"Debate Summary:\n{summary}"
                    )
                    docs.append(
                        RawDocument(
                            doc_id=f"const_{part_idx}_art_{article_number}_summary",
                            text=text,
                            metadata={**base_meta, "content_variant": "summary"},
                        )
                    )

                for version_idx, version in enumerate(versions, start=1):
                    version_title = (version.get("title") or f"Version {version_idx}").strip()
                    version_content = (version.get("content") or "").strip()
                    if not version_content:
                        continue

                    text = (
                        f"Constitution Part: {part_name} - {part_title}\n"
                        f"Article {article_number}: {article_title}\n"
                        f"Historical Version: {version_title}\n\n"
                        f"{version_content}"
                    )
                    docs.append(
                        RawDocument(
                            doc_id=f"const_{part_idx}_art_{article_number}_ver_{version_idx}",
                            text=text,
                            metadata={
                                **base_meta,
                                "content_variant": "version",
                                "version_title": version_title,
                            },
                        )
                    )

        logger.info("Loaded %s raw documents from constitution dataset", len(docs))
        return docs

    def load_cases(self, file_path: str) -> List[RawDocument]:
        with open(file_path, "r", encoding="utf-8") as f:
            categories = json.load(f)

        docs: List[RawDocument] = []

        for category_idx, category in enumerate(categories, start=1):
            category_name = (category.get("name") or f"Category {category_idx}").strip()
            category_url = category.get("url") or ""

            for case_idx, case in enumerate(category.get("judgement") or [], start=1):
                case_title = (case.get("title") or f"Case {case_idx}").strip()
                case_url = case.get("url") or ""
                case_year = str(case.get("year") or "")
                case_subject = (case.get("subject") or category_name).strip()

                blocks = case.get("content") or []
                rendered_blocks: List[str] = []
                for block in blocks:
                    heading = (block.get("heading") or "").strip()
                    points = block.get("points") or []
                    point_lines = [f"- {str(point).strip()}" for point in points if str(point).strip()]

                    if heading and point_lines:
                        rendered_blocks.append(heading + "\n" + "\n".join(point_lines))
                    elif heading:
                        rendered_blocks.append(heading)
                    elif point_lines:
                        rendered_blocks.append("\n".join(point_lines))

                case_text = "\n\n".join(block for block in rendered_blocks if block.strip())
                if len(case_text) < 40:
                    continue

                text = (
                    f"Landmark Case: {case_title}\n"
                    f"Year: {case_year}\n"
                    f"Subject: {case_subject}\n"
                    f"Category: {category_name}\n\n"
                    f"{case_text}"
                )

                docs.append(
                    RawDocument(
                        doc_id=f"case_{category_idx}_{case_idx}",
                        text=text,
                        metadata={
                            "dataset": "final_with_content",
                            "source_type": "landmark_case",
                            "category_name": category_name,
                            "category_url": category_url,
                            "case_title": case_title,
                            "case_url": case_url,
                            "case_year": case_year,
                            "case_subject": case_subject,
                        },
                    )
                )

        logger.info("Loaded %s raw documents from landmark cases dataset", len(docs))
        return docs

    def build_unified_chunks(
        self,
        acts_file: str,
        constitution_file: str,
        cases_file: str,
        output_file: str,
    ) -> int:
        all_docs: List[RawDocument] = []
        all_docs.extend(self.load_acts(acts_file))
        all_docs.extend(self.load_constitution(constitution_file))
        all_docs.extend(self.load_cases(cases_file))

        records = self._build_chunk_records(all_docs)

        with open(output_file, "w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("Wrote %s chunk records to %s", len(records), output_file)
        return len(records)



def build_embeddings() -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings
    from deepinfra_embeddings import DeepInfraEmbeddings

    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-large-v2")
    embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    embedding_device = os.getenv("EMBEDDING_DEVICE", "cuda")
    deepinfra_token = os.getenv("DEEPINFRA_TOKEN")

    class E5Embeddings:
        def __init__(self, model_name: str, batch_size: int):
            try:
                self.base = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": embedding_device},
                    encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
                )
                logger.info("Loaded embedding model on device=%s", embedding_device)
            except Exception as exc:
                if embedding_device == "cpu":
                    raise
                logger.warning("Embedding init failed on device=%s (%s). Falling back to cpu.", embedding_device, exc)
                self.base = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
                )
                logger.info("Loaded embedding model on device=cpu")

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # e5 models expect passages to be prefixed for best retrieval quality.
            return self.base.embed_documents([f"passage: {text}" for text in texts])

        def embed_query(self, text: str) -> List[float]:
            return self.base.embed_query(f"query: {text}")

    class DeepInfraE5Embeddings:
        def __init__(self, model_name: str, api_token: str, local_factory):
            self.base = DeepInfraEmbeddings(model_name=model_name, api_token=api_token)
            self.local_factory = local_factory
            self.local_embedder = None

        def _fallback(self, exc: Exception):
            if self.local_embedder is None:
                logger.warning("DeepInfra embedding request failed (%s). Falling back to local model.", exc)
                self.local_embedder = self.local_factory()
            return self.local_embedder

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # Keep prefixes aligned with local e5 usage for consistent retrieval behavior.
            try:
                return self.base.embed_documents([f"passage: {text}" for text in texts])
            except Exception as exc:
                return self._fallback(exc).embed_documents(texts)

        def embed_query(self, text: str) -> List[float]:
            try:
                return self.base.embed_query(f"query: {text}")
            except Exception as exc:
                return self._fallback(exc).embed_query(text)

    if deepinfra_token:
        try:
            logger.info("DEEPINFRA_TOKEN detected. Using DeepInfra embeddings with model=%s", embedding_model_name)
            return DeepInfraE5Embeddings(
                model_name=embedding_model_name,
                api_token=deepinfra_token,
                local_factory=lambda: E5Embeddings(
                    model_name=embedding_model_name,
                    batch_size=embedding_batch_size,
                ),
            )
        except Exception as exc:
            logger.warning("DeepInfra embedding init failed (%s). Falling back to local model.", exc)

    logger.info("Using local HuggingFace embeddings with model=%s", embedding_model_name)
    return E5Embeddings(model_name=embedding_model_name, batch_size=embedding_batch_size)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ingest_to_pinecone(
    chunk_file: str,
    index_name: str,
    namespace: Optional[str] = None,
    batch_size: int = 64,
) -> int:
    from langchain_pinecone import PineconeVectorStore

    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY is required for ingestion")

    embeddings = build_embeddings()
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key="text",
        namespace=namespace,
    )

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    total = 0

    def flush_batch() -> None:
        nonlocal texts, metadatas, ids, total
        if not texts:
            return

        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        total += len(texts)
        logger.info("Ingested %s vectors", total)

        texts = []
        metadatas = []
        ids = []

    for record in iter_jsonl(chunk_file):
        texts.append(record["text"])
        metadatas.append(record.get("metadata", {}))
        ids.append(record["id"])

        if len(texts) >= batch_size:
            flush_batch()

    flush_batch()
    return total



def build_rag_chain(index_name: str, namespace: Optional[str] = None, top_k: int = 6):
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_groq import ChatGroq
    from langchain_pinecone import PineconeVectorStore

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is required for querying")

    embeddings = build_embeddings()
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key="text",
        namespace=namespace,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    try:
        prompt = hub.pull("therager4000/legal_llm_prompt_indian")
    except Exception:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            "You are a legal assistant. Use only the provided context to answer.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )

    model_name = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
    llm = ChatGroq(model=model_name, verbose=False)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def query_rag(query: str, index_name: str, namespace: Optional[str], top_k: int) -> str:
    rag_chain = build_rag_chain(index_name=index_name, namespace=namespace, top_k=top_k)

    retries = 3
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            return rag_chain.invoke(query)
        except Exception as exc:
            if attempt == retries:
                raise
            logger.warning("Query failed on attempt %s/%s: %s", attempt, retries, exc)
            time.sleep(delay)
            delay *= 1.5

    raise RuntimeError("Unreachable retry state")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified RAG pipeline for new_data JSON files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert new_data JSON files into chunked JSONL")
    prepare.add_argument("--acts", default="new_data/acts_sections (2).json")
    prepare.add_argument("--constitution", default="new_data/constitution_parts_enriched.json")
    prepare.add_argument("--cases", default="new_data/final_with_content.json")
    prepare.add_argument("--output", default="new_data_chunked_documents.jsonl")
    prepare.add_argument("--chunk-size", type=int, default=1000)
    prepare.add_argument("--chunk-overlap", type=int, default=150)

    ingest = subparsers.add_parser("ingest", help="Ingest prepared chunks into Pinecone")
    ingest.add_argument("--chunk-file", default="new_data_chunked_documents.jsonl")
    ingest.add_argument("--index-name", default=os.getenv("PINECONE_NEW_INDEX_NAME", os.getenv("PINECONE_INDEX_NAME", "caus-legal-vdb")))
    ingest.add_argument("--namespace", default="new-data-v1")
    ingest.add_argument("--batch-size", type=int, default=64)

    query = subparsers.add_parser("query", help="Query the new RAG pipeline")
    query.add_argument("--query", required=True)
    query.add_argument("--index-name", default=os.getenv("PINECONE_NEW_INDEX_NAME", os.getenv("PINECONE_INDEX_NAME", "caus-legal-vdb")))
    query.add_argument("--namespace", default="new-data-v1")
    query.add_argument("--top-k", type=int, default=6)

    return parser.parse_args()



def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv is not installed; using environment variables from shell only.")

    args = parse_args()

    if args.command == "prepare":
        builder = UnifiedLegalChunkBuilder(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = builder.build_unified_chunks(
            acts_file=args.acts,
            constitution_file=args.constitution,
            cases_file=args.cases,
            output_file=args.output,
        )
        print(f"Prepared {total} chunks at {args.output}")

    elif args.command == "ingest":
        total = ingest_to_pinecone(
            chunk_file=args.chunk_file,
            index_name=args.index_name,
            namespace=args.namespace,
            batch_size=args.batch_size,
        )
        print(f"Ingested {total} vectors into index={args.index_name}, namespace={args.namespace}")

    elif args.command == "query":
        answer = query_rag(
            query=args.query,
            index_name=args.index_name,
            namespace=args.namespace,
            top_k=args.top_k,
        )
        print(answer)


if __name__ == "__main__":
    main()
