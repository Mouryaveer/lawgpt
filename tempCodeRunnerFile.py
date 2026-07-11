import langchain_groq, langchain_community, langchainhub,langchain,os
import json
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

os.environ["GROQ_API_KEY"]=("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")
os.environ["USER_AGENT"] = "MyLangChainScraper/mk2"


index_name = "caus-legal-vdb"
os.environ["PINECONE_API_KEY"] = "pcsk_51f5pJ_61yjbxDxYCwcsnESLu2RMsUjfBP9j1tbUaPBH9WRgQjX7YLXvsKAYHAnqB25SBo"

# Initialize embeddings with GPU support
model_kwargs = {
    'device': 'cuda'  # Use 'cuda' for GPU, 'cpu' for CPU
}

encode_kwargs = {
    'normalize_embeddings': True,  # Optional: normalize embeddings
    'batch_size': 8  # Adjust based on your GPU memory
}



embedz = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


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
vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embedz,
    text_key="text"
)




retriever=vector_store.as_retriever()

from langchain import hub
prompt=hub.pull("therager4000/indian-legal-assistant")


from langchain_groq import ChatGroq
llm =ChatGroq(model="qwen/qwen3-32b")


from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain=({"context":  retriever | format_docs, "question": RunnablePassthrough()}
          | prompt
          | llm
          | StrOutputParser())


def ragu(query):
    return rag_chain.invoke(query)


if __name__ == "__main__":
    response = ragu("What is the procedure for filing a legal appeal?")
    print(response)
