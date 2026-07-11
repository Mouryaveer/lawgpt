# Turn2Law API

A FastAPI-based legal query API with RAG (Retrieval Augmented Generation) for answering legal questions using Indian legal documents.

## Features

- **RAG System**: Uses Pinecone vector database for legal document retrieval
- **AI-Powered**: Leverages Groq's LLM for generating responses
- **RESTful API**: Clean FastAPI endpoints for frontend integration
- **CORS Enabled**: Ready for frontend integration

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory. You can copy from `.env.example`:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

```env
# Required
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=caus-legal-vdb

# Optional (with defaults)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langchain_api_key
EMBEDDING_MODEL_NAME=intfloat/e5-large-v2
EMBEDDING_BATCH_SIZE=8
GROQ_MODEL_NAME=openai/gpt-oss-20b
USER_AGENT=MyLangChainScraper/mk2
```

**Required Variables:**
- `GROQ_API_KEY` - Get from https://console.groq.com/
- `PINECONE_API_KEY` - Get from https://www.pinecone.io/
- `PINECONE_INDEX_NAME` - Your Pinecone index name

**Optional Variables (have defaults):**
- `LANGCHAIN_TRACING_V2` - Enable/disable LangChain tracing
- `LANGCHAIN_API_KEY` - For LangChain tracing
- `EMBEDDING_MODEL_NAME` - HuggingFace embedding model
- `EMBEDDING_BATCH_SIZE` - Batch size for embeddings
- `GROQ_MODEL_NAME` - Groq model to use
- `USER_AGENT` - User agent string

### 3. Run the API Server

```bash
python app.py
```

The API will be available at `http://localhost:8001`

## API Endpoints

### POST `/api/query`

Process a legal query and return an AI-generated response.

**Request:**
```json
{
  "query": "What is the punishment for murder?"
}
```

**Response:**
```json
{
  "response": "According to Section 103(1)...",
  "model_used": "openai/gpt-oss-20b"
}
```

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "openai/gpt-oss-20b",
  "rag_enabled": true
}
```

### GET `/`

Root endpoint with API information.

## Frontend Integration

### JavaScript/TypeScript Example

```javascript
async function sendQuery(query) {
  const response = await fetch('http://localhost:8001/api/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query: query })
  });
  
  const data = await response.json();
  return data.response;
}

// Usage
const answer = await sendQuery("What is the punishment for murder?");
console.log(answer);
```

### React Example

```jsx
import { useState } from 'react';

function LegalQueryComponent() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const res = await fetch('http://localhost:8001/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      console.error('Error:', error);
      setResponse('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your legal query"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Submit Query'}
        </button>
      </form>
      {response && <div>{response}</div>}
    </div>
  );
}
```

## Testing

You can test the API using the provided `frontend_example.html` file:

1. Open `frontend_example.html` in your browser
2. Enter your API URL (default: `http://localhost:8001/api/query`)
3. Type your legal query
4. Click "Get Legal Answer"

Or use curl:

```bash
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the punishment for murder?"}'
```

## Deployment

### Using Uvicorn (Production)

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --workers 4
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

Build and run:

```bash
docker build -t turn2law-api .
docker run -p 8001:8001 --env-file .env turn2law-api
```

### Environment Variables for Production

Update CORS settings in `app.py`:

```python
allow_origins=["https://your-frontend-domain.com"]
```

## Project Structure

```
.
├── app.py              # FastAPI application
├── main.py             # RAG system implementation
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not in git)
├── frontend_example.html  # Example frontend interface
└── README.md           # This file
```

## Notes

- The RAG system uses Pinecone vector database for document retrieval
- The system automatically falls back to CPU if CUDA is not available
- All queries are processed asynchronously for better performance
- The API includes proper error handling and logging

## New Data RAG Pipeline

The repository includes a unified pipeline for the new datasets in `new_data/`:

- `acts_sections (2).json` (bare acts and sections)
- `constitution_parts_enriched.json` (Constitution parts/articles)
- `final_with_content.json` (landmark case law)

Script: `new_data_rag_pipeline.py`

Embeddings for this pipeline are local HuggingFace embeddings using `intfloat/e5-large-v2` (with e5 query/passages prefixes). `DEEPINFRA_TOKEN` is ignored by this script.

1. Prepare unified chunks

```bash
python new_data_rag_pipeline.py prepare \
  --acts "new_data/acts_sections (2).json" \
  --constitution "new_data/constitution_parts_enriched.json" \
  --cases "new_data/final_with_content.json" \
  --output new_data_chunked_documents.jsonl
```

2. Ingest chunks to Pinecone (recommended: dedicated namespace)

```bash
python new_data_rag_pipeline.py ingest \
  --chunk-file new_data_chunked_documents.jsonl \
  --index-name caus-legal-vdb \
  --namespace new-data-v1
```

3. Query the new-data RAG pipeline

```bash
python new_data_rag_pipeline.py query \
  --query "What does Article 21 protect?" \
  --index-name caus-legal-vdb \
  --namespace new-data-v1
```

Optional environment variable:

- `PINECONE_NEW_INDEX_NAME` to default the new pipeline index.

## License

[Your License Here]

