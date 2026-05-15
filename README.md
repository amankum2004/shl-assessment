# SHL Assessment Recommender

Conversational agent that recommends SHL assessments via a FastAPI service.

## Setup

```bash
pip install -r requirements.txt
```

Set your Groq API key in `.env`:
```
GROQ_API_KEY=your_key_here
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /chat` → conversational agent response

### POST /chat request
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years"}
  ]
}
```

### POST /chat response
```json
{
  "reply": "Here are 3 assessments for a mid-level Java developer.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Test

```bash
python test_api.py
```

## Architecture

- **Catalog**: 377 SHL Individual Test Solutions loaded from `catalog.json`
- **Retrieval**: `all-MiniLM-L6-v2` sentence embeddings + cosine similarity → top-30 items per query
- **LLM**: Groq `llama-3.3-70b-versatile` with JSON mode
- **Context**: ~4k tokens per request (fits Groq free tier)
