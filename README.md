# SHL Assessment Recommender

Conversational agent that recommends SHL assessments via a FastAPI service.

## Local setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test:
```bash
python test_api.py
```

## Deploy to Render

1. Push this repo to GitHub (already done).
2. Go to [render.com](https://render.com) → New → Web Service.
3. Connect your GitHub repo.
4. Render auto-detects `render.yaml` — no manual config needed.
5. Add environment variable: `GROQ_API_KEY` = your key.
6. Deploy. The `/health` endpoint confirms readiness.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` |
| POST | `/chat` | Conversational agent |
| GET | `/docs` | Swagger UI |

### POST /chat

Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years"}
  ]
}
```

Response:
```json
{
  "reply": "Here are 3 assessments for a mid-level Java developer.",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Architecture

- **Catalog**: 377 SHL Individual Test Solutions in `catalog.json`
- **Retrieval**: `all-MiniLM-L6-v2` embeddings + cosine similarity → top-25 items per query
- **Mentioned items**: Any catalog item named in the conversation is always included in context
- **LLM**: Groq `llama-3.3-70b-versatile` with JSON mode
- **Token budget**: ~4-5k tokens per request (fits Groq free tier)
- **Turn cap**: Enforced server-side at 8 turns
