## Assistly AI Chatbot Backend (FastAPI + WebSocket)

A lightweight FastAPI backend that exposes a WebSocket endpoint for a lead-generation chatbot. On connection, it fetches user context from your API using signed headers, conducts a short guided conversation (lead type → service type → name → phone), answers user questions using a small GPT model, and then posts the lead to your API.

### Environment
Create a `.env` file in the project root with:

```
# Backend API base URL (memory: single .env; default shown)
API_BASE_URL=http://localhost:5000

# OpenAI API key for GPT.
OPENAI_API_KEY=
# GPT model name. Defaults to latest nano tier.
GPT_MODEL=gpt-5-nano

# TP signing secret for context fetch headers (HMAC secret you provide)
TP_SIGN_SECRET=
```

### Install & Run

```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### WebSocket Endpoint
- URL: `ws://localhost:8000/ws?user_id=PUBLIC_USER_ID`
- Messages are JSON objects. Minimal shape:
  - Client → Server: `{ "type": "user", "content": "hi, I want an implant" }`
  - Server → Client: `{ "type": "bot", "content": "...", "step": "lead_type|service_type|name|phone|complete" }`

On connect, the server will:
1) Fetch user context from `GET /api/v1/users/public/{id}/context` using signed headers (`x-tp-ts`, `x-tp-nonce`, `x-tp-sign`).
2) Greet, ask for a lead type, then a service type, then name and phone.
3) Create the lead using public endpoint: `POST /api/v1/leads/public/{USER_ID}` with signed headers.

### Notes
- The signer computes headers `x-tp-ts`, `x-tp-nonce`, and `x-tp-sign` using an HMAC secret you provide via `TP_SIGN_SECRET`.
- If a user asks questions during the flow, the bot answers briefly first, then continues the lead-collection prompts.
