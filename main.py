# ============================================================
# Spiralside API Backend
# FastAPI server — proxies Anthropic API calls safely
# Loads character personas from HuggingFace Space at startup
# Deploy on Railway — set env vars ANTHROPIC_API_KEY + SUPABASE_*
# ============================================================

import os
import httpx
from datetime import date
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ── ENV VARS ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL      = os.environ.get("SUPABASE_URL")
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY")

# ── LIMITS ────────────────────────────────────────────────
FREE_DAILY_LIMIT  = 10
CREDIT_COST       = 0.01

# ── HF CHARACTER FILES ────────────────────────────────────
HF_RAW = "https://huggingface.co/spaces/quarterbitgames/spiralside/raw/main"

CHARACTER_FILES = {
    "sky":       "characters/sky.txt",
    "cold":      "characters/cold.txt",
    "monday":    "characters/monday.txt",
    "grit":      "characters/grit.txt",
    "architect": "characters/architect.txt",
    "cat":       "characters/cat.txt",
}

character_cache: dict = {}

async def load_characters():
    async with httpx.AsyncClient(timeout=15) as client:
        for name, path in CHARACTER_FILES.items():
            try:
                resp = await client.get(f"{HF_RAW}/{path}")
                if resp.status_code == 200:
                    character_cache[name] = resp.text.strip()
                    print(f"[characters] loaded: {name}")
                else:
                    print(f"[characters] not found: {name} ({resp.status_code})")
            except Exception as e:
                print(f"[characters] error loading {name}: {e}")

# ── APP LIFESPAN ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_characters()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SUPABASE CLIENT ───────────────────────────────────────
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── AUTH HELPER ───────────────────────────────────────────
async def verify_user(authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    try:
        sb = get_supabase()
        user_resp = sb.auth.get_user(token)
        if not user_resp or not user_resp.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_resp.user.id, sb
    except HTTPException:
        raise
    except Exception as e:
        print(f"[auth error] {type(e).__name__}: {e}")
        raise HTTPException(status_code=401, detail=f"Auth failed: {str(e)}")

# ── REQUEST MODELS ────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    system_prompt: str
    vault_context: str = ""
    bot_name: str = ""

# ── HEALTH CHECK ──────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "spiralside-api", "characters_loaded": list(character_cache.keys())}

# ── USAGE ENDPOINT ────────────────────────────────────────
@app.get("/usage")
async def get_usage(authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    today = str(date.today())
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    if not usage.data:
        return {"credits": 0.0, "free_messages_today": 0, "is_paid": False}
    record = usage.data[0]
    if record.get("last_reset_date") != today:
        sb.table("user_usage").update({"free_messages_today": 0, "last_reset_date": today}).eq("user_id", user_id).execute()
        record["free_messages_today"] = 0
    return {"credits": record["credits"], "free_messages_today": record["free_messages_today"], "is_paid": record["is_paid"]}

# ── CHAT ENDPOINT ─────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest, authorization: str = Header(None)):

    user_id, sb = await verify_user(authorization)

    today = str(date.today())
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()

    if not usage.data:
        sb.table("user_usage").insert({"user_id": user_id, "credits": 0.0, "free_messages_today": 0, "last_reset_date": today, "is_paid": False}).execute()
        free_count, credits, is_paid = 0, 0.0, False
    else:
        record = usage.data[0]
        credits, is_paid, free_count = record["credits"], record["is_paid"], record["free_messages_today"]
        if record.get("last_reset_date") != today:
            free_count = 0
            sb.table("user_usage").update({"free_messages_today": 0, "last_reset_date": today}).eq("user_id", user_id).execute()

    if not is_paid:
        if free_count >= FREE_DAILY_LIMIT:
            raise HTTPException(status_code=429, detail=f"Free limit reached ({FREE_DAILY_LIMIT}/day). Add credits to continue.")
    else:
        if credits < CREDIT_COST:
            raise HTTPException(status_code=402, detail="Out of credits. Please add more to continue.")

    # ── Build system prompt — check for Spiral City character ──
    bot_name_lower = (req.bot_name or "").strip().lower()
    character_prompt = character_cache.get(bot_name_lower)

    if character_prompt:
        system = character_prompt
        print(f"[chat] character mode: {bot_name_lower}")
    else:
        system = req.system_prompt

    if req.vault_context:
        system += f"\n\nThe user has shared these files:\n{req.vault_context}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1000, "system": system, "messages": [{"role": "user", "content": req.message}]}
            )
        resp.raise_for_status()
        reply = resp.json()["content"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

    if not is_paid:
        sb.table("user_usage").update({"free_messages_today": free_count + 1}).eq("user_id", user_id).execute()
    else:
        sb.table("user_usage").update({"credits": round(credits - CREDIT_COST, 4)}).eq("user_id", user_id).execute()

    return {
        "reply": reply,
        "usage": {
            "is_paid": is_paid,
            "free_messages_today": free_count + 1 if not is_paid else None,
            "free_limit": FREE_DAILY_LIMIT,
            "credits_remaining": round(credits - CREDIT_COST, 4) if is_paid else None
        }
    }

# ── SHEET ENDPOINT ────────────────────────────────────────
@app.post("/sheet")
async def update_sheet(req: ChatRequest, authorization: str = Header(None)):
    await verify_user(authorization)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 500, "system": 'Extract character traits from conversation. Return ONLY valid JSON: {"traits":[{"label":"string","value":"string","score":0-100}],"summary":"one sentence"} No markdown.', "messages": [{"role": "user", "content": f'User: "{req.message}"\nBot: "{req.system_prompt}"\nExtract 4 traits.'}]}
            )
        return {"sheet": resp.json()["content"][0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── RELOAD CHARACTERS (no redeploy needed) ────────────────
@app.post("/reload-characters")
async def reload_characters():
    await load_characters()
    return {"loaded": list(character_cache.keys())}
