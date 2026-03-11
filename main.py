# ============================================================
# Spiralside API Backend
# FastAPI server — proxies Anthropic API calls safely
# Handles: free tier limits, credit deduction, rate limiting
# Deploy on Railway — set env vars ANTHROPIC_API_KEY + SUPABASE_*
# ============================================================

import os
import httpx
from datetime import date
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ── APP SETUP ──────────────────────────────────────────────
app = FastAPI()

# Allow requests from your frontend domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ENV VARS (set in Railway dashboard) ───────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL      = os.environ.get("SUPABASE_URL")
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY")  # service role key (private)

# ── LIMITS ────────────────────────────────────────────────
FREE_DAILY_LIMIT  = 10    # free messages per day
CREDIT_COST       = 0.01  # credits deducted per message
RATE_LIMIT_SECS   = 3     # min seconds between messages (future)

# ── SUPABASE CLIENT ───────────────────────────────────────
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── REQUEST MODELS ────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str          # user's message
    system_prompt: str    # bot personality built on frontend
    vault_context: str = ""  # optional file context

# ── HEALTH CHECK ──────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "spiralside-api"}

# ── CHAT ENDPOINT ─────────────────────────────────────────
@app.post("/chat")
async def chat(
    req: ChatRequest,
    authorization: str = Header(None)  # Supabase JWT from frontend
):
    # ── 1. Verify user is logged in ──
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.replace("Bearer ", "")

    # ── 2. Get user from Supabase token ──
    try:
        sb = get_supabase()
        # Use admin client to verify the JWT
        user_resp = sb.auth.get_user(token)
        if not user_resp or not user_resp.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = user_resp.user.id
    except HTTPException:
        raise
    except Exception as e:
        # Log the actual error to Railway logs for debugging
        print(f"[auth error] {type(e).__name__}: {e}")
        raise HTTPException(status_code=401, detail=f"Auth failed: {str(e)}")

    # ── 3. Check/update usage limits ──
    today = str(date.today())

    # Get or create usage record for this user
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()

    if not usage.data:
        # First time user — create record
        sb.table("user_usage").insert({
            "user_id": user_id,
            "credits": 0.0,
            "free_messages_today": 0,
            "last_reset_date": today,
            "is_paid": False
        }).execute()
        free_count = 0
        credits = 0.0
        is_paid = False
        last_reset = today
    else:
        record = usage.data[0]
        credits = record["credits"]
        is_paid = record["is_paid"]
        last_reset = record["last_reset_date"]
        free_count = record["free_messages_today"]

        # Reset daily counter if it's a new day
        if last_reset != today:
            free_count = 0
            sb.table("user_usage").update({
                "free_messages_today": 0,
                "last_reset_date": today
            }).eq("user_id", user_id).execute()

    # ── 4. Enforce limits ──
    if not is_paid:
        # Free tier check
        if free_count >= FREE_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Free limit reached ({FREE_DAILY_LIMIT} messages/day). Add credits to continue."
            )
    else:
        # Paid tier — check credits
        if credits < CREDIT_COST:
            raise HTTPException(
                status_code=402,
                detail="Out of credits. Please add more to continue."
            )

    # ── 5. Build system prompt ──
    system = req.system_prompt
    if req.vault_context:
        system += f"\n\nThe user has shared these files:\n{req.vault_context}"

    # ── 6. Call Anthropic API ──
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1000,
                    "system": system,
                    "messages": [{"role": "user", "content": req.message}]
                }
            )
        resp.raise_for_status()
        data = resp.json()
        reply = data["content"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

    # ── 7. Update usage after successful call ──
    if not is_paid:
        sb.table("user_usage").update({
            "free_messages_today": free_count + 1
        }).eq("user_id", user_id).execute()
    else:
        sb.table("user_usage").update({
            "credits": round(credits - CREDIT_COST, 4)
        }).eq("user_id", user_id).execute()

    # ── 8. Return reply ──
    return {
        "reply": reply,
        "usage": {
            "is_paid": is_paid,
            "free_messages_today": free_count + 1 if not is_paid else None,
            "free_limit": FREE_DAILY_LIMIT,
            "credits_remaining": round(credits - CREDIT_COST, 4) if is_paid else None
        }
    }

# ── CHARACTER SHEET ENDPOINT ──────────────────────────────
@app.post("/sheet")
async def update_sheet(
    req: ChatRequest,
    authorization: str = Header(None)
):
    # Verify auth (same as chat)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    try:
        sb = get_supabase()
        user_resp = sb.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Call Anthropic for trait extraction
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 500,
                    "system": 'Extract character traits from conversation. Return ONLY valid JSON: {"traits":[{"label":"string","value":"string","score":0-100}],"summary":"one sentence"} No markdown.',
                    "messages": [{"role": "user", "content": f'User: "{req.message}"\nBot: "{req.system_prompt}"\nExtract 4 traits.'}]
                }
            )
        data = resp.json()
        return {"sheet": data["content"][0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
