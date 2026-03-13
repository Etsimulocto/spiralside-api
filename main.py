# ============================================================
# Spiralside API Backend — with PayPal checkout
# ============================================================

import os
import httpx
import json
from datetime import date
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from collections import defaultdict
import time

# ── RATE LIMITING ─────────────────────────────────────────────
# In-memory per-user rate limit: max 20 requests per 60 seconds
RATE_LIMIT_MAX     = 20
RATE_LIMIT_WINDOW  = 60  # seconds
_rate_buckets: dict = defaultdict(list)

def check_rate_limit(user_id: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Drop timestamps outside the window
    _rate_buckets[user_id] = [t for t in _rate_buckets[user_id] if t > window_start]
    if len(_rate_buckets[user_id]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down a little. 🌀")
    _rate_buckets[user_id].append(now)

# ── ENV VARS ──────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL       = os.environ.get("SUPABASE_URL")
SUPABASE_KEY       = os.environ.get("SUPABASE_SERVICE_KEY")
PAYPAL_CLIENT_ID   = os.environ.get("PAYPAL_CLIENT_ID")
PAYPAL_SECRET      = os.environ.get("PAYPAL_SECRET")
HF_TOKEN           = os.environ.get("HF_TOKEN")
HF_TOKEN           = os.environ.get("HF_TOKEN")
PAYPAL_BASE        = "https://api-m.paypal.com"  # live

# ── LIMITS ────────────────────────────────────────────────
FREE_LIFETIME_LIMIT = 15  # total messages ever, never resets
CREDIT_COST      = 0.01

# Credit packs: amount in USD -> credits granted
CREDIT_PACKS = {
    "5":  500,
    "10": 1100,
    "20": 2400,
}

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

# ── PAYPAL HELPERS ────────────────────────────────────────
async def get_paypal_token() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PAYPAL_BASE}/v1/oauth2/token",
            auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
            data={"grant_type": "client_credentials"}
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

async def create_paypal_order(amount: str, user_id: str) -> dict:
    token = await get_paypal_token()
    credits = CREDIT_PACKS.get(amount, 0)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PAYPAL_BASE}/v2/checkout/orders",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "intent": "CAPTURE",
                "purchase_units": [{
                    "amount": {"currency_code": "USD", "value": amount},
                    "description": f"Spiralside {credits} credits",
                    "custom_id": f"{user_id}|{amount}"  # store user+amount for webhook
                }],
                "application_context": {
                    "return_url": "https://www.spiralside.com/?payment=success",
                    "cancel_url": "https://www.spiralside.com/?payment=cancelled"
                }
            }
        )
        resp.raise_for_status()
        return resp.json()

async def capture_paypal_order(order_id: str) -> dict:
    token = await get_paypal_token()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PAYPAL_BASE}/v2/checkout/orders/{order_id}/capture",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        resp.raise_for_status()
        return resp.json()

# ── APP LIFESPAN ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_characters()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── SUPABASE ──────────────────────────────────────────────
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

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
        raise HTTPException(status_code=401, detail=f"Auth failed: {str(e)}")

# ── REQUEST MODELS ────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    system_prompt: str
    vault_context: str = ""
    bot_name: str = ""

class OrderRequest(BaseModel):
    amount: str  # "5", "10", or "20"

class CaptureRequest(BaseModel):
    order_id: str

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512

# ── HEALTH ────────────────────────────────────────────────
# v1.1 — image generation endpoint added
# v1.1 — image generation endpoint added
@app.get("/")
def health():
    return {"status": "ok", "service": "spiralside-api", "characters_loaded": list(character_cache.keys())}

# ── USAGE ─────────────────────────────────────────────────
@app.get("/usage")
async def get_usage(authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    check_rate_limit(user_id)
    today = str(date.today())
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    if not usage.data:
        return {"credits": 0.0, "free_messages_today": 0, "is_paid": False}
    record = usage.data[0]
    if record.get("last_reset_date") != today:
        sb.table("user_usage").update({"free_messages_today": 0, "last_reset_date": today}).eq("user_id", user_id).execute()
        record["free_messages_today"] = 0
    return {"credits": record["credits"], "free_messages_today": record["free_messages_today"], "is_paid": record["is_paid"]}

# ── CHAT ──────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    today = str(date.today())
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    if not usage.data:
        sb.table("user_usage").insert({"user_id": user_id, "credits": 0.0, "total_messages": 0, "is_paid": False}).execute()
        credits, is_paid, total_msgs = 0.0, False, 0
    else:
        record = usage.data[0]
        credits, is_paid = record["credits"], record["is_paid"]
        total_msgs = record.get("total_messages", 0)
    if not is_paid:
        if total_msgs >= FREE_LIFETIME_LIMIT:
            raise HTTPException(status_code=429, detail=f"Free messages used ({FREE_LIFETIME_LIMIT} lifetime). Add credits to continue.")
    else:
        if credits < CREDIT_COST:
            raise HTTPException(status_code=402, detail="Out of credits. Please add more to continue.")
    bot_name_lower = (req.bot_name or "").strip().lower()
    character_prompt = character_cache.get(bot_name_lower)
    system = character_prompt if character_prompt else req.system_prompt
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
        sb.table("user_usage").update({"total_messages": total_msgs + 1}).eq("user_id", user_id).execute()
    else:
        sb.table("user_usage").update({"credits": round(credits - CREDIT_COST, 4), "total_messages": total_msgs + 1}).eq("user_id", user_id).execute()
    return {"reply": reply, "usage": {"is_paid": is_paid, "total_messages": total_msgs + 1, "lifetime_limit": FREE_LIFETIME_LIMIT, "credits_remaining": round(credits - CREDIT_COST, 4) if is_paid else None}}

# ── SHEET ─────────────────────────────────────────────────
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

# ── PAYPAL: CREATE ORDER ──────────────────────────────────
@app.post("/create-order")
async def create_order(req: OrderRequest, authorization: str = Header(None)):
    user_id, _ = await verify_user(authorization)
    if req.amount not in CREDIT_PACKS:
        raise HTTPException(status_code=400, detail="Invalid amount. Choose 5, 10, or 20.")
    try:
        order = await create_paypal_order(req.amount, user_id)
        # Find the approval URL for redirect
        approve_url = next((l["href"] for l in order["links"] if l["rel"] == "approve"), None)
        return {"order_id": order["id"], "approve_url": approve_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PayPal error: {str(e)}")

# ── PAYPAL: CAPTURE ORDER (user returns after payment) ────
@app.post("/capture-order")
async def capture_order(req: CaptureRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    try:
        result = await capture_paypal_order(req.order_id)
        if result["status"] != "COMPLETED":
            raise HTTPException(status_code=400, detail="Payment not completed.")
        # Extract amount from custom_id: "user_id|amount"
        custom_id = result["purchase_units"][0].get("custom_id", "")
        parts = custom_id.split("|")
        amount = parts[1] if len(parts) == 2 else "5"
        credits_to_add = CREDIT_PACKS.get(amount, 500)
        # Add credits to user
        usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
        if not usage.data:
            sb.table("user_usage").insert({"user_id": user_id, "credits": float(credits_to_add), "free_messages_today": 0, "last_reset_date": str(date.today()), "is_paid": True}).execute()
        else:
            current = usage.data[0]["credits"] or 0
            sb.table("user_usage").update({"credits": current + credits_to_add, "is_paid": True}).eq("user_id", user_id).execute()
        print(f"[payment] {user_id} purchased {credits_to_add} credits (${amount})")
        return {"success": True, "credits_added": credits_to_add, "amount": amount}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capture error: {str(e)}")


# ── IMAGE GENERATION ──────────────────────────────────────────
IMAGE_FREE_DAILY  = 3
IMAGE_CREDIT_COST = 5
HF_FLUX = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

@app.post("/generate-image")
async def generate_image(req: ImageRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    check_rate_limit(user_id)
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Image generation not configured.")
    today = str(date.today())
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    if not usage.data:
        raise HTTPException(status_code=403, detail="No usage record found.")
    record    = usage.data[0]
    is_paid   = record.get("is_paid", False)
    credits   = record.get("credits", 0)
    img_today = record.get("images_today", 0)
    img_date  = record.get("images_reset_date", "")
    if img_date != today:
        img_today = 0
        sb.table("user_usage").update({"images_today": 0, "images_reset_date": today}).eq("user_id", user_id).execute()
    if not is_paid:
        if img_today >= IMAGE_FREE_DAILY:
            raise HTTPException(status_code=429, detail=f"Free image limit ({IMAGE_FREE_DAILY}/day) reached. Add credits for more.")
        req.width = 512
        req.height = 512
    else:
        if credits < IMAGE_CREDIT_COST:
            raise HTTPException(status_code=402, detail="Not enough credits. Need 5 per image.")
        req.width  = min(max(req.width,  256), 1024)
        req.height = min(max(req.height, 256), 1024)
    try:
        payload = {"inputs": req.prompt}
        if req.negative_prompt:
            payload["negative_prompt"] = req.negative_prompt
        payload["width"]  = req.width
        payload["height"] = req.height
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                HF_FLUX,
                headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                json=payload
            )
        if r.status_code == 503:
            raise HTTPException(status_code=503, detail="Model loading, try again in 20 seconds.")
        if not r.is_success:
            raise HTTPException(status_code=500, detail=f"HF error {r.status_code}: {r.text[:200]}")
        if is_paid:
            sb.table("user_usage").update({"credits": round(credits - IMAGE_CREDIT_COST, 4)}).eq("user_id", user_id).execute()
        else:
            sb.table("user_usage").update({"images_today": img_today + 1}).eq("user_id", user_id).execute()
        import base64
        img_b64 = base64.b64encode(r.content).decode()
        return {"image": img_b64, "width": req.width, "height": req.height, "is_paid": is_paid,
                "free_images_used": img_today + 1 if not is_paid else None,
                "free_images_limit": IMAGE_FREE_DAILY if not is_paid else None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# ── RELOAD CHARACTERS ─────────────────────────────────────
@app.post("/reload-characters")
async def reload_characters():
    await load_characters()
    return {"loaded": list(character_cache.keys())}

# ── ADMIN: ADD CREDITS ────────────────────────────────────
# Protected by admin secret — set ADMIN_SECRET in Railway env vars
@app.post("/admin/add-credits")
async def admin_add_credits(request: Request):
    body = await request.json()
    secret = body.get("secret")
    email  = body.get("email")
    amount = int(body.get("amount", 0))
    admin_secret = os.environ.get("ADMIN_SECRET")
    if not admin_secret or secret != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not email or amount <= 0:
        raise HTTPException(status_code=400, detail="Need email and amount")
    sb = get_supabase()
    # Look up user by email
    users = sb.auth.admin.list_users()
    user = next((u for u in users if u.email == email), None)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {email} not found")
    usage = sb.table("user_usage").select("*").eq("user_id", user.id).execute()
    if not usage.data:
        sb.table("user_usage").insert({"user_id": user.id, "credits": float(amount), "free_messages_today": 0, "last_reset_date": str(date.today()), "is_paid": True}).execute()
    else:
        current = usage.data[0]["credits"] or 0
        sb.table("user_usage").update({"credits": current + amount, "is_paid": True}).eq("user_id", user.id).execute()
    print(f"[admin] added {amount} credits to {email}")
    return {"success": True, "email": email, "credits_added": amount}
