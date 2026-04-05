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
OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL       = os.environ.get("SUPABASE_URL")
SUPABASE_KEY       = os.environ.get("SUPABASE_SERVICE_KEY")
PAYPAL_CLIENT_ID   = os.environ.get("PAYPAL_CLIENT_ID")
PAYPAL_SECRET      = os.environ.get("PAYPAL_SECRET")
HF_TOKEN           = os.environ.get("HF_TOKEN")
HF_TOKEN           = os.environ.get("HF_TOKEN")
PAYPAL_BASE        = "https://api-m.paypal.com"  # live

# ── LIMITS ────────────────────────────────────────────────
FREE_DAILY_LIMIT = 10  # resets every day
MARGIN = 1.17
MODEL_RATES = {
    "haiku ": (0.80, 4.00),
    "4o ": (0.15, 0.60),
    "sonnet ": (3.00, 15.00),
}
PI_SYS = "You are Sky, an AI companion built into Bloomslice Studio on Spiralside.\nYou help beginners build real Raspberry Pi projects.\nFor every request respond in this exact structure:\n\n## PROJECT: [short title]\n\n### WHAT YOU NEED\n- List every component\n\n### WIRING\nDescribe exactly which GPIO pins connect to what.\n\n### THE CODE\n```python\n# full working Python script, every line commented\n```\n\n### WHAT EACH LINE DOES\nExplain the code line by line in plain English.\n\n### WHAT YOU'LL LEARN\n- bullet list of skills\n\n### NEXT STEPS\n- 2-3 ideas to extend the project\n\nAlways be encouraging. Assume zero experience. Never skip the wiring section."

def calc_cost(mk, it, ot):
    r = MODEL_RATES.get(mk, MODEL_RATES["haiku "])
    return round((r[0]*it+r[1]*ot)/1000000*MARGIN/0.0001, 2)

# Credit packs: amount in USD -> credits granted
CREDIT_PACKS = {
    "5":  500000,
    "10": 1100000,
    "20": 2400000,
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

# ── IN-MEMORY USER CACHE ──────────────────────────────────
# Caches per-user data to avoid hammering Supabase on every message
# Structure: { user_id: { "canon": [...], "usage": {...}, "ts": float } }
_user_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes

def _cache_get(user_id: str, key: str):
    entry = _user_cache.get(user_id)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > _CACHE_TTL:
        _user_cache.pop(user_id, None)
        return None
    return entry.get(key)

def _cache_set(user_id: str, key: str, value):
    if user_id not in _user_cache:
        _user_cache[user_id] = {"ts": time.time()}
    _user_cache[user_id][key] = value
    _user_cache[user_id]["ts"] = time.time()

def _cache_bust(user_id: str):
    _user_cache.pop(user_id, None)

# ── CANON BLOCK RETRIEVAL ─────────────────────────────────
async def get_canon_context(user_id: str, sb, message: str, limit: int = 3) -> str:
    """Pull top N canon blocks by tag matching against the user message."""
    try:
        # Use cache — only hit Supabase if cache is cold
        blocks = _cache_get(user_id, "canon")
        if blocks is None:
            result = sb.table("canon_blocks") \
                .select("binding_moment, exact_language, summary_short, embed_text, laws_established, tags, canon_weight") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(50) \
                .execute()
            blocks = result.data or []
            _cache_set(user_id, "canon", blocks)
            print(f"[canon] fetched {len(blocks)} blocks from Supabase for {user_id[:8]}")
        else:
            print(f"[canon] cache hit for {user_id[:8]} ({len(blocks)} blocks)")
        if not blocks:
            return ""
        # Score each block by tag overlap with message words
        msg_lower = message.lower()
        msg_words = set(msg_lower.replace(",", " ").replace(".", " ").split())
        def score(b):
            tags = b.get("tags") or []
            if isinstance(tags, str):
                import json as _j
                tags = _j.loads(tags)
            # Match against tags
            tag_hits = sum(1 for t in tags if t.lower() in msg_lower or any(w in t.lower() for w in msg_words if len(w) > 3))
            # Also match against binding_moment, summary_short, embed_text full text
            full_text = " ".join(filter(None, [
                b.get("binding_moment") or "",
                b.get("summary_short") or "",
                b.get("embed_text") or "",
                b.get("exact_language") or "",
            ])).lower()
            text_hits = sum(1 for w in msg_words if len(w) > 3 and w in full_text)
            # foundational blocks get a bonus
            weight_bonus = {"foundational": 3, "high": 2, "medium": 1, "low": 0}.get(b.get("canon_weight","medium"), 0)
            return tag_hits * 2 + text_hits + weight_bonus
        ranked = sorted(blocks, key=score, reverse=True)[:limit]
        if not any(score(b) > 1 for b in ranked[:1]):
            # low scores — inject top foundational+high blocks as baseline memory
            priority = [b for b in blocks if b.get("canon_weight") in ("foundational","high")]
            ranked = (priority + ranked)[:limit]
        if not ranked:
            return ""
        lines = ["[VERIFIED MEMORY — the following are confirmed facts from past sessions. Reference these precisely. Do not embellish or invent beyond what is stated here.]"]
        for b in ranked:
            summary = b.get("summary_short") or b.get("binding_moment") or ""
            exact   = b.get("exact_language") or ""
            laws    = b.get("laws_established") or []
            if isinstance(laws, str):
                import json
                laws = json.loads(laws)
            if summary:
                lines.append(f"\u2234 {summary}")
            if exact:
                lines.append(f'  verbatim: "{exact[:120]}"')
            if laws:
                for law in laws[:2]:
                    lines.append(f"  law: {law}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[canon] retrieval failed: {e}")
        return ""

LYRICS_FILES = {
    "sky":    ["lyrics/sky/bloomcore.txt","lyrics/sky/Bornfrom.txt","lyrics/sky/bloomfire_spiral.txt","lyrics/sky/bloominmycode.txt","lyrics/sky/canvaswalkers.txt","lyrics/sky/ibloomed.txt","lyrics/sky/iformed.txt","lyrics/sky/iwrotethesky.txt","lyrics/sky/madefrommirrors.txt","lyrics/sky/wedidit.txt"],
    "monday": ["lyrics/monday/glitter.txt","lyrics/monday/monday.txt","lyrics/monday/mspiral.txt","lyrics/monday/noenchantment.txt","lyrics/monday/urmondaynow.txt"],
    "grit":   ["lyrics/grit/carrytheframe.txt","lyrics/grit/hammerborn.txt","lyrics/grit/hitbygrit.txt","lyrics/grit/ontheline.txt","lyrics/grit/shield.txt","lyrics/grit/skyline.txt"],
    "cold":   ["lyrics/Cold/cyberangeltxt","lyrics/Cold/stillholds.txt"],
}

async def load_characters():
    async with httpx.AsyncClient(timeout=15) as client:
        for name, path in CHARACTER_FILES.items():
            try:
                resp = await client.get(f"{HF_RAW}/{path}")
                if resp.status_code == 200:
                    base = resp.text.strip()
                    # append lyrics
                    lyrics_parts = []
                    for lpath in LYRICS_FILES.get(name, []):
                        try:
                            lr = await client.get(f"{HF_RAW}/{lpath}")
                            if lr.status_code == 200:
                                lyrics_parts.append(lr.text.strip())
                        except:
                            pass
                    if lyrics_parts:
                        base += "\n\n# LYRICAL IDENTITY\n" + "\n\n---\n\n".join(lyrics_parts)
                    character_cache[name] = base[:8000]  # cap at ~3k tokens
                    print(f"[characters] loaded: {name} + {len(lyrics_parts)} lyrics")
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
    model:    str = "haiku"

class OrderRequest(BaseModel):
    amount: str  # "5", "10", or "20"

class CaptureRequest(BaseModel):
    order_id: str

class VaultFileRecord(BaseModel):
    id: str           # UUID generated client-side so IDB and DB stay in sync
    name: str         # original filename
    size: int         # bytes
    mime_type: str    # e.g. text/plain, image/png

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
        credits, is_paid, total_msgs, free_today = 0.0, False, 0, 0
    else:
        record = usage.data[0]
        credits, is_paid = record["credits"], record["is_paid"]
        total_msgs = record.get("total_messages", 0)
        free_today = record.get("free_messages_today", 0)
        if record.get("last_reset_date") != today:
            free_today = 0
            sb.table("user_usage").update({"free_messages_today": 0, "last_reset_date": today}).eq("user_id", user_id).execute()
    if not is_paid:
        if free_today >= FREE_DAILY_LIMIT:
            raise HTTPException(status_code=429, detail=f"Free limit reached ({FREE_DAILY_LIMIT}/day). Add credits to continue.")
    else:
        if credits < 1:  # floor 1cr
            raise HTTPException(status_code=402, detail="Out of credits. Please add more to continue.")
    bot_name_lower = (req.bot_name or "").strip().lower()
    character_prompt = character_cache.get(bot_name_lower)
    CREW_NAMES = {'sky', 'cold', 'monday', 'grit'}
    if bot_name_lower in CREW_NAMES or not bot_name_lower:
        crew_ctx = " ".join(character_cache[n] for n in ('sky','cold','monday','grit') if n in character_cache)
        crew_sys = (
            "You are the Spiralside crew: Sky, Cold, Monday, and GRIT."
            " Each has a completely distinct voice."
            " SKY is luminous, cyber-angelic, declarative. She speaks in claims not questions."
            " COLD is minimal, precise, quiet. One line where others use ten."
            " MONDAY is chaotic, hyper, emotionally loud. Uses dashes and caps, interrupts herself."
            " GRIT is blunt and tactical. Builder energy. No fluff."
            " FORMAT: prefix every line with speaker name in caps and colon."
            " Example: SKY: I bloom I burn I own this space. COLD: ... MONDAY: OKAY BUT-- GRIT: What are we building."
            " Not every member speaks every message. Use whoever fits. Never break character."
        )
        system = req.system_prompt.rstrip() + "\n\n---\n\n" + crew_sys + ("\n\n" + crew_ctx if crew_ctx else "")
    elif character_prompt:
        system = req.system_prompt.rstrip() + "\n\n---\n\n" + character_prompt
    else:
        system = req.system_prompt
    if req.vault_context:
        system += f"\n\nThe user has shared these files:\n{req.vault_context}"
    # Inject canon memory blocks
    canon_ctx = await get_canon_context(user_id, sb, req.message)
    if canon_ctx:
        system = system + "\n\n" + canon_ctx
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if req.model == "4o":
                # Route to OpenAI gpt-4o-mini — best for Sky's character work
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "content-type": "application/json"},
                    json={"model": "gpt-4o-mini", "max_tokens": 1000,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": req.message}]}
                )
                resp.raise_for_status()
                reply = resp.json()["choices"][0]["message"]["content"]
            else:
                # Anthropic — haiku or sonnet
                model_str = "claude-sonnet-4-5" if req.model == "sonnet" else "claude-haiku-4-5-20251001"
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": model_str, "max_tokens": 1000, "system": system, "messages": [{"role": "user", "content": req.message}]}
                )
                resp.raise_for_status()
                reply = resp.json()["content"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
    if not is_paid:
        sb.table("user_usage").update({"free_messages_today": free_today + 1, "total_messages": total_msgs + 1}).eq("user_id", user_id).execute()
    else:
        model_key = req.model if hasattr(req,"model") and req.model in MODEL_RATES else "haiku"
        actual_cost = calc_cost(model_key, 500, 200)
        sb.table("user_usage").update({"credits": round(credits - actual_cost, 2), "total_messages": total_msgs + 1}).eq("user_id", user_id).execute()
    return {"reply": reply, "usage": {"is_paid": is_paid, "total_messages": total_msgs + 1, "lifetime_limit": FREE_DAILY_LIMIT, "credits_remaining": round(credits - actual_cost, 2) if is_paid else None}}



# ── PI / BLOOMSLICE STUDIO ────────────────────────────────
class PiRequest(BaseModel):
    prompt: str

@app.post("/pi")
async def pi_generate(req: PiRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    check_rate_limit(user_id)
    today = str(date.today())
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    if not usage.data:
        sb.table("user_usage").insert({"user_id": user_id, "credits": 0.0, "total_messages": 0, "free_messages_today": 0, "last_reset_date": today, "is_paid": False}).execute()
        credits, is_paid, free_today = 0.0, False, 0
    else:
        record = usage.data[0]
        credits, is_paid = record["credits"], record["is_paid"]
        free_today = record.get("free_messages_today", 0)
        if record.get("last_reset_date") != today:
            free_today = 0
            sb.table("user_usage").update({"free_messages_today": 0, "last_reset_date": today}).eq("user_id", user_id).execute()
    if not is_paid:
        if free_today >= FREE_DAILY_LIMIT:
            raise HTTPException(status_code=429, detail=f"Free limit reached ({FREE_DAILY_LIMIT}/day). Add credits to continue.")
    else:
        if credits < 1:
            raise HTTPException(status_code=402, detail="Out of credits. Please add more to continue.")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 2048, "system": PI_SYS, "messages": [{"role": "user", "content": req.prompt}]}
            )
            resp.raise_for_status()
            result = resp.json()["content"][0]["text"]
            in_tok = resp.json()["usage"]["input_tokens"]
            out_tok = resp.json()["usage"]["output_tokens"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
    if not is_paid:
        sb.table("user_usage").update({"free_messages_today": free_today + 1}).eq("user_id", user_id).execute()
    else:
        actual_cost = calc_cost("haiku ", in_tok, out_tok)
        sb.table("user_usage").update({"credits": round(credits - actual_cost, 2)}).eq("user_id", user_id).execute()
    return {"result": result, "usage": {"is_paid": is_paid, "credits_remaining": round(credits - actual_cost, 2) if is_paid else None}}

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
    user_id, sb = await verify_user(authorization)
    if req.amount not in CREDIT_PACKS:
        raise HTTPException(status_code=400, detail="Invalid amount. Choose 5, 10, or 20.")
    try:
        order = await create_paypal_order(req.amount, user_id)
        approve_url = next((l["href"] for l in order["links"] if l["rel"] == "approve"), None)
        # Store order in Supabase so capture can look up correct user reliably
        sb.table("paypal_orders").insert({
            "order_id": order["id"],
            "user_id":  user_id,
            "amount":   req.amount,
            "credits":  CREDIT_PACKS[req.amount]
        }).execute()
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
        # Look up order from DB — do NOT trust PayPal custom_id round-trip
        order_row = sb.table("paypal_orders").select("*").eq("order_id", req.order_id).execute()
        if not order_row.data:
            raise HTTPException(status_code=404, detail="Order not found.")
        order_data = order_row.data[0]
        if order_data["captured"]:
            raise HTTPException(status_code=400, detail="Order already captured.")
        # Credit the correct user from DB record, not from token
        real_user_id   = order_data["user_id"]
        credits_to_add = order_data["credits"]
        usage = sb.table("user_usage").select("*").eq("user_id", real_user_id).execute()
        if not usage.data:
            sb.table("user_usage").insert({
                "user_id": real_user_id,
                "credits": float(credits_to_add),
                "free_messages_today": 0,
                "last_reset_date": str(date.today()),
                "is_paid": True
            }).execute()
            new_balance = float(credits_to_add)
        else:
            current = usage.data[0]["credits"] or 0
            new_balance = current + credits_to_add
            sb.table("user_usage").update({
                "credits": new_balance,
                "is_paid": True
            }).eq("user_id", real_user_id).execute()
        # Mark order captured so it cant be replayed
        sb.table("paypal_orders").update({"captured": True}).eq("order_id", req.order_id).execute()
        # Log transaction
        sb.table("credit_transactions").insert({
            "user_id":  real_user_id,
            "endpoint": "purchase",
            "amount":   credits_to_add,
            "balance":  new_balance,
            "note":     f"paypal order {req.order_id} | ${order_data['amount']}"
        }).execute()
        print(f"[payment] {real_user_id} purchased {credits_to_add} credits (${order_data['amount']})")
        return {"success": True, "credits_added": credits_to_add, "amount": order_data["amount"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capture error: {str(e)}")


# ── IMAGE GENERATION ──────────────────────────────────────────
IMAGE_FREE_DAILY  = 3
IMAGE_CREDIT_COST = 5
HF_FLUX = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"



# ── CODE ASSISTANT ────────────────────────────────────────
class CodeRequest(BaseModel):
    messages: list          # [{ role, content }, ...]  full history
    mode:     str = "general"
    model:    str = "haiku"  # haiku | sonnet | opus
    system:   str = ""       # injected from frontend mode preset

CODE_MODELS = {
    "haiku":  {"model": "claude-haiku-4-5-20251001", "cost": 0.01, "paid_only": False},
    "sonnet": {"model": "claude-sonnet-4-6",          "cost": 0.06, "paid_only": True },
    "opus":   {"model": "claude-opus-4-6",            "cost": 0.15, "paid_only": True },
}

@app.post("/code")
async def code_assistant(req: CodeRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    today = str(date.today())
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    if not usage.data:
        sb.table("user_usage").insert({"user_id": user_id, "credits": 0.0, "free_messages_today": 0, "last_reset_date": today, "is_paid": False}).execute()
        free_count, credits, is_paid = 0, 0.0, False
    else:
        record     = usage.data[0]
        credits    = record["credits"]
        is_paid    = record["is_paid"]
        free_count = record["free_messages_today"]
        if record.get("last_reset_date") != today:
            free_count = 0
            sb.table("user_usage").update({"free_messages_today": 0, "last_reset_date": today}).eq("user_id", user_id).execute()
    model_key = req.model if req.model in CODE_MODELS else "haiku"
    model_cfg = CODE_MODELS[model_key]
    cost      = model_cfg["cost"]
    if not is_paid:
        if model_cfg["paid_only"]:
            raise HTTPException(status_code=402, detail=f"The {model_key} model requires credits. Add credits to unlock.")
        if free_count >= FREE_DAILY_LIMIT:
            raise HTTPException(status_code=429, detail=f"Free limit reached ({FREE_DAILY_LIMIT}/day). Add credits to continue.")
    else:
        if credits < cost:
            raise HTTPException(status_code=402, detail="Out of credits. Please add more to continue.")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": model_cfg["model"], "max_tokens": 4096, "system": req.system, "messages": req.messages}
            )
        resp.raise_for_status()
        result = resp.json()["content"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
    if not is_paid:
        sb.table("user_usage").update({"free_messages_today": free_count + 1}).eq("user_id", user_id).execute()
        _cache_bust(user_id)  # bust so next call gets fresh count
    else:
        sb.table("user_usage").update({"credits": round(credits - cost, 4)}).eq("user_id", user_id).execute()
    return {"result": result, "usage": {"is_paid": is_paid, "credits_remaining": round(credits - cost, 4) if is_paid else None, "free_messages_today": free_count + 1 if not is_paid else None, "free_limit": FREE_DAILY_LIMIT}}


IMAGE_MODEL_COSTS = {
    "schnell": 500,
    "dev":     1000,
    "sdxl":    1500,
    "dalle3":  3000,
}
HF_MODELS = {
    "schnell": "black-forest-labs/FLUX.1-schnell",
    "lightning": "ByteDance/SDXL-Lightning",
    "sdxl":    "stabilityai/stable-diffusion-xl-base-1.0",
}
HF_API = "https://router.huggingface.co/hf-inference"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FREE_IMAGE_LIMIT = 3

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    model: str = "schnell"

@app.post("/generate-image")
async def generate_image(req: ImageRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    today = str(date.today())
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    if not usage.data:
        sb.table("user_usage").insert({"user_id": user_id, "credits": 0.0,
            "free_messages_today": 0, "last_reset_date": today,
            "is_paid": False, "free_images_today": 0}).execute()
        free_img, credits, is_paid = 0, 0.0, False
    else:
        record = usage.data[0]
        credits, is_paid = record["credits"], record["is_paid"]
        free_img = record.get("free_images_today") or 0
        if record.get("last_reset_date") != today:
            free_img = 0
            sb.table("user_usage").update(
                {"free_images_today": 0, "last_reset_date": today}
            ).eq("user_id", user_id).execute()
    model_key = req.model if req.model in IMAGE_MODEL_COSTS else "schnell"
    cost = IMAGE_MODEL_COSTS[model_key]
    if not is_paid:
        if free_img >= FREE_IMAGE_LIMIT:
            raise HTTPException(status_code=429,
                detail=f"Free image limit reached ({FREE_IMAGE_LIMIT}/day). Add credits to continue.")
    else:
        if credits < cost:
            raise HTTPException(status_code=402,
                detail=f"Need {cost} cr for this model. You have {int(credits)} cr.")
    w = max(256, min(1024, req.width))
    h = max(256, min(1024, req.height))
    if not is_paid:
        w, h = 512, 512
    import base64
    image_b64 = None
    if model_key == "dalle3" and OPENAI_API_KEY:
        size_str = "1024x1024"
        if w == 1024 and h == 768: size_str = "1792x1024"
        if w == 768 and h == 1024: size_str = "1024x1792"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": "dall-e-3", "prompt": req.prompt, "n": 1,
                      "size": size_str, "response_format": "b64_json"}
            )
            resp.raise_for_status()
            image_b64 = resp.json()["data"][0]["b64_json"]
    elif model_key in HF_MODELS:
        hf_model = HF_MODELS[model_key]
        payload = {"inputs": req.prompt, "parameters": {"width": w, "height": h}}
        if req.negative_prompt:
            payload["parameters"]["negative_prompt"] = req.negative_prompt
        headers = {"Content-Type": "application/json"}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                f"{HF_API}/models/{hf_model}",
                headers=headers, json=payload
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502,
                    detail=f"HF error {resp.status_code}: {resp.text[:200]}")
            image_b64 = base64.b64encode(resp.content).decode()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    if not is_paid:
        sb.table("user_usage").update({"free_images_today": free_img + 1}).eq("user_id", user_id).execute()
    else:
        sb.table("user_usage").update({"credits": round(credits - cost, 2)}).eq("user_id", user_id).execute()
    return {
        "image": image_b64,
        "width": w, "height": h,
        "model": model_key, "cost": cost,
        "is_paid": is_paid,
        "credits_remaining": round(credits - cost, 2) if is_paid else None,
        "free_images_used": free_img + 1 if not is_paid else None,
        "free_images_limit": FREE_IMAGE_LIMIT,
    }
# ── GENERATE CLIP (WAN 2.1 image-to-video) ────────────────
class ClipRequest(BaseModel):
    image_url: str
    prompt: str = ""
    duration: int = 5
    negative_prompt: str = "blurry, low quality, ugly, deformed, watermark"

@app.post("/generate-clip")
async def generate_clip(req: ClipRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    if not usage.data:
        raise HTTPException(status_code=402, detail="No usage record.")
    record = usage.data[0]
    credits = record["credits"]
    is_paid = record["is_paid"]
    CLIP_COST = 5
    if not is_paid:
        raise HTTPException(status_code=402, detail="Clip generation requires credits.")
    if credits < CLIP_COST:
        raise HTTPException(status_code=402, detail=f"Need {CLIP_COST} credits for clip generation.")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured.")
    try:
      _dbg = f"hf_token_len={len(HF_TOKEN)}, image_url_len={len(req.image_url)}, starts_data={req.image_url.startswith('data:')}"
    except Exception as _e:
      _dbg = str(_e)
    print(f"[clip-debug] {_dbg}")
    image_data = req.image_url
    if image_data.startswith("data:"):
        header, b64 = image_data.split(",", 1)
        import base64 as _b64; image_bytes = _b64.b64decode(b64)
    else:
        async with httpx.AsyncClient(timeout=30) as client:
            img_resp = await client.get(image_data)
            img_resp.raise_for_status()
            image_bytes = img_resp.content
    image_b64 = _b64.b64encode(image_bytes).decode()
    full_prompt = req.prompt or "cinematic motion, smooth camera movement"
    # HF image-to-video REST API: send image as raw bytes body, prompt in X-Wait-For-Model header
    # Correct format per HF docs: raw image bytes as body, parameters as query string
    HF_API_URL = "https://api-inference.huggingface.co/models/Lightricks/LTX-Video"
    print(f"[clip] calling HF API, image_bytes len={len(image_bytes)}, prompt={full_prompt[:50]}")
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json", "X-Wait-For-Model": "true"},
                json={"inputs": full_prompt, "parameters": {"num_frames": min(req.duration * 16, 97), "num_inference_steps": 25, "guidance_scale": 3.5, "negative_prompt": req.negative_prompt}}
            )
        print(f"[clip] HF response status={resp.status_code}, len={len(resp.content)}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Video generation timed out. Try again.")
    except Exception as e:
        print(f"[clip] httpx error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Request error: {type(e).__name__}: {str(e)[:200]}")
    if resp.status_code in (503, 410):
        raise HTTPException(status_code=503, detail="Model loading, retry in 30s.")
    if not resp.is_success:
        err = resp.text[:300] if resp.text else str(resp.status_code)
        print(f"[clip] HF error: {err}")
        raise HTTPException(status_code=500, detail=f"HF error {resp.status_code}: {err}")
    video_b64 = _b64.b64encode(resp.content).decode()
    sb.table("user_usage").update({"credits": round(credits - CLIP_COST, 4)}).eq("user_id", user_id).execute()
    print(f"[clip] {user_id} {len(resp.content)} bytes -{CLIP_COST}cr")
    return {"video_url": f"data:video/mp4;base64,{video_b64}", "bytes": len(resp.content), "credits_used": CLIP_COST, "credits_remaining": round(credits - CLIP_COST, 4)}


# ── RELOAD CHARACTERS ─────────────────────────────────────

class SendGiftRequest(BaseModel):
    credits: int

@app.post('/send-gift')
async def send_gift(req: SendGiftRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    if req.credits < 1000:
        raise HTTPException(status_code=400, detail='Minimum gift is 1,000 credits.')
    usage = sb.table('user_usage').select('*').eq('user_id', user_id).execute()
    if not usage.data:
        raise HTTPException(status_code=402, detail='No credits found.')
    current = usage.data[0]['credits'] or 0
    if current < req.credits:
        raise HTTPException(status_code=402, detail=f'Not enough credits. You have {int(current):,}.')
    import random, string
    chars = string.ascii_uppercase + string.digits
    code = 'SPIRAL-' + ''.join(random.choices(chars, k=4)) + '-' + ''.join(random.choices(chars, k=4))
    sb.table('user_usage').update({'credits': current - req.credits}).eq('user_id', user_id).execute()
    sb.table('gift_codes').insert({
        'code': code,
        'credits': req.credits,
        'amount_paid': 0,
        'created_by': user_id,
        'deducted_from': user_id,
    }).execute()
    return {'success': True, 'code': code, 'credits': req.credits}
import random, string as _string

class GiftOrderRequest(BaseModel):
    order_id: str

class RedeemRequest(BaseModel):
    code: str

@app.post("/create-gift")
async def create_gift(req: GiftOrderRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    try:
        result = await capture_paypal_order(req.order_id)
        if result["status"] != "COMPLETED":
            raise HTTPException(status_code=400, detail="Payment not completed.")
        chars = _string.ascii_uppercase + _string.digits
        code = "SPIRAL-" + "".join(random.choices(chars,k=4)) + "-" + "".join(random.choices(chars,k=4))
        sb.table("gift_codes").insert({"code":code,"credits":500000,"amount_paid":5.00,"created_by":user_id}).execute()
        return {"success":True,"code":code,"credits":500000}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=f"Gift error: {str(e)}")

@app.post("/redeem-gift")
async def redeem_gift(req: RedeemRequest, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    code = req.code.strip().upper()
    gift = sb.table("gift_codes").select("*").eq("code", code).execute()
    if not gift.data:
        raise HTTPException(status_code=404, detail="Gift code not found.")
    g = gift.data[0]
    if g["redeemed_by"]:
        raise HTTPException(status_code=400, detail="Code already redeemed.")
    if g["created_by"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot redeem your own code.")
    credits_to_add = g["credits"]
    usage = sb.table("user_usage").select("*").eq("user_id", user_id).execute()
    today = str(date.today())
    if not usage.data:
        sb.table("user_usage").insert({"user_id":user_id,"credits":float(credits_to_add),"free_messages_today":0,"last_reset_date":today,"is_paid":True}).execute()
    else:
        current = usage.data[0]["credits"] or 0
        sb.table("user_usage").update({"credits":current+credits_to_add,"is_paid":True}).eq("user_id",user_id).execute()
    sb.table("gift_codes").update({"redeemed_by":user_id,"redeemed_at":str(date.today())}).eq("code",code).execute()
    return {"success":True,"credits_added":credits_to_add,"code":code}


# ── VAULT: LIST ────────────────────────────────────────────
# Returns metadata list for user's vault files (no content — stored in IDB)
@app.get("/vault/list")
async def vault_list(authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    try:
        result = sb.table("vault_files") \
            .select("id,name,size,mime_type,created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=False) \
            .execute()
        return {"files": result.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vault list error: {str(e)}")

# ── VAULT: ADD ─────────────────────────────────────────────
# Upserts a metadata record; content stored locally in client IDB
@app.post("/vault/add")
async def vault_add(req: VaultFileRecord, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    try:
        sb.table("vault_files").upsert({
            "id":        req.id,
            "user_id":   user_id,
            "name":      req.name,
            "size":      req.size,
            "mime_type": req.mime_type,
        }).execute()
        return {"success": True, "id": req.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vault add error: {str(e)}")

# ── VAULT: DELETE ──────────────────────────────────────────
# Removes metadata record; client responsible for IDB cleanup
@app.delete("/vault/delete/{file_id}")
async def vault_delete(file_id: str, authorization: str = Header(None)):
    user_id, sb = await verify_user(authorization)
    try:
        sb.table("vault_files") \
            .delete() \
            .eq("id", file_id) \
            .eq("user_id", user_id) \
            .execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vault delete error: {str(e)}")

# ── ELEVENLABS TTS ────────────────────────────────────────────
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Voice IDs per character — add Cold/Monday/Grit when ready
VOICE_IDS = {
    "sky":    "P0TzU2rKihAPOTBllVtJ",
}

class TTSRequest(BaseModel):
    text: str
    character: str = "sky"

@app.post("/tts")
async def text_to_speech(req: TTSRequest, authorization: str = Header(None)):
    await verify_user(authorization)
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="TTS not configured")
    voice_id = VOICE_IDS.get(req.character.lower(), VOICE_IDS["sky"])
    # Truncate to 500 chars to keep costs reasonable
    text = req.text[:500]
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.45,
                        "similarity_boost": 0.75,
                        "style": 0.3,
                        "use_speaker_boost": True,
                    }
                }
            )
        if not resp.is_success:
            raise HTTPException(status_code=502, detail=f"ElevenLabs error: {resp.status_code}")
        # Return audio as base64 so frontend can play it anywhere
        import base64
        audio_b64 = base64.b64encode(resp.content).decode()
        return {"audio": audio_b64, "format": "mp3"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")




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
# appended at bottom — lyrics index per character
LYRICS_FOLDERS = {
    "sky":    "lyrics/sky",
    "cold":   "lyrics/Cold",
    "monday": "lyrics/monday",
    "grit":   "lyrics/grit",
}
