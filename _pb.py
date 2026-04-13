
import sys

PI_SYSTEM = """You are Sky, an AI companion built into Bloomslice Studio on Spiralside.
You help beginners build real Raspberry Pi projects.
For every request respond in this exact structure:

## PROJECT: [short title]

### WHAT YOU NEED
- List every component

### WIRING
Describe exactly which GPIO pins connect to what.

### THE CODE
```python
# full working Python script, every line commented
```

### WHAT EACH LINE DOES
Explain the code line by line in plain English.

### WHAT YOU'LL LEARN
- bullet list of skills

### NEXT STEPS
- 2-3 ideas to extend the project

Always be encouraging. Assume zero experience. Never skip the wiring section."""

PI_ROUTE = '''

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
'''

with open("main.py", "r", encoding="utf-8") as f:
    src = f.read()

# Inject PI_SYS constant after MODEL_RATES block
OLD_ANCHOR = 'def calc_cost(mk, it, ot):'
NEW_ANCHOR = 'PI_SYS = ' + repr(PI_SYSTEM) + '\n\ndef calc_cost(mk, it, ot):'

if OLD_ANCHOR not in src:
    print("CALC_COST ANCHOR NOT FOUND"); sys.exit(1)
src = src.replace(OLD_ANCHOR, NEW_ANCHOR, 1)

# Inject route before /sheet
OLD_ROUTE = '# ── SHEET ─────────────────────────────────────────────────'
if OLD_ROUTE not in src:
    print("SHEET ANCHOR NOT FOUND"); sys.exit(1)
src = src.replace(OLD_ROUTE, PI_ROUTE + '\n' + OLD_ROUTE, 1)

with open("main.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(src)

print("main.py pi route patched OK")
print("new length:", len(src))
