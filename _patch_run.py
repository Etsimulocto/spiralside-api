FILE = r"C:/Users/quart/spiralside-api/main.py"
import sys

with open(FILE, encoding="utf-8") as f:
    src = f.read()

# Find reload endpoint anchor
idx = src.find("/reload-characters")
if idx < 0:
    print("MISS: /reload-characters")
    sys.exit(1)

# Find the decorator line before it
dec_start = src.rfind("@app.post", 0, idx)
print("inserting before:", repr(src[dec_start:dec_start+40]))

new_endpoint = '''@app.post("/create-storage-order")
async def create_storage_order(authorization: str = Header(None)):
    user_id, _ = await verify_user(authorization)
    try:
        order = await create_paypal_order("2", user_id)
        approve_url = next((l["href"] for l in order["links"] if l["rel"] == "approve"), None)
        return {"order_id": order["id"], "approve_url": approve_url, "plan": "archive"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PayPal error: {str(e)}")

'''

src = src[:dec_start] + new_endpoint + src[dec_start:]
print("OK: /create-storage-order inserted")

# Handle storage activation in capture-order
old_print = 'print(f"[payment] {user_id} purchased {credits_to_add} credits (${amount})")'
if old_print in src:
    new_print = (
        'print(f"[payment] {user_id} purchased {credits_to_add} credits (${amount})")\n'
        '        if amount == "2":\n'
        '            from datetime import timedelta, datetime as dt\n'
        '            expires = dt.utcnow() + timedelta(days=30)\n'
        '            sb.table("user_usage").update({\n'
        '                "storage_plan": "archive",\n'
        '                "storage_expires_at": expires.isoformat()\n'
        '            }).eq("user_id", user_id).execute()\n'
        '            print(f"[storage] archive plan activated for {user_id}")'
    )
    src = src.replace(old_print, new_print)
    print("OK: storage activation on capture")
else:
    print("MISS: capture print — skipping")

with open(FILE, "w", encoding="utf-8") as f:
    f.write(src)
print("DONE")