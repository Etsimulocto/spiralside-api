# phase_backend_cleanup.py
# ============================================================
# SPIRALSIDE-API - DEAD CODE CLEANUP (backend - extra careful)
# ============================================================
# 1. Removes the /pi-gpio endpoint (replaced by the local
#    parser in pi.js, never used in production)
# 2. Removes huggingface_hub from requirements.txt IF AND ONLY
#    IF main.py no longer imports anything huggingface-related
#    (the old HF video path was replaced by fal.ai)
# SAFETY: backs up main.py first, syntax-checks the result with
# py_compile, and auto-restores the backup if the check fails.
# Run from ~/spiralside-api:   python phase_backend_cleanup.py
# Then push NORMALLY - never --force on the backend.

import sys, os, shutil, py_compile

PATH = "main.py"

raw = open(PATH, encoding="utf-8").read()
had_crlf = "\r\n" in raw
src = raw.replace("\r\n", "\n")

# --- guard ------------------------------------------------------------
if "/pi-gpio" not in src:
    print("[1/2] /pi-gpio: already gone")
else:
    # backup before touching anything
    shutil.copy(PATH, PATH + ".bak_cleanup")
    print("      backup written: main.py.bak_cleanup")

    # locate the block: from the PiGpioRequest model (and its comment
    # line above, if present) to the next @app. route after the endpoint
    i0 = src.find("class PiGpioRequest")
    if i0 < 0:
        print("[1/2] FAIL: PiGpioRequest class not found - layout changed. Tell Claude.")
        sys.exit(1)
    # walk back to include the section comment line right above, if any
    line_start = src.rfind("\n", 0, i0) + 1
    prev_line_start = src.rfind("\n", 0, line_start - 1) + 1
    prev_line = src[prev_line_start:line_start]
    if "PI GPIO" in prev_line.upper():
        i0 = prev_line_start

    j = src.find('@app.post("/pi-gpio")', i0)
    if j < 0:
        print("[1/2] FAIL: /pi-gpio decorator not found after model. Tell Claude.")
        sys.exit(1)
    k = src.find("\n@app.", j + 10)
    if k < 0:
        # endpoint might be the last route - cut to end of file
        k = len(src) - 1
    removed = src[i0:k + 1]
    src = src[:i0] + src[k + 1:]
    print("[1/2] /pi-gpio: removed", len(removed), "chars")
    print("      cut starts:", repr(removed[:60]))
    print("      cut ends:  ", repr(removed[-60:]))

    out = src.replace("\n", "\r\n") if had_crlf else src
    open(PATH, "w", encoding="utf-8", newline="").write(out)

    # --- mandatory syntax check; restore backup on failure -----------
    try:
        py_compile.compile(PATH, doraise=True)
        print("      py_compile: OK")
    except py_compile.PyCompileError as e:
        shutil.copy(PATH + ".bak_cleanup", PATH)
        print("      py_compile FAILED - main.py RESTORED from backup. Tell Claude:")
        print(e)
        sys.exit(1)

# ------------------------------------------------------------
# 2. drop huggingface_hub from requirements if truly unused
# ------------------------------------------------------------
src_now = open(PATH, encoding="utf-8").read()
uses_hf = ("huggingface" in src_now) or ("hf_hub" in src_now) or ("InferenceClient" in src_now)
req_path = "requirements.txt"
if uses_hf:
    print("[2/2] huggingface_hub: main.py still references huggingface - KEEPING the dependency")
elif os.path.exists(req_path):
    req_raw = open(req_path, encoding="utf-8").read()
    lines = req_raw.replace("\r\n", "\n").split("\n")
    kept = [l for l in lines if "huggingface" not in l.lower()]
    if len(kept) == len(lines):
        print("[2/2] huggingface_hub: not in requirements.txt - nothing to do")
    else:
        open(req_path, "w", encoding="utf-8", newline="").write("\r\n".join(kept) if "\r\n" in req_raw else "\n".join(kept))
        print("[2/2] huggingface_hub: removed from requirements.txt (faster Railway builds)")
else:
    print("[2/2] requirements.txt not found - skipped")

print()
print("DONE. Delete main.py.bak_cleanup after Railway deploys clean.")
print('Now: git add . && git commit -m "cleanup: remove /pi-gpio + unused HF dependency" && git push origin main')
print("(plain push - NEVER --force on the backend)")
