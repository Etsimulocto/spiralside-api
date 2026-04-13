
import sys

with open("js/app/views/pi.js", "r", encoding="utf-8") as f:
    src = f.read()

OLD = "    const resp = await fetch('https://qfawusrelwthxabfbglg.supabase.co/functions/v1/pi-generate', {"
NEW = "    const rail = state.RAIL || 'https://web-production-4e6f3.up.railway.app';\n    const resp = await fetch(rail + '/pi', {"

if OLD not in src:
    print("ANCHOR NOT FOUND"); sys.exit(1)

src = src.replace(OLD, NEW, 1)

with open("js/app/views/pi.js", "w", encoding="utf-8", newline="\n") as f:
    f.write(src)

print("pi.js reverted to railway OK")
