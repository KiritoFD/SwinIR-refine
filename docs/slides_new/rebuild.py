import subprocess, os, sys, time

os.chdir(r"G:\RealSR\docs\slides_new")
SLIDEP = r"C:\Users\xy\.workbuddy-ai\binaries\node\versions\22.22.2-3\slidep.cmd"
PPTX = "RealSR_真实超分_论文汇报.pptx"
ORDER = ["01", "02", "03", "11", "05", "06", "07", "12", "14", "13", "08", "09", "10"]

def run(args, retries=3):
    for attempt in range(1, retries + 1):
        p = subprocess.run([SLIDEP] + args, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        if '"ok":true' in out:
            if '"failures":[]' not in out and '"failures"' in out:
                print("  !! failures present:", out.strip()[-300:])
                return False, out
            return True, out
        print(f"  attempt {attempt} FAILED: {out.strip()[:300]}")
        time.sleep(1.5)
    return False, out

# 1. create fresh blank deck
print("== create == ")
ok, out = run(["create", PPTX])
print("create ok:", ok)
if not ok:
    print(out[:400]); sys.exit(1)

# 2. upsert all pages
for i, name in enumerate(ORDER):
    ok, out = run(["upsert-dsl", PPTX, "--page-index", str(i), "--dsl-file", f"slides/{name}.slide"])
    print(f"page {i+1:02d} <- {name}.slide  ok={ok}")
    if not ok:
        print(out[:500]); sys.exit(1)

print("\n== done ==")
print(os.path.getsize(PPTX), "bytes")
