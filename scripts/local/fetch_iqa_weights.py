"""Fetch pyiqa's MUSIQ / MANIQA checkpoints on the Windows side.

github.com is unreachable from the training box, so the weights have to be
pulled here and scp'd over.  The connection is flaky (intermittent connect
timeouts), so this retries a lot and validates Content-Length -- an earlier
attempt "succeeded" with a 9-byte error page, which is worse than a clean
failure because it looks like it worked.
"""
import os
import sys
import time
import urllib.request

BASE = "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights"
FILES = {
    # MUSIQ, trained on KonIQ-10k -- this is what pyiqa.create_metric('musiq') wants
    "musiq_koniq_ckpt-e95806b9.pth": 100_000_000,
    # MANIQA.  pyiqa's `maniqa` default points at ckpt_koniq10k.pt, but that
    # asset does not exist in the v0.1-weights release (only PIPAL does), so the
    # default is a 404 and we have to use `maniqa-pipal`.
    "MANIQA_PIPAL-ae6d356b.pth": 500_000_000,
}
OUT = os.path.dirname(os.path.abspath(__file__))

PROXIES = [
    None,                      # direct
    "http://127.0.0.1:1956",   # local proxy
    "http://127.0.0.1:7890",
]


def try_download(url, dest, min_size):
    for attempt in range(1, 41):
        proxy = PROXIES[(attempt - 1) % len(PROXIES)]
        try:
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy})
                )
            else:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with opener.open(req, timeout=45) as r:
                data = r.read()
            if len(data) >= min_size:
                with open(dest, "wb") as f:
                    f.write(data)
                return len(data), proxy or "direct"
            print(f"    attempt {attempt:2d} proxy={proxy or 'direct':22s} "
                  f"got {len(data)} bytes (too small)", flush=True)
        except Exception as e:
            print(f"    attempt {attempt:2d} proxy={proxy or 'direct':22s} "
                  f"{type(e).__name__}: {str(e)[:70]}", flush=True)
        time.sleep(2)
    return None, None


for name, min_size in FILES.items():
    dest = os.path.join(OUT, name)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"{name}: already have {os.path.getsize(dest)} bytes")
        continue
    print(f"downloading {name} ...", flush=True)
    size, via = try_download(f"{BASE}/{name}", dest, min_size)
    if size:
        print(f"  OK  {size/1e6:.1f} MB via {via}")
    else:
        print(f"  FAILED {name}")
