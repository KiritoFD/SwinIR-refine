"""Download/load a VAE preset on the server (prints info)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/ds/realsr")
from diffusion.vae import load_vae  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="flux1-dev")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    vae, info = load_vae(args.vae, device=args.device)
    print("OK", info)


if __name__ == "__main__":
    main()
