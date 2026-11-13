"""Smoke-test a Cloudflare chat model and print raw response shape."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from kgrag import config


def main() -> None:
    config.load_env()
    account_id = config.cloudflare_account_id()
    api_token = config.cloudflare_api_token()
    base_url = config.cloudflare_base_url()
    if not account_id or not api_token or not base_url:
        raise SystemExit("Missing Cloudflare credentials")

    model = "@cf/google/gemma-sea-lion-v4-27b-it"
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON with key 'sentences' containing one Vietnamese sentence."},
            {"role": "user", "content": "Say hello in Vietnamese"},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "seed": 42,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    print("status:", response.status_code)
    data = response.json()
    print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
