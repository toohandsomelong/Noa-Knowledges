"""Test one batch verbalization call against Cloudflare to inspect response."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from kgrag import config
from kgrag.providers import CloudflareChat
from kgrag.verbalize import _SYSTEM_PROMPT_SINGLE, _render_walk
from kgrag.walks import load_walks


def main() -> None:
    config.load_env()
    walks = load_walks("artifacts/kg/bakeoff/sample_walks.jsonl")
    batch = walks[:1]

    system = _SYSTEM_PROMPT_SINGLE
    user = _render_walk(batch[0], 400) + "\n\nHãy viết lại thành một câu:"

    chat = CloudflareChat(
        model="@cf/zai-org/glm-4.7-flash",
        account_id=config.cloudflare_account_id(),
        api_token=config.cloudflare_api_token(),
        base_url=config.cloudflare_base_url(),
        chat_template_kwargs={"enable_thinking": False},
    )
    text = chat.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        num_predict=512,
        seed=42,
    )
    print(text)


if __name__ == "__main__":
    main()
