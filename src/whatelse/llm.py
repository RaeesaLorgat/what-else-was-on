"""Connection to a model deployed in Microsoft Foundry.

Uses the OpenAI SDK against the Foundry resource's /openai/v1/ route, so the
deployment name goes in the ``model`` field and no api-version is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class FoundrySettings:
    endpoint: str
    api_key: str
    model: str

    @property
    def base_url(self) -> str:
        return self.endpoint.rstrip("/") + "/openai/v1/"


def load_settings() -> FoundrySettings:
    """Read the three settings from .env or the environment, with clear errors."""
    load_dotenv()
    missing = [k for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_MODEL")
               if not os.getenv(k) or "<" in os.getenv(k, "")]
    if missing:
        raise RuntimeError(
            "Missing or placeholder values in .env: " + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in (see docs/SETUP.md)."
        )
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    # Accept a pasted URL that already ends in /openai/v1 or similar.
    for suffix in ("/openai/v1/", "/openai/v1", "/openai/", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    return FoundrySettings(endpoint=endpoint, api_key=os.environ["AZURE_OPENAI_API_KEY"],
                           model=os.environ["AZURE_OPENAI_MODEL"])


def make_client(settings: FoundrySettings):
    from openai import OpenAI

    return OpenAI(base_url=settings.base_url, api_key=settings.api_key)
