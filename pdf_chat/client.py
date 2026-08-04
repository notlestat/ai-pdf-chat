"""Anthropic client setup.

Import this module before anything that opens an HTTPS connection. The
truststore injection below has to happen first — see the comment on it.
"""

# Some antivirus products (Norton, Kaspersky, ESET) intercept HTTPS by issuing
# their own certificates from a locally installed root CA. Norton's CA declares
# `Basic Constraints` without marking the extension critical, which OpenSSL 3.x
# rejects outright, so every Python HTTPS call fails with CERTIFICATE_VERIFY_FAILED
# even though the connection is otherwise fine. Windows' own certificate stack is
# more lenient and accepts it. truststore routes verification through the OS
# instead of OpenSSL, which sidesteps the problem without weakening it: the chain
# is still verified, just by a different verifier.
import truststore

truststore.inject_into_ssl()

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Opus 5 is the default: this app answers questions about contracts and policies
# where a wrong answer is worse than an expensive one. Swap to claude-sonnet-5 to
# cut cost roughly 40% if your documents are simpler.
MODEL = "claude-opus-5"

# Required to reference an uploaded PDF by file_id rather than re-sending bytes.
FILES_BETA = "files-api-2025-04-14"


class MissingAPIKey(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not configured."""


def get_client() -> anthropic.Anthropic:
    """Build an Anthropic client, or explain clearly why we can't."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKey(
            "ANTHROPIC_API_KEY is not set.\n\n"
            "Copy .env.example to .env and add your key from "
            "https://console.anthropic.com/settings/keys"
        )
    return anthropic.Anthropic(api_key=api_key)
