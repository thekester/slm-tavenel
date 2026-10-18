"""Rule-based redaction of common personal identifiers.

Regular expressions catch structured identifiers (emails, IP addresses, phone
numbers, IBANs); they do not detect names or free-text personal information. The
privacy review recorded in the source registry remains required.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# International (+33 6 12 34 56 78), French (06 12 34 56 78) and NANP (555-123-4567) shapes.
PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d{2,4}(?![\w.])"
)
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b")

PLACEHOLDERS = {"email": "<EMAIL>", "ip": "<IP_ADDRESS>", "phone": "<PHONE>", "iban": "<IBAN>"}


def _public_ipv4(match: re.Match) -> bool:
    try:
        ip = ipaddress.IPv4Address(match.group(0))
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_unspecified or ip.is_reserved)


def _phone_like(match: re.Match) -> bool:
    digits = sum(c.isdigit() for c in match.group(0))
    return 9 <= digits <= 15


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts = {k: 0 for k in PLACEHOLDERS}

    def sub(pattern: re.Pattern, kind: str, text: str, accept=lambda m: True) -> str:
        def repl(m: re.Match) -> str:
            if not accept(m):
                return m.group(0)
            counts[kind] += 1
            return PLACEHOLDERS[kind]

        return pattern.sub(repl, text)

    text = sub(EMAIL, "email", text)
    text = sub(IBAN, "iban", text)
    text = sub(IPV4, "ip", text, _public_ipv4)
    text = sub(PHONE, "phone", text, _phone_like)
    return text, counts


def redact_doc(doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    text, counts = redact(doc["text"])
    return {**doc, "text": text}, counts
