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
# Only explicit phone layouts: international (+33 6 12 34 56 78), French national
# (06 12 34 56 78) and North American ((555) 123-4567, 555-123-4567). A generic
# "run of digit groups" pattern also matches dates, ISBNs, measurements and archive
# URLs, which removes legitimate numbers from the corpus.
_NB = r"(?<![\w/.,:+-])"
_NA = r"(?![\w/.,-]\d|\w)"
PHONE = re.compile(
    _NB + r"(?:"
    r"\+\d{1,3}[ .-]?(?:\(0\)[ .-]?)?\(?\d{1,4}\)?(?:[ .-]\d{2,4}){2,4}"
    r"|0[1-9](?:[ .-]\d{2}){4}"
    r"|\(\d{3}\)[ ]?\d{3}[ .-]\d{4}"
    r"|\d{3}[.-]\d{3}[.-]\d{4}"
    r")" + _NA
)
IBAN = re.compile(r"\b[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
# Dotted quads are far more often section numbers ("Digest 48.22.7.1") or versions than
# addresses in encyclopedic text, so an address is redacted only with a contextual cue
# before it or a port / prefix-length suffix after it.
_IP_CUE = re.compile(r"(?:\bIP|address|adresse|server|serveur|host|hôte|DNS|ping|proxy|gateway|passerelle)\W{0,3}(?:\w+\W{1,3}){0,3}$", re.IGNORECASE)
_IP_SUFFIX = re.compile(r"^(?::\d{1,5}\b|/\d{1,2}\b)")

PLACEHOLDERS = {"email": "<EMAIL>", "ip": "<IP_ADDRESS>", "phone": "<PHONE>", "iban": "<IBAN>"}


def _public_ipv4(match: re.Match) -> bool:
    try:
        ip = ipaddress.IPv4Address(match.group(0))
    except ValueError:
        return False
    if ip.is_private or ip.is_loopback or ip.is_unspecified or ip.is_reserved:
        return False
    text, start, end = match.string, match.start(), match.end()
    return bool(_IP_CUE.search(text[max(0, start - 40) : start]) or _IP_SUFFIX.match(text[end : end + 6]))


def _valid_iban(match: re.Match) -> bool:
    """ISO 13616 check: move the first four characters to the end, map letters to 10-35, mod 97 == 1."""
    s = "".join(match.group(0).split())
    if not 15 <= len(s) <= 34:
        return False
    digits = "".join(str(int(c, 36)) for c in s[4:] + s[:4])
    return int(digits) % 97 == 1


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
    text = sub(IBAN, "iban", text, _valid_iban)
    text = sub(IPV4, "ip", text, _public_ipv4)
    text = sub(PHONE, "phone", text, _phone_like)
    return text, counts


def redact_doc(doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    text, counts = redact(doc["text"])
    return {**doc, "text": text}, counts
