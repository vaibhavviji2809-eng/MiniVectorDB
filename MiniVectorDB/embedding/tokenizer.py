"""Tokenization helpers."""

from __future__ import annotations

import re
from typing import List

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class Tokenizer:
    def tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower())

