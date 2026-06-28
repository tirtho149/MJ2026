"""Bangla-aware text preprocessing — the 13-step pipeline from the NLP notebook.

The notebook depended on ``bnlp-toolkit`` for the Unicode-aware tokenizer and the
Bengali stopword list.  Installing bnlp into the pinned vLLM venv risks pulling a
conflicting torch, so this module makes bnlp **optional**:

  * if ``bnlp`` imports cleanly, it is used (identical behaviour to the notebook);
  * otherwise a pure-Python fallback reproduces the documented BasicTokenizer
    logic (split on any ``unicodedata.category() == 'P*'`` or ASCII punctuation)
    and ships a built-in Bengali stopword set.

Either way the public API is the same: ``BanglaEmailPreprocessor().preprocess(s)``.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# ── Optional bnlp backend ─────────────────────────────────────────────────────
try:
    from bnlp import BasicTokenizer, CleanText
    from bnlp.corpus import BengaliCorpus
    _HAS_BNLP = True
except Exception:                                   # pragma: no cover - env dependent
    _HAS_BNLP = False


# A compact built-in Bengali stopword list for the no-bnlp fallback.
_FALLBACK_STOPWORDS = {
    "এই", "ও", "এবং", "আর", "কিন্তু", "তবে", "যে", "যা", "যার", "যাকে", "যিনি",
    "তিনি", "তার", "তাকে", "তাদের", "আমি", "আমার", "আমাকে", "আমরা", "আমাদের",
    "তুমি", "তোমার", "তোমাকে", "আপনি", "আপনার", "আপনাকে", "সে", "তা", "এটা",
    "ইহা", "ওই", "এ", "একটি", "একটা", "এক", "করে", "করা", "করার", "করুন",
    "হয়", "হবে", "হয়েছে", "ছিল", "আছে", "নেই", "না", "জন্য", "থেকে", "দিয়ে",
    "সাথে", "মধ্যে", "উপর", "নিচে", "পরে", "আগে", "মত", "মতো", "যখন", "তখন",
    "এখন", "তাই", "অথবা", "বা", "যদি", "তাহলে", "কি", "কী", "কেন", "কীভাবে",
    "কোন", "কোনো", "সব", "সকল", "প্রতি", "টি", "টা", "খুব", "অনেক", "আরও",
}


class BanglaEmailPreprocessor:
    """Full Bangla NLP preprocessing pipeline — 13 steps (see module docstring)."""

    HTML_TAG_RE     = re.compile(r"<[^>]+>")
    MULTI_SPACE_RE  = re.compile(r"\s+")
    URL_RE          = re.compile(r"(https?://\S+|www\.\S+)")
    EMAIL_RE        = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    EMOJI_RE        = re.compile(
        "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F000-\U0001F0FF"
        "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "\U0000FE00-\U0000FE0F" "]+",
        flags=re.UNICODE,
    )
    BANGLA_DIGIT_RE = re.compile(r"[০-৯]")
    ASCII_DIGIT_RE  = re.compile(r"[0-9]")
    BN_DIGIT_MAP    = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

    def __init__(
        self,
        remove_stopwords:    bool = True,
        normalize_digits:    bool = True,
        remove_digits:       bool = False,
        remove_english_text: bool = False,
        min_token_len:       int  = 2,
        verbose:             bool = True,
    ):
        self.remove_stopwords    = remove_stopwords
        self.normalize_digits    = normalize_digits
        self.remove_digits       = remove_digits
        self.remove_english_text = remove_english_text
        self.min_token_len       = min_token_len

        if _HAS_BNLP:
            self._cleaner = CleanText(
                fix_unicode=True, unicode_norm=True, unicode_norm_form="NFKC",
                remove_url=True, remove_email=True, remove_emoji=True, remove_punct=True,
                replace_with_url="", replace_with_email="", replace_with_punct=" ",
            )
            self._tokenizer = BasicTokenizer()
            self._stopwords = set(BengaliCorpus.stopwords)
            self._backend = "bnlp BasicTokenizer + CleanText"
        else:
            self._cleaner = None
            self._tokenizer = None
            self._stopwords = set(_FALLBACK_STOPWORDS)
            self._backend = "pure-Python fallback (unicodedata)"

        if verbose:
            print(f"  🔤 Tokenizer : {self._backend}")
            print(f"  📚 Stopwords : {len(self._stopwords)} Bengali words")
            print(f"  📋 Pipeline  : 13 steps")

    # ── Step implementations ──────────────────────────────────────────────────
    def _remove_html(self, text):  return self.HTML_TAG_RE.sub(" ", text)

    def _clean_text(self, text):
        """Steps 3-8: unicode fix + NFKC + url/email/emoji/punct removal."""
        if self._cleaner is not None:
            return self._cleaner(text)
        # fallback: replicate the same intent without bnlp
        text = unicodedata.normalize("NFKC", text)
        text = self.URL_RE.sub(" ", text)
        text = self.EMAIL_RE.sub(" ", text)
        text = self.EMOJI_RE.sub(" ", text)
        # strip punctuation: any Unicode 'P*' category char -> space
        text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
        return text

    def _handle_digits(self, text):
        if self.remove_digits:
            text = self.BANGLA_DIGIT_RE.sub(" ", text)
            text = self.ASCII_DIGIT_RE.sub(" ", text)
        elif self.normalize_digits:
            text = text.translate(self.BN_DIGIT_MAP)
        return text

    def _tokenize(self, text):
        """Step 11: Unicode + punctuation-aware tokenization.

        With bnlp this is BasicTokenizer; without it we split on whitespace then
        peel any leading/trailing punctuation off each token — the same
        ``unicodedata.category().startswith('P')`` rule the notebook documents.
        """
        if self._tokenizer is not None:
            return self._tokenizer.tokenize(text)

        out = []
        for word in text.split():
            buf = ""
            for ch in word:
                if unicodedata.category(ch).startswith("P") or (33 <= ord(ch) <= 47):
                    if buf:
                        out.append(buf); buf = ""
                    out.append(ch)
                else:
                    buf += ch
            if buf:
                out.append(buf)
        return out

    def _filter_tokens(self, tokens):
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        tokens = [t for t in tokens if len(t) >= self.min_token_len]
        if self.remove_english_text:
            tokens = [t for t in tokens if not re.match(r"^[a-zA-Z0-9@._%-]+$", t)]
        return tokens

    # ── Main entry point ──────────────────────────────────────────────────────
    def preprocess(self, text) -> str:
        """Run all 13 steps on one Bangla email string."""
        if pd.isna(text) or str(text).strip() == "":
            return ""
        text = str(text)
        text = self._remove_html(text)               # 1-2
        text = self._clean_text(text)                # 3-8
        text = self._handle_digits(text)             # 9
        text = self.MULTI_SPACE_RE.sub(" ", text).strip()   # 10
        tokens = self._tokenize(text)                # 11
        tokens = self._filter_tokens(tokens)         # 12-13
        return " ".join(tokens)

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return series.apply(self.preprocess)
