"""Text vectorization helpers for the active learning baseline."""

from __future__ import annotations

from typing import Iterable

try:  # pragma: no cover - exercised when sklearn is installed in the runtime.
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - fallback keeps local/offline tests runnable.
    TfidfVectorizer = None  # type: ignore[assignment]


if TfidfVectorizer is None:
    import math
    import re
    from collections import Counter, defaultdict


class SimpleTfidfVectorizer:
    """Minimal TF-IDF vectorizer suitable for multiclass text classification."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_: dict[str, int] = {}
        self.idf_: dict[str, float] = {}
        self.feature_names_: list[str] = []

    def fit(self, texts: Iterable[object]) -> "SimpleTfidfVectorizer":
        """Learn document frequencies and a compact vocabulary from raw texts."""

        documents = [self._extract_tokens(text) for text in texts]
        doc_frequency: dict[str, int] = defaultdict(int)

        for tokens in documents:
            for token in set(self._generate_ngrams(tokens)):
                doc_frequency[token] += 1

        sorted_features = sorted(doc_frequency.items(), key=lambda item: (-item[1], item[0]))
        limited_features = sorted_features[: self.max_features]

        self.feature_names_ = [feature for feature, _ in limited_features]
        self.vocabulary_ = {feature: index for index, feature in enumerate(self.feature_names_)}

        n_documents = len(documents)
        self.idf_ = {
            feature: math.log((1 + n_documents) / (1 + frequency)) + 1.0
            for feature, frequency in limited_features
        }
        return self

    def transform(self, texts: Iterable[object]) -> list[dict[str, float]]:
        """Transform texts into sparse TF-IDF feature dictionaries."""

        if not self.vocabulary_:
            raise ValueError("The vectorizer must be fitted before calling transform")

        return [self._vectorize(text) for text in texts]

    def fit_transform(self, texts: Iterable[object]) -> list[dict[str, float]]:
        """Fit the vectorizer and transform the same input in one pass."""

        return self.fit(texts).transform(texts)

    def _vectorize(self, text: object) -> dict[str, float]:
        """Convert one text into a normalized sparse tf-idf vector."""

        tokens = self._generate_ngrams(self._extract_tokens(text))
        if not tokens:
            return {}

        counts = Counter(token for token in tokens if token in self.vocabulary_)
        if not counts:
            return {}

        total = sum(counts.values())
        values: dict[str, float] = {}
        for token, count in counts.items():
            tf = count / total
            values[token] = tf * self.idf_.get(token, 1.0)

        norm = math.sqrt(sum(weight * weight for weight in values.values()))
        if norm > 0:
            values = {token: weight / norm for token, weight in values.items()}
        return values

    def _extract_tokens(self, text: object) -> list[str]:
        """Tokenize text into lowercase word tokens with whitespace safety."""

        if text is None:
            return []
        raw = str(text).strip().lower()
        if not raw:
            return []
        return re.findall(r"[a-z0-9']+", raw)

    def _generate_ngrams(self, tokens: list[str]) -> list[str]:
        """Expand a token list into the configured n-gram range."""

        if not tokens:
            return []

        min_n, max_n = self.ngram_range
        ngrams: list[str] = []
        for n in range(min_n, max_n + 1):
            if n <= 0 or n > len(tokens):
                continue
            if n == 1:
                ngrams.extend(tokens)
                continue
            ngrams.extend([" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)])
        return ngrams


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> object:
    """Build a compact TF-IDF vectorizer suitable for multiclass text classification."""

    if TfidfVectorizer is not None:
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )

    return SimpleTfidfVectorizer(max_features=max_features, ngram_range=ngram_range)