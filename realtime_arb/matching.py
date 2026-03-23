from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .constants import GENERIC_CONTENT_TOKENS, STOP_WORDS
from .models import Event, MatchedPair
from .utils import ensure_aware

class CategoryClassifier:
    def __init__(self, categories: Sequence[Dict[str, Any]]) -> None:
        self.keyword_to_categories: Dict[str, List[str]] = defaultdict(list)
        for category in categories:
            for keyword in category["keywords"]:
                self.keyword_to_categories[keyword.lower()].append(category["name"])

    def classify(self, text: str) -> List[str]:
        lowered = text.lower()
        matched: set[str] = set()
        for keyword, categories in self.keyword_to_categories.items():
            if keyword in lowered:
                matched.update(categories)
        return sorted(matched)

class BaselinePythonVectorizer:
    def __init__(self) -> None:
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        for raw in text.lower().replace("-", " ").split():
            clean = "".join(ch for ch in raw if ch.isalnum())
            if len(clean) >= 2 and clean not in STOP_WORDS and not clean.isdigit():
                tokens.append(clean)
        return tokens

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        tokenized = [self.tokenize(doc) for doc in documents]
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))
        vocab = sorted(doc_freq.keys())
        self.vocabulary = {token: index for index, token in enumerate(vocab)}
        self.idf = [math.log((1.0 + len(documents)) / (1.0 + doc_freq[token])) + 1.0 for token in vocab]
        matrix = np.zeros((len(documents), len(vocab)), dtype=np.float32)
        for row_index, tokens in enumerate(tokenized):
            term_freq = Counter(tokens)
            for token, count in term_freq.items():
                matrix[row_index, self.vocabulary[token]] = count * self.idf[self.vocabulary[token]]
            norm = np.linalg.norm(matrix[row_index])
            if norm > 1e-12:
                matrix[row_index] /= norm
        return matrix

class FastStreamingMatcher:
    def __init__(self, similarity_threshold: float, classifier: CategoryClassifier, chunk_size: int = 512, n_features: int = 2 ** 16) -> None:
        self.similarity_threshold = similarity_threshold
        self.classifier = classifier
        self.chunk_size = chunk_size
        self.hash_features = n_features
        self.vectorizer = HashingVectorizer(
            lowercase=True,
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),
            norm=None,
            alternate_sign=False,
            n_features=n_features,
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b",
        )
        self.transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
        self.pm_matrix = None
        self.kalshi_matrix = None
        self.feature_dimension = n_features

    @staticmethod
    def _numeric_tokens(text: str) -> set[str]:
        tokens = set()
        for raw in re.findall(r"\d+(?:\.\d+)?", text.lower()):
            try:
                value = int(float(raw))
            except ValueError:
                tokens.add(raw)
                continue
            if 1900 <= value <= 2100:
                continue
            tokens.add(str(value))
        return tokens

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        tokens = set()
        for raw in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower()):
            if len(raw) >= 3 and raw not in GENERIC_CONTENT_TOKENS:
                tokens.add(raw)
        return tokens

    @staticmethod
    def _market_shape(text: str) -> Tuple[str, Optional[str]]:
        lowered = text.lower()
        place_match = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+place\b", lowered)
        if place_match:
            return ("place_n", place_match.group(1))
        top_match = re.search(r"\btop\s+(\d+)\b", lowered)
        if top_match:
            return ("top_n", top_match.group(1))
        if "make the cut" in lowered:
            return ("make_cut", None)
        if "on the ballot" in lowered:
            return ("on_ballot", None)
        if "minimum wage" in lowered:
            return ("min_wage", None)
        if "world cup" in lowered and "play" in lowered:
            return ("play_world_cup", None)
        if "play for" in lowered or "sign for" in lowered:
            return ("play_for_team", None)
        if "ipo" in lowered:
            return ("ipo", None)
        if "nominee" in lowered:
            return ("nominee", None)
        if "winner" in lowered or re.search(r"\bwin\b", lowered):
            return ("winner", None)
        if "recognize" in lowered:
            return ("recognition", None)
        if "o/u" in lowered or "over/under" in lowered:
            return ("over_under", None)
        return ("generic", None)

    @staticmethod
    def _date_boost(event1: Event, event2: Event) -> float:
        dt1 = ensure_aware(event1.resolution_date)
        dt2 = ensure_aware(event2.resolution_date)
        if not dt1 or not dt2:
            return 0.0
        diff_seconds = abs((dt1 - dt2).total_seconds())
        if diff_seconds <= 86400:
            return 0.05
        if diff_seconds <= 7 * 86400:
            return 0.02
        return 0.0

    def _compatible_pair(self, event1: Event, event2: Event) -> bool:
        numeric1 = self._numeric_tokens(event1.title)
        numeric2 = self._numeric_tokens(event2.title)
        if numeric1 and numeric2 and not (numeric1 & numeric2):
            return False
        content1 = self._content_tokens(event1.title)
        content2 = self._content_tokens(event2.title)
        if content1 and content2 and not (content1 & content2):
            return False
        shape1 = self._market_shape(event1.title)
        shape2 = self._market_shape(event2.title)
        if shape1[0] != "generic" and shape2[0] != "generic":
            if shape1[0] != shape2[0]:
                return False
            if shape1[1] and shape2[1] and shape1[1] != shape2[1]:
                return False
        dt1 = ensure_aware(event1.resolution_date)
        dt2 = ensure_aware(event2.resolution_date)
        if dt1 and dt2 and abs((dt1 - dt2).total_seconds()) > 180 * 86400:
            return False
        return True

    def fit(self, pm_events: Sequence[Event], kalshi_events: Sequence[Event]) -> None:
        corpus = [event.title for event in pm_events] + [event.title for event in kalshi_events]
        counts = self.vectorizer.transform(corpus)
        matrix = self.transformer.fit_transform(counts)
        self.pm_matrix = matrix[:len(pm_events)]
        self.kalshi_matrix = matrix[len(pm_events):]
        self.feature_dimension = matrix.shape[1]

    def _block_best(self, pm_indices: Sequence[int], kalshi_indices: Sequence[int], pm_events: Sequence[Event],
                    kalshi_events: Sequence[Event], pm_best: Dict[int, Tuple[int, float]],
                    kalshi_best: Dict[int, Tuple[int, float]]) -> None:
        if not pm_indices or not kalshi_indices:
            return
        kalshi_matrix = self.kalshi_matrix[list(kalshi_indices)]
        kalshi_local_scores = np.full(len(kalshi_indices), -1.0, dtype=np.float32)
        kalshi_local_pm = np.full(len(kalshi_indices), -1, dtype=np.int32)
        for start in range(0, len(pm_indices), self.chunk_size):
            chunk_pm_indices = list(pm_indices[start:start + self.chunk_size])
            similarity = linear_kernel(self.pm_matrix[chunk_pm_indices], kalshi_matrix)
            if similarity.size == 0:
                continue
            row_best_cols = similarity.argmax(axis=1)
            row_best_scores = similarity[np.arange(len(chunk_pm_indices)), row_best_cols]
            for row_offset, raw_score in enumerate(row_best_scores):
                score = float(raw_score)
                if score < self.similarity_threshold:
                    continue
                pm_index = chunk_pm_indices[row_offset]
                kalshi_index = kalshi_indices[int(row_best_cols[row_offset])]
                if not self._compatible_pair(pm_events[pm_index], kalshi_events[kalshi_index]):
                    continue
                score = min(1.0, score + self._date_boost(pm_events[pm_index], kalshi_events[kalshi_index]))
                if score > pm_best.get(pm_index, (-1, -1.0))[1]:
                    pm_best[pm_index] = (kalshi_index, score)
            col_best_rows = similarity.argmax(axis=0)
            col_best_scores = similarity[col_best_rows, np.arange(len(kalshi_indices))]
            for col_offset, raw_score in enumerate(col_best_scores):
                score = float(raw_score)
                if score < self.similarity_threshold or score <= float(kalshi_local_scores[col_offset]):
                    continue
                pm_index = chunk_pm_indices[int(col_best_rows[col_offset])]
                kalshi_index = kalshi_indices[col_offset]
                if not self._compatible_pair(pm_events[pm_index], kalshi_events[kalshi_index]):
                    continue
                kalshi_local_scores[col_offset] = score
                kalshi_local_pm[col_offset] = pm_index
        for col_offset, pm_index in enumerate(kalshi_local_pm):
            if pm_index < 0:
                continue
            kalshi_index = kalshi_indices[col_offset]
            score = float(kalshi_local_scores[col_offset])
            score = min(1.0, score + self._date_boost(pm_events[pm_index], kalshi_events[kalshi_index]))
            if score > kalshi_best.get(kalshi_index, (-1, -1.0))[1]:
                kalshi_best[kalshi_index] = (pm_index, score)

    def match_bidirectional(self, pm_events: Sequence[Event], kalshi_events: Sequence[Event]) -> List[MatchedPair]:
        if self.pm_matrix is None or self.kalshi_matrix is None:
            raise RuntimeError("Matcher must be fitted before matching.")
        pm_category_map: Dict[str, List[int]] = defaultdict(list)
        kalshi_category_map: Dict[str, List[int]] = defaultdict(list)
        for index, event in enumerate(pm_events):
            event.categories = self.classifier.classify(event.title)
            for category in event.categories:
                pm_category_map[category].append(index)
        for index, event in enumerate(kalshi_events):
            event.categories = self.classifier.classify(event.title)
            for category in event.categories:
                kalshi_category_map[category].append(index)
        pm_best: Dict[int, Tuple[int, float]] = {}
        kalshi_best: Dict[int, Tuple[int, float]] = {}
        for category in sorted(set(pm_category_map) & set(kalshi_category_map)):
            self._block_best(pm_category_map[category], kalshi_category_map[category], pm_events, kalshi_events, pm_best, kalshi_best)
        if not pm_best and not kalshi_best:
            self._block_best(list(range(len(pm_events))), list(range(len(kalshi_events))), pm_events, kalshi_events, pm_best, kalshi_best)
        seen: set[Tuple[int, int]] = set()
        pairs: List[MatchedPair] = []
        for pm_index, (kalshi_index, score) in pm_best.items():
            key = (pm_index, kalshi_index)
            if key not in seen:
                seen.add(key)
                pairs.append(MatchedPair(pm_events[pm_index], kalshi_events[kalshi_index], score))
        for kalshi_index, (pm_index, score) in kalshi_best.items():
            key = (pm_index, kalshi_index)
            if key not in seen:
                seen.add(key)
                pairs.append(MatchedPair(pm_events[pm_index], kalshi_events[kalshi_index], score))
        pairs.sort(key=lambda item: item.similarity, reverse=True)
        return pairs

def benchmark_vector_pipeline(titles: Sequence[str], hash_features: int) -> Dict[str, Any]:
    baseline = BaselinePythonVectorizer()
    start = time.perf_counter()
    baseline_matrix = baseline.fit_transform(titles)
    baseline_elapsed = time.perf_counter() - start

    version2_tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        dtype=np.float32,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b",
    )
    start = time.perf_counter()
    version2_matrix = version2_tfidf.fit_transform(titles)
    version2_elapsed = time.perf_counter() - start

    hash_vectorizer = HashingVectorizer(
        lowercase=True,
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 2),
        norm=None,
        alternate_sign=False,
        n_features=hash_features,
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b",
    )
    transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
    start = time.perf_counter()
    version3_matrix = transformer.fit_transform(hash_vectorizer.transform(titles))
    version3_elapsed = time.perf_counter() - start

    return {
        "documents": len(titles),
        "baseline_shape": list(baseline_matrix.shape),
        "version2_shape": list(version2_matrix.shape),
        "version3_shape": list(version3_matrix.shape),
        "baseline_elapsed_seconds": round(baseline_elapsed, 6),
        "version2_elapsed_seconds": round(version2_elapsed, 6),
        "version3_elapsed_seconds": round(version3_elapsed, 6),
        "version2_speedup_vs_baseline_x": round(baseline_elapsed / version2_elapsed, 3) if version2_elapsed > 1e-12 else None,
        "version3_speedup_vs_baseline_x": round(baseline_elapsed / version3_elapsed, 3) if version3_elapsed > 1e-12 else None,
        "version3_speedup_vs_version2_x": round(version2_elapsed / version3_elapsed, 3) if version3_elapsed > 1e-12 else None,
        "hash_feature_dimension": hash_features,
    }

