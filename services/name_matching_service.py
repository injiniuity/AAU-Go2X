from typing import Any

import numpy as np

from ..utils.logging_utils import print
from ..utils.text_utils import ascii_safe_text, normalize_name_key


class NameMatchingService:
    def __init__(
        self,
        *,
        state: Any,
        model_id: str,
        g2p_cls: type | None,
        sentence_transformer_cls: type | None,
    ) -> None:
        self._state = state
        self._model_id = model_id
        self._g2p_cls = g2p_cls
        self._sentence_transformer_cls = sentence_transformer_cls

    def ensure_arpabet_predictor(self):
        if self._state.arpabet_predictor is not None or self._g2p_cls is None:
            return self._state.arpabet_predictor
        try:
            self._state.arpabet_predictor = self._g2p_cls()
        except Exception as exc:
            print(f"[NameMatch] Failed to initialize ARPAbet predictor: {exc}", flush=True)
            self._state.arpabet_predictor = None
        return self._state.arpabet_predictor

    def text_to_arpabet(self, text: str) -> str:
        key = str(text).strip()
        if not key:
            return ""
        cached = self._state.arpabet_cache.get(key)
        if cached is not None:
            return cached

        predictor = self.ensure_arpabet_predictor()
        if predictor is not None:
            try:
                predicted = [
                    part
                    for part in predictor(key)
                    if isinstance(part, str) and part.strip() and part != " "
                ]
                result = " ".join(predicted).strip()
                if result:
                    self._state.arpabet_cache[key] = result
                    return result
            except Exception as exc:
                print(
                    f"[NameMatch] ARPAbet prediction failed for '{ascii_safe_text(key)}': {exc}",
                    flush=True,
                )

        self._state.arpabet_cache[key] = key
        return key

    def g2p_transform_text(self, text: str) -> str:
        transformed = self.text_to_arpabet(text)
        return transformed or str(text)

    def ensure_matcher(self) -> bool:
        if self._state.name_matcher_ready:
            return True
        if self._state.name_matcher_failed:
            return False
        if self._sentence_transformer_cls is None or self._g2p_cls is None:
            self._state.name_matcher_failed = True
            print("[NameMatch] Optional dependencies are missing; using direct name matching only", flush=True)
            return False

        try:
            print(f"[NameMatch] Loading embedding model: {self._model_id}", flush=True)
            self._state.name_matcher_model = self._sentence_transformer_cls(self._model_id)
            self._state.name_matcher_ready = True
            return True
        except Exception as exc:
            self._state.name_matcher_failed = True
            print(f"[NameMatch] Failed to initialize matcher: {exc}", flush=True)
            return False

    def clear_cache(self) -> None:
        self._state.name_matcher_embeddings = {}
        self._state.name_matcher_transforms = {}

    def rebuild_embeddings(self) -> None:
        self.clear_cache()

        if (
            not self._state.named_locations
            or not self.ensure_matcher()
            or self._state.name_matcher_model is None
        ):
            return

        candidate_keys = list(self._state.named_locations.keys())
        transformed = [self.g2p_transform_text(candidate) for candidate in candidate_keys]
        vectors = self._state.name_matcher_model.encode(
            transformed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for candidate, transformed_text, vector in zip(candidate_keys, transformed, vectors):
            self._state.name_matcher_transforms[candidate] = transformed_text
            self._state.name_matcher_embeddings[candidate] = vector

    def preload(self) -> None:
        if not self._state.named_locations:
            print("[NameMatch] No named locations loaded; skipping preload", flush=True)
            return
        if not self.ensure_matcher() or self._state.name_matcher_model is None:
            print("[NameMatch] Embedding matcher unavailable; continuing without preload", flush=True)
            return
        print(f"[NameMatch] Precomputing embeddings for {len(self._state.named_locations)} locations...", flush=True)
        self.rebuild_embeddings()
        if self._state.name_matcher_embeddings:
            print("[NameMatch] Embedding matcher is ready", flush=True)
        else:
            print("[NameMatch] Embedding preload produced no candidates", flush=True)

    def match_location(self, location: str):
        if not self._state.named_locations:
            return None
        if not self.ensure_matcher() or self._state.name_matcher_model is None:
            return None
        if not self._state.name_matcher_embeddings:
            self.rebuild_embeddings()
        if not self._state.name_matcher_embeddings:
            return None

        query_key = normalize_name_key(location)
        query_transformed = self.g2p_transform_text(query_key)
        query_vector = self._state.name_matcher_model.encode(
            [query_transformed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        best_key = None
        best_score = -1.0
        second_score = -1.0
        for candidate_key, candidate_vector in self._state.name_matcher_embeddings.items():
            score = float(np.dot(query_vector, candidate_vector))
            if score > best_score:
                second_score = best_score
                best_score = score
                best_key = candidate_key
            elif score > second_score:
                second_score = score

        if best_key is None:
            return None

        score_gap = best_score - second_score if second_score >= 0.0 else best_score
        print(
            "[NameMatch] "
            f"query='{ascii_safe_text(location)}' g2p='{ascii_safe_text(query_transformed)}' "
            f"best='{best_key}' score={best_score:.4f} gap={score_gap:.4f}",
            flush=True,
        )
        return self._state.named_locations[best_key]
