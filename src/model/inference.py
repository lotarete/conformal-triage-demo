"""
Triage classification inference via OpenAI API (GPT-4o-mini).

Uses GPT-4o-mini with JSON mode + logprobs to extract real
4-class probability distributions for conformal prediction.

Approach:
1. Force JSON output: {"classification": "3"} via response_format
2. Extract top_logprobs at the classification value token
3. Find logprobs for all 4 class tokens and softmax into probabilities

This solves the vet demo's failure mode: instead of verbalized confidence
(1.5-2% separation), we get actual logit-derived probabilities at the
decision token.

NOTE: Originally built for DeepSeek-chat, but DeepSeek's API returns -9999.0
sentinel values for ALL non-chosen token logprobs, making real probability
extraction impossible. GPT-4o-mini returns genuine alternative-token logprobs.

Pricing: ~$0.15/M input, $0.60/M output — pilot of 250 examples costs < $0.05.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

# Triage class labels — order matters (matches prompt numbering)
TRIAGE_CLASSES = ["self_care", "gp_visit", "urgent_care", "emergency"]
CLASS_TOKENS = {
    # Possible token representations the model might use in JSON value
    "self_care": ["self_care", "self", "1"],
    "gp_visit": ["gp_visit", "gp", "2"],
    "urgent_care": ["urgent_care", "urgent", "3"],
    "emergency": ["emergency", "4"],
}


class TriageModel(ABC):
    """Base class for triage classification models."""

    @abstractmethod
    def predict(self, symptom_text: str) -> Dict:
        """
        Classify a symptom description into one of 4 triage levels.

        Returns:
            Dict with keys:
            - predicted_class: str (self_care, gp_visit, urgent_care, emergency)
            - probs: np.ndarray of shape (4,) — softmax probabilities
            - raw_response: str — model's raw text output
        """
        pass

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Batch prediction for efficiency."""
        pass


class MockTriageModel(TriageModel):
    """
    Mock model for pipeline testing. Generates plausible but synthetic
    probabilities to validate the CP pipeline before real model is ready.

    Simulates ~70% accuracy with ~15% confidence separation.
    """

    def __init__(self, accuracy: float = 0.70, seed: int = 42):
        self.accuracy = accuracy
        self.rng = np.random.RandomState(seed)

    def predict(self, symptom_text: str) -> Dict:
        h = hash(symptom_text) % 10000
        self.rng.seed(h)

        true_class = self._guess_class(symptom_text)
        is_correct = self.rng.random() < self.accuracy

        if is_correct:
            main_prob = 0.5 + self.rng.random() * 0.35
            probs = self._make_probs(true_class, main_prob)
        else:
            wrong_class = (true_class + self.rng.randint(1, 4)) % 4
            main_prob = 0.3 + self.rng.random() * 0.3
            probs = self._make_probs(wrong_class, main_prob)

        predicted = int(np.argmax(probs))

        # Generate synthetic logits from probs (inverse softmax)
        raw_logits = np.log(probs + 1e-10)

        return {
            "predicted_class": TRIAGE_CLASSES[predicted],
            "probs": probs,
            "raw_logits": raw_logits,
            "raw_response": f"Mock prediction: {TRIAGE_CLASSES[predicted]}",
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]

    def _guess_class(self, text: str) -> int:
        text = text.lower()
        if any(w in text for w in ["chest pain", "breathing", "unconscious", "stroke", "severe bleeding"]):
            return 3
        elif any(w in text for w in ["broken", "high fever", "vomiting blood", "sudden"]):
            return 2
        elif any(w in text for w in ["persistent", "recurring", "mild fever", "rash"]):
            return 1
        else:
            return 0

    def _make_probs(self, main_class: int, main_prob: float) -> np.ndarray:
        probs = np.zeros(4)
        probs[main_class] = main_prob
        remaining = 1.0 - main_prob
        other_probs = self.rng.dirichlet(np.ones(3)) * remaining
        other_idx = [i for i in range(4) if i != main_class]
        for i, idx in enumerate(other_idx):
            probs[idx] = other_probs[i]
        return probs


class OpenAITriageModel(TriageModel):
    """
    OpenAI GPT-4o-mini API with JSON mode + logprobs.

    Strategy for extracting 4-class probabilities:
    1. System prompt defines the 4 triage classes with numeric labels.
    2. JSON mode forces output like {"classification": "3"}.
    3. logprobs=True + top_logprobs=20 captures the probability
       distribution at the classification token position.
    4. Extract logprobs for tokens "1", "2", "3", "4" and softmax
       them into a proper 4-class probability distribution.

    This gives us REAL model probabilities at the decision point —
    not verbalized confidence, not averaged token logprobs.
    """

    SYSTEM_PROMPT = (
        "You are a medical triage classifier. Given a patient's symptom description, "
        "classify the urgency into exactly ONE of these four categories:\n\n"
        "1 = self_care (can manage at home, no medical visit needed)\n"
        "2 = gp_visit (should see a doctor within 1-3 days)\n"
        "3 = urgent_care (needs attention within hours, not life-threatening)\n"
        "4 = emergency (immediate emergency department, potentially life-threatening)\n\n"
        "Respond with a JSON object containing only the classification number.\n"
        'Example: {"classification": "2"}'
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.1,
        calibration_T: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.calibration_T = calibration_T  # ConfTS post-hoc temperature (1.0 = no scaling)

        # Initialize OpenAI client
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    @staticmethod
    def apply_temperature_scaling(raw_logits: np.ndarray, T: float = 1.0) -> np.ndarray:
        """
        Apply post-hoc temperature scaling to raw logits and return probabilities.

        This implements Conformal Temperature Scaling (ConfTS): instead of tuning T
        to minimize calibration error (Guo et al. 2017), T is tuned to minimize
        conformal prediction set sizes while maintaining coverage.

        Args:
            raw_logits: Array of shape (4,) with raw log-probabilities.
            T: Temperature parameter. T=1.0 means no scaling.
                T>1 softens (spreads probability), T<1 sharpens (concentrates).

        Returns:
            Array of shape (4,) with softmax probabilities.
        """
        if T <= 0:
            raise ValueError(f"Temperature must be positive, got {T}")
        scaled = raw_logits / T
        shifted = scaled - np.max(scaled)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        return probs

    def predict(self, symptom_text: str) -> Dict:
        """
        Classify a symptom and extract logprob-derived 4-class probabilities.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": symptom_text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    max_tokens=20,
                    logprobs=True,
                    top_logprobs=20,
                )

                content = response.choices[0].message.content or ""
                logprobs_data = response.choices[0].logprobs

                # Parse the predicted class from JSON
                predicted_class = self._parse_classification(content)

                # Extract 4-class probabilities and raw logits from logprobs
                probs, raw_logits = self._extract_class_probs(logprobs_data)

                time.sleep(self.rate_limit_delay)

                return {
                    "predicted_class": predicted_class,
                    "probs": probs,
                    "raw_logits": raw_logits,
                    "raw_response": content,
                    "logprobs_raw": logprobs_data,
                }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    print(f"  All {self.max_retries} attempts failed for: {symptom_text[:60]}...")
                    # Return uniform fallback
                    uniform_logits = np.zeros(4)
                    return {
                        "predicted_class": "gp_visit",
                        "probs": np.ones(4) / 4,
                        "raw_logits": uniform_logits,
                        "raw_response": f"ERROR: {e}",
                    }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Sequential batch prediction (API is per-request)."""
        results = []
        for i, text in enumerate(texts):
            if i > 0 and i % 50 == 0:
                print(f"  Processed {i}/{len(texts)} examples...")
            results.append(self.predict(text))
        return results

    def _parse_classification(self, content: str) -> str:
        """Parse the classification from JSON response."""
        try:
            data = json.loads(content)
            raw_val = str(data.get("classification", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            # Fallback: try to find a number in the response
            raw_val = ""
            match = re.search(r"[1-4]", content)
            if match:
                raw_val = match.group()

        # Map to class name
        num_to_class = {"1": "self_care", "2": "gp_visit", "3": "urgent_care", "4": "emergency"}
        if raw_val in num_to_class:
            return num_to_class[raw_val]

        # Try matching class names directly
        for cls in TRIAGE_CLASSES:
            if cls in raw_val.lower():
                return cls

        return "gp_visit"  # safe fallback

    def _extract_class_probs(self, logprobs_data) -> np.ndarray:
        """
        Extract 4-class probabilities from the logprobs at the
        classification value token.

        The JSON output looks like: {"classification": "3"}
        We want the logprobs at the token position where "3" appears
        (the value token), and specifically the logprobs for tokens
        "1", "2", "3", "4" at that position.

        These logprobs represent the model's actual probability
        distribution over the 4 triage classes.

        Robustness measures:
        - Handles multi-char tokens like '"3"', '3"}', '3"}'
        - Skips tokens before the colon (avoids matching in key name)
        - Falls back to class name matching if numbers aren't found
        - Logs a warning if no class tokens found (silent uniform is dangerous)
        """
        if logprobs_data is None or not hasattr(logprobs_data, "content"):
            print("  WARNING: No logprobs data — returning uniform probabilities")
            return np.ones(4) / 4, np.zeros(4)  # uniform fallback

        # Search through token positions for the classification value
        # In {"classification": "3"}, the value token is "3"
        #
        # Default logprob for classes NOT found in top-20 alternatives.
        # If a class token doesn't appear in GPT-4o-mini's top-20, the model
        # genuinely considers it negligible. We use -20.0 so that after softmax
        # (even with ConfTS temperature scaling), these produce ~0% probability.
        # Previously -10.0, which softmaxed to ~1.7% — enough to inflate CP
        # prediction sets with spurious classes the model never actually considered.
        MISSING_LOGPROB = -20.0

        class_logprobs = {"1": MISSING_LOGPROB, "2": MISSING_LOGPROB,
                          "3": MISSING_LOGPROB, "4": MISSING_LOGPROB}

        # Also track class name tokens in case model outputs names
        name_logprobs = {
            "self_care": MISSING_LOGPROB, "gp_visit": MISSING_LOGPROB,
            "urgent_care": MISSING_LOGPROB, "emergency": MISSING_LOGPROB,
        }
        name_to_num = {
            "self_care": "1", "self": "1",
            "gp_visit": "2", "gp": "2",
            "urgent_care": "3", "urgent": "3",
            "emergency": "4",
        }

        found_class_token = False

        # Skip tokens that are part of the key (before the colon).
        # Track whether we've passed the colon in the JSON.
        past_colon = False

        for token_info in logprobs_data.content:
            token_text = token_info.token

            # Track position: once we see ":" we're in the value area
            if ":" in token_text:
                past_colon = True
                continue

            if not past_colon:
                continue

            # Check if this token or its alternatives contain class numbers
            candidates = [token_info] + (token_info.top_logprobs or [])

            has_class_token = False
            for candidate in candidates:
                # Robust stripping: handle '"3"', '3"}', '3\n', etc.
                stripped = re.sub(r'[^0-9a-z_]', '', candidate.token.lower().strip())
                if stripped in ("1", "2", "3", "4"):
                    has_class_token = True
                    break
                if stripped in name_to_num:
                    has_class_token = True
                    break

            if not has_class_token:
                continue

            found_class_token = True

            # This token position contains class information — extract all
            for candidate in candidates:
                stripped = re.sub(r'[^0-9a-z_]', '', candidate.token.lower().strip())

                # Numeric tokens
                if stripped in class_logprobs:
                    class_logprobs[stripped] = max(
                        class_logprobs[stripped], candidate.logprob
                    )

                # Class name tokens (fallback)
                if stripped in name_logprobs:
                    name_logprobs[stripped] = max(
                        name_logprobs[stripped], candidate.logprob
                    )
                # Partial name matches
                if stripped in name_to_num:
                    num = name_to_num[stripped]
                    class_logprobs[num] = max(
                        class_logprobs[num], candidate.logprob
                    )

        # Check if we found anything useful
        all_default = all(v == MISSING_LOGPROB for v in class_logprobs.values())
        if all_default and not found_class_token:
            print("  WARNING: No class tokens found in logprobs — returning uniform probabilities")
            print(f"    Token stream: {[t.token for t in logprobs_data.content]}")
            return np.ones(4) / 4, np.zeros(4)

        # Collect raw logits (before any temperature scaling)
        raw_logits = np.array([
            class_logprobs["1"],
            class_logprobs["2"],
            class_logprobs["3"],
            class_logprobs["4"],
        ])

        # Apply calibration temperature if set, then softmax
        probs = self.apply_temperature_scaling(raw_logits, self.calibration_T)

        return probs, raw_logits
