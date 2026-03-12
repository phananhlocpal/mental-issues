"""Medical entity extraction using keyword matching + BERT embeddings.

Since full UMLS integration requires an API key, we implement a
rule-based extractor backed by a curated psychiatric keyword lexicon
that maps surface forms to canonical concepts and symptom categories.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from transformers import AutoModel, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Lexicon – surface form → (canonical_concept, symptom_category)
# ──────────────────────────────────────────────────────────────────────────────

LEXICON: dict[str, tuple[str, str]] = {
    # mood_disorder
    "depress": ("depression", "mood_disorder"),
    "depressed": ("depression", "mood_disorder"),
    "depression": ("depression", "mood_disorder"),
    "hopeless": ("hopelessness", "mood_disorder"),
    "hopelessness": ("hopelessness", "mood_disorder"),
    "worthless": ("worthlessness", "mood_disorder"),
    "worthlessness": ("worthlessness", "mood_disorder"),
    "sad": ("sadness", "mood_disorder"),
    "sadness": ("sadness", "mood_disorder"),
    "numb": ("emotional_numbness", "mood_disorder"),
    "numbness": ("emotional_numbness", "mood_disorder"),
    "empty": ("emptiness", "mood_disorder"),
    "emptiness": ("emptiness", "mood_disorder"),
    "tearful": ("sadness", "mood_disorder"),
    "crying": ("sadness", "mood_disorder"),
    "melancholy": ("depression", "mood_disorder"),
    # anxiety_disorder
    "anxious": ("anxiety", "anxiety_disorder"),
    "anxiety": ("anxiety", "anxiety_disorder"),
    "panic": ("panic_attack", "anxiety_disorder"),
    "stress": ("stress", "anxiety_disorder"),
    "stressed": ("stress", "anxiety_disorder"),
    "overwhelm": ("overwhelm", "anxiety_disorder"),
    "overwhelmed": ("overwhelm", "anxiety_disorder"),
    "worry": ("worry", "anxiety_disorder"),
    "worrying": ("worry", "anxiety_disorder"),
    "nervous": ("nervousness", "anxiety_disorder"),
    "nervousness": ("nervousness", "anxiety_disorder"),
    "phobia": ("phobia", "anxiety_disorder"),
    "fear": ("fear", "anxiety_disorder"),
    "dread": ("dread", "anxiety_disorder"),
    "tension": ("tension", "anxiety_disorder"),
    # sleep_disorder
    "insomnia": ("insomnia", "sleep_disorder"),
    "sleepless": ("insomnia", "sleep_disorder"),
    "fatigue": ("fatigue", "sleep_disorder"),
    "exhausted": ("fatigue", "sleep_disorder"),
    "exhaustion": ("fatigue", "sleep_disorder"),
    "tired": ("fatigue", "sleep_disorder"),
    "tiredness": ("fatigue", "sleep_disorder"),
    "sleep": ("sleep_disturbance", "sleep_disorder"),
    "oversleeping": ("hypersomnia", "sleep_disorder"),
    "hypersomnia": ("hypersomnia", "sleep_disorder"),
    # social_disorder
    "lonely": ("loneliness", "social_disorder"),
    "loneliness": ("loneliness", "social_disorder"),
    "isolated": ("social_isolation", "social_disorder"),
    "isolation": ("social_isolation", "social_disorder"),
    "irritable": ("irritability", "social_disorder"),
    "irritability": ("irritability", "social_disorder"),
    "angry": ("anger", "social_disorder"),
    "anger": ("anger", "social_disorder"),
    "withdrawn": ("social_withdrawal", "social_disorder"),
    # crisis
    "suicidal": ("suicidal_ideation", "crisis"),
    "suicide": ("suicidal_ideation", "crisis"),
    "self-harm": ("self_harm", "crisis"),
    "self harm": ("self_harm", "crisis"),
    "cutting": ("self_harm", "crisis"),
    "overdose": ("self_harm", "crisis"),
    "die": ("suicidal_ideation", "crisis"),
    "dying": ("suicidal_ideation", "crisis"),
    "kill myself": ("suicidal_ideation", "crisis"),
}

# Unique concepts and categories
ALL_CONCEPTS: list[str] = sorted({v[0] for v in LEXICON.values()})
ALL_CATEGORIES: list[str] = sorted({v[1] for v in LEXICON.values()})


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    surface: str          # matched text span
    concept: str          # canonical UMLS-like concept
    category: str         # symptom category
    start: int = 0
    end: int = 0


@dataclass
class ExtractionResult:
    text: str
    entities: list[Entity] = field(default_factory=list)

    @property
    def concepts(self) -> list[str]:
        return list({e.concept for e in self.entities})

    @property
    def categories(self) -> list[str]:
        return list({e.category for e in self.entities})


# ──────────────────────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────────────────────

class MedicalEntityExtractor:
    """Keyword-based psychiatric entity extractor."""

    def __init__(self) -> None:
        # Pre-compile sorted patterns (longest match first)
        sorted_keys = sorted(LEXICON.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(k) for k in sorted_keys)
        self._re = re.compile(pattern, re.IGNORECASE)

    def extract(self, text: str) -> ExtractionResult:
        result = ExtractionResult(text=text)
        seen: set[str] = set()
        for match in self._re.finditer(text):
            surface = match.group(0).lower()
            if surface not in seen:
                seen.add(surface)
                concept, category = LEXICON[surface]
                result.entities.append(
                    Entity(
                        surface=surface,
                        concept=concept,
                        category=category,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return result

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        return [self.extract(t) for t in texts]


# ──────────────────────────────────────────────────────────────────────────────
# Concept Embeddings
# ──────────────────────────────────────────────────────────────────────────────

class ConceptEmbedder:
    """Compute BERT embeddings for medical concepts and symptom categories."""

    # Brief definitions used as input to BERT
    CONCEPT_DEFINITIONS: dict[str, str] = {
        "depression": "A mood disorder causing persistent sadness and loss of interest.",
        "hopelessness": "A feeling of having no hope or expectation for the future.",
        "worthlessness": "A sense of having no value or importance.",
        "sadness": "An emotional state of sorrow and unhappiness.",
        "emotional_numbness": "Inability to feel emotions, feeling detached.",
        "emptiness": "A feeling of inner void and lack of meaning.",
        "anxiety": "Excessive worry, nervousness, or unease.",
        "panic_attack": "Sudden intense fear with physical symptoms.",
        "stress": "Mental or emotional strain from demanding circumstances.",
        "overwhelm": "Feeling buried under excessive demands or emotions.",
        "worry": "Persistent anxious thoughts about possible outcomes.",
        "nervousness": "A state of being nervous or agitated.",
        "phobia": "An irrational intense fear of a specific object or situation.",
        "fear": "An unpleasant emotion caused by perceived threat.",
        "dread": "Deep fear or apprehension about the future.",
        "tension": "Mental or emotional strain and anxiety.",
        "insomnia": "Persistent difficulty falling or staying asleep.",
        "fatigue": "Extreme tiredness and lack of energy.",
        "sleep_disturbance": "Disruption of normal sleep patterns.",
        "hypersomnia": "Excessive daytime sleepiness or prolonged nighttime sleep.",
        "loneliness": "Feeling of being alone and isolated from others.",
        "social_isolation": "Lack of social contact and relationships.",
        "irritability": "Tendency to get annoyed or agitated easily.",
        "anger": "A strong emotion of displeasure or hostility.",
        "social_withdrawal": "Avoiding social interactions and relationships.",
        "suicidal_ideation": "Thoughts of ending one's own life.",
        "self_harm": "Deliberate injury to oneself as a coping mechanism.",
    }

    CATEGORY_DEFINITIONS: dict[str, str] = {
        "mood_disorder": "Conditions affecting emotional state including depression and related disorders.",
        "anxiety_disorder": "Conditions involving excessive fear and worry.",
        "sleep_disorder": "Conditions disrupting normal sleep and rest patterns.",
        "social_disorder": "Conditions affecting social functioning and relationships.",
        "crisis": "Acute mental health emergencies requiring immediate attention.",
    }

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        print(f"  [ConceptEmbedder] Loading {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        out = self.model(**enc)
        # CLS token embedding
        return out.last_hidden_state[:, 0, :]  # (N, 768)

    def get_concept_embeddings(self) -> dict[str, torch.Tensor]:
        concepts = ALL_CONCEPTS
        definitions = [
            self.CONCEPT_DEFINITIONS.get(c, c.replace("_", " "))
            for c in concepts
        ]
        embeddings = self._embed_texts(definitions)
        return {c: embeddings[i] for i, c in enumerate(concepts)}

    def get_category_embeddings(
        self, concept_embeddings: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Category embedding = mean of its concept embeddings."""
        cat_to_concepts: dict[str, list[str]] = {}
        for surface, (concept, category) in LEXICON.items():
            cat_to_concepts.setdefault(category, []).append(concept)

        cat_embs: dict[str, torch.Tensor] = {}
        for cat, concepts in cat_to_concepts.items():
            embs = torch.stack(
                [concept_embeddings[c] for c in set(concepts) if c in concept_embeddings]
            )
            cat_embs[cat] = embs.mean(dim=0)
        return cat_embs


if __name__ == "__main__":
    extractor = MedicalEntityExtractor()
    sample = "I can't sleep and feel exhausted all the time. I feel hopeless."
    result = extractor.extract(sample)
    print("Entities:", [(e.surface, e.concept, e.category) for e in result.entities])
    print("Concepts:", result.concepts)
    print("Categories:", result.categories)
