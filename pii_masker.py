from dataclasses import dataclass, field
from typing import Optional

try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print(
        "Warning: presidio not installed. PII masking disabled. Run: "
        "pip install presidio-analyzer presidio-anonymizer && "
        "python -m spacy download en_core_web_lg"
    )


@dataclass
class MaskingResult:
    masked_text: str
    restore_map: dict = field(default_factory=dict)
    entities_found: list = field(default_factory=list)


class DWCPresidio:
    """Singleton Presidio engine with DWC custom recognizers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        if not PRESIDIO_AVAILABLE:
            self.analyzer = None
            self.anonymizer = None
            return
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self._add_custom_recognizers()

    def _add_custom_recognizers(self):
        # TRUST_NAME recognizer
        trust_recognizer = PatternRecognizer(
            supported_entity="TRUST_NAME",
            patterns=[
                Pattern(
                    "trust_name_dated",
                    r"The\s+\w[\w\s]+(?:Family|Living|Revocable|Irrevocable)?\s+Trust(?:\s+dated\s+\w+\s+\d{1,2},?\s+\d{4})?",
                    0.80,
                ),
                Pattern(
                    "trust_name_short",
                    r"\w[\w\s]{2,30}\s+(?:Family\s+)?Trust(?:\s+\(\d{4}\))?",
                    0.65,
                ),
            ],
        )
        # ENTITY_NAME recognizer
        entity_recognizer = PatternRecognizer(
            supported_entity="ENTITY_NAME",
            patterns=[
                Pattern(
                    "entity_name",
                    r"\b[A-Z][A-Za-z\s&]{2,40}\s+(?:LLC|PLLC|PA|Inc\.|Corp\.)",
                    0.75,
                ),
            ],
        )
        # FL_PARCEL_ID recognizer
        parcel_recognizer = PatternRecognizer(
            supported_entity="FL_PARCEL_ID",
            patterns=[
                Pattern(
                    "fl_parcel",
                    r"\d{2}-\d{2}-\d{2}-\d{2}-\d{3}-\d{3}-\d{4}",
                    0.90,
                ),
            ],
        )
        for recognizer in [trust_recognizer, entity_recognizer, parcel_recognizer]:
            self.analyzer.registry.add_recognizer(recognizer)

    def mask(self, text: str) -> MaskingResult:
        if not PRESIDIO_AVAILABLE or not self.analyzer or not text or not text.strip():
            return MaskingResult(masked_text=text)

        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
                "US_BANK_NUMBER", "LOCATION", "DATE_TIME", "TRUST_NAME",
                "ENTITY_NAME", "FL_PARCEL_ID",
            ],
            score_threshold=0.60,
        )
        if not results:
            return MaskingResult(masked_text=text)

        restore_map = {}
        counter = {}
        operators = {}
        temp_counter = {}

        for r in results:
            entity_type = r.entity_type
            original_value = text[r.start:r.end]
            if entity_type not in counter:
                counter[entity_type] = 0
            counter[entity_type] += 1
            placeholder = f"<{entity_type}_{counter[entity_type]}>"
            restore_map[placeholder] = original_value

        for r in results:
            entity_type = r.entity_type
            if entity_type not in temp_counter:
                temp_counter[entity_type] = 0
            temp_counter[entity_type] += 1
            placeholder = f"<{entity_type}_{temp_counter[entity_type]}>"
            operators[entity_type] = OperatorConfig("replace", {"new_value": placeholder})

        anonymized = self.anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        )
        return MaskingResult(
            masked_text=anonymized.text,
            restore_map=restore_map,
            entities_found=[r.entity_type for r in results],
        )

    def restore(self, masked_text: str, restore_map: dict) -> str:
        if not restore_map:
            return masked_text
        text = masked_text
        for placeholder, original in restore_map.items():
            text = text.replace(placeholder, original)
        return text


_presidio = None


def get_presidio() -> DWCPresidio:
    global _presidio
    if _presidio is None:
        _presidio = DWCPresidio()
    return _presidio


def mask_client_data(text: str) -> MaskingResult:
    """Call this before every Claude or OpenAI API call. Returns masked text + restore map."""
    return get_presidio().mask(text)


def restore_pii(masked_response: str, restore_map: dict) -> str:
    """Call this after Claude response returns to restore client identifiers."""
    return get_presidio().restore(masked_response, restore_map)
