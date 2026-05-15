from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_DATA_DIR = REPO_ROOT / "shared" / "data"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
AGENTIC_DIR = EXPERIMENTS_DIR / "agentic"
DSPY_DIR = EXPERIMENTS_DIR / "dspy"
AGENTIC_ON_DSPY_DIR = EXPERIMENTS_DIR / "agentic_on_dspy"

MODEL_PRICING_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4.1": {"input": 2.00, "output": 8.00},
    "openai/gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

KNOWN_ATIS_LABELS = [
    "abbreviation",
    "aircraft",
    "airfare",
    "airline",
    "airport",
    "capacity",
    "city",
    "distance",
    "flight",
    "flight+airfare",
    "flight_no",
    "flight_time",
    "ground_fare",
    "ground_service",
    "meal",
    "quantity",
    "restriction",
]

LABEL_GUIDANCE = {
    "abbreviation": "questions asking what an airline, airport, or code abbreviation stands for",
    "aircraft": "questions about aircraft type or plane model used for a flight",
    "airfare": "questions about ticket price or fare only, without asking to list flights",
    "airline": "questions asking which airline operates or serves a route",
    "airport": "questions asking for an airport for a city or location",
    "capacity": "questions about seating capacity or how many people an aircraft holds",
    "city": "questions asking what city an airport belongs to",
    "distance": "questions asking for distance or miles between places",
    "flight": "questions asking to find, list, or describe flights",
    "flight+airfare": "questions explicitly asking for both flights and fares together",
    "flight_no": "questions asking for a flight number",
    "flight_time": "questions asking for arrival time, departure time, or flight duration/time",
    "ground_fare": "questions asking how much a taxi, limousine, rental car, or other ground transport costs",
    "ground_service": "questions asking what ground transportation is available, not its price",
    "meal": "questions asking whether a meal is served or what meal is available",
    "quantity": "questions asking how many flights or items exist",
    "restriction": "questions about fare restrictions or limits",
}
