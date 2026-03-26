"""Shared constants for the pipeline core layer."""

DEFAULT_RANDOM_SEED = 42

# These are the canonical columns that downstream stages can rely on.
STANDARD_COLUMNS = (
    "id",
    "source",
    "text",
    "label",
    "rating",
    "created_at",
    "split",
)