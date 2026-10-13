"""GloVe-300d sentence embeddings, as used by the RelBench examples."""
import math

from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/average_word_embeddings_glove.6B.300d"


class GloveTextEmbedding:
    def __init__(self, device=None):
        self.model = SentenceTransformer(MODEL, device=device)

    def __call__(self, sentences):
        return self.model.encode([_as_text(s) for s in sentences], convert_to_tensor=True)


def _as_text(value):
    """Missing values (None, NaN) become empty strings; everything else is stringified."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value if isinstance(value, str) else str(value)
