# batcher.py
from typing import List, Iterable


class BatchEmbedder:
    """
    Utility class for batching text inputs before sending them to the
    Gemini Embedding API. This reduces RPD (Requests Per Day) and RPM
    (Requests Per Minute) by grouping multiple texts into a single API call.

    The class does not perform embedding itself; instead, it delegates
    embedding to a provided callback function. This keeps the design modular
    and allows integration with any embedding backend.

    Attributes:
        batch_size (int): Maximum number of items per batch.
    """

    def __init__(self, batch_size: int = 10):
        """
        Initialize the batch embedder.

        Args:
            batch_size (int): Maximum number of items per batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size

    def chunk(self, items: Iterable[str]) -> Iterable[List[str]]:
        """
        Split an iterable of text items into batches.

        Args:
            items (Iterable[str]): Input texts.

        Returns:
            Iterable[List[str]]: Batches of texts.
        """
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def embed_batches(self, items: Iterable[str], embed_callback):
        """
        Embed items in batches using the provided callback function.

        Args:
            items (Iterable[str]): Input texts to embed.
            embed_callback (Callable[[List[str]], List[List[float]]]):
                A function that embeds a batch of texts and returns
                a list of embedding vectors.

        Returns:
            List[List[float]]: Flattened list of embedding vectors.
        """
        results = []
        for batch in self.chunk(items):
            vectors = embed_callback(batch)
            results.extend(vectors)
        return results
