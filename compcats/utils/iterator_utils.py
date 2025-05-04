import gc
import torch
from collections.abc import Iterator
from typing import Iterator as TypingIterator

def convert_to_cyclic(iterator: TypingIterator) -> TypingIterator:
    """
    Converts an iterator to a cyclic iterator. This means that when the
    original iterator is exhausted, it will start again from the beginning.
    """
    return CyclicIterator(iterator)

class CyclicIterator(Iterator):
    """
    Creates a cyclic iterator that returns elements from the original iterator
    in a cyclic manner. When the original iterator is exhausted, it resets
    and continues iterating from the beginning.
    """
    def __init__(self, iterator: TypingIterator):
        self.original_iterator = iterator
        self.buffer = None
        self.iterator = iter(self.original_iterator)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.original_iterator)

    def reset(self):
        """
        Resets the iterator to the beginning. Depending on the current state of
        the iterator, this may first exhaust the iterator and then reset it.
        """
        if self.buffer is not None:
            # If the buffer is not empty, we need to exhaust the iterator
            # before resetting it.
            while self.buffer is not None:
                self.buffer = next(self.iterator, None)
        # Reset the iterator
        self._reset()

    def _reset(self):
        """
        Resets the iterator to the beginning. This is useful when the
        original iterator is exhausted and we want to start again from
        the beginning.
        """
        if hasattr(self, 'iterator'):
            # If the iterator is already set, we need to free up CUDA memory
            # before resetting it.
            del self.buffer
            del self.iterator
            # Free up CUDA cache
            torch.cuda.empty_cache()
            # Free up RAM
            gc.collect()
        # Recreate the iterator and buffer
        self.iterator = iter(self.original_iterator)
        self.buffer = None

    def __next__(self) -> any:
        self.buffer = next(self.iterator, None)
        if self.buffer is None:
            self.reset()
            self.buffer = next(self.iterator, None)
        current = self.buffer
        return current
