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
        self.iterator = self._reset()
        self.buffer = None

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.original_iterator)

    def _reset(self):
        # Free up CUDA memory
        del self.iterator
        torch.cuda.empty_cache()
        # Reset the iterator
        iterator = iter(self.original_iterator)
        return iterator

    def __next__(self) -> any:
        self.buffer = next(self.iterator, None)
        if self.buffer is None:
            self.iterator = self._reset()
            self.buffer = next(self.iterator, None)
        current = self.buffer
        return current
