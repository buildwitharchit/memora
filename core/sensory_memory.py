"""
Tier 1 -- Sensory Memory.

A bounded buffer of the most recent conversation messages. This is the LLM's
"working context" -- the raw messages passed directly to the model.

IMPORTANT: The internal deque does NOT use maxlen. Overflow is managed
externally by MemoryManager, which pops the oldest messages and routes them
through compression (Tier 2) before they are discarded. Using maxlen would
silently drop messages without compression.
"""

from collections import deque
from typing import TypeAlias

Message: TypeAlias = dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


class SensoryMemory:
    """Tier 1: Fixed-capacity buffer of recent conversation messages."""

    def __init__(self, max_messages: int = 8) -> None:
        self._buffer: deque[Message] = deque()  # No maxlen -- see module docstring.
        self._max = max_messages

    def add(self, message: Message) -> None:
        """Append a message to the buffer."""
        self._buffer.append(message)

    def get_messages(self) -> list[Message]:
        """Return all buffered messages in chronological order."""
        return list(self._buffer)

    def is_full(self) -> bool:
        """True when the buffer has reached or exceeded the configured capacity."""
        return len(self._buffer) >= self._max

    def pop_oldest(self, count: int) -> list[Message]:
        """Remove and return the oldest `count` messages (or fewer if buffer is smaller)."""
        result: list[Message] = []
        for _ in range(min(count, len(self._buffer))):
            result.append(self._buffer.popleft())
        return result

    def clear(self) -> None:
        """Discard all messages."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Current number of messages in the buffer."""
        return len(self._buffer)

    @property
    def max_messages(self) -> int:
        """Configured capacity threshold."""
        return self._max
