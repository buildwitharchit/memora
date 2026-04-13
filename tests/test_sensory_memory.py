# Tests for core.sensory_memory -- Tier 1 buffer.
# Pure logic tests, no mocks needed.

from core.sensory_memory import SensoryMemory


class TestSensoryMemory:
    """Verify the bounded message buffer behaves correctly."""

    def test_add_and_retrieve(self):
        sm = SensoryMemory(max_messages=4)
        sm.add({"role": "user", "content": "hello"})
        sm.add({"role": "assistant", "content": "hi"})
        assert sm.get_messages() == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_size_tracks_count(self):
        sm = SensoryMemory(max_messages=4)
        assert sm.size == 0
        sm.add({"role": "user", "content": "a"})
        assert sm.size == 1

    def test_is_full_at_capacity(self):
        sm = SensoryMemory(max_messages=2)
        sm.add({"role": "user", "content": "a"})
        assert not sm.is_full()
        sm.add({"role": "assistant", "content": "b"})
        assert sm.is_full()

    def test_buffer_grows_beyond_max(self):
        """Confirm no maxlen -- buffer can exceed capacity (overflow managed externally)."""
        sm = SensoryMemory(max_messages=2)
        sm.add({"role": "user", "content": "a"})
        sm.add({"role": "assistant", "content": "b"})
        sm.add({"role": "user", "content": "c"})
        assert sm.size == 3  # No silent drop.
        assert sm.is_full()

    def test_pop_oldest_returns_in_order(self):
        sm = SensoryMemory(max_messages=4)
        sm.add({"role": "user", "content": "first"})
        sm.add({"role": "assistant", "content": "second"})
        sm.add({"role": "user", "content": "third"})

        popped = sm.pop_oldest(2)
        assert popped == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        assert sm.size == 1
        assert sm.get_messages() == [{"role": "user", "content": "third"}]

    def test_pop_oldest_more_than_available(self):
        sm = SensoryMemory(max_messages=4)
        sm.add({"role": "user", "content": "only"})
        popped = sm.pop_oldest(5)
        assert popped == [{"role": "user", "content": "only"}]
        assert sm.size == 0

    def test_clear(self):
        sm = SensoryMemory(max_messages=4)
        sm.add({"role": "user", "content": "a"})
        sm.add({"role": "assistant", "content": "b"})
        sm.clear()
        assert sm.size == 0
        assert sm.get_messages() == []

    def test_max_messages_property(self):
        sm = SensoryMemory(max_messages=10)
        assert sm.max_messages == 10

    def test_messages_returned_in_chronological_order(self):
        sm = SensoryMemory(max_messages=10)
        for i in range(5):
            sm.add({"role": "user", "content": str(i)})
        contents = [m["content"] for m in sm.get_messages()]
        assert contents == ["0", "1", "2", "3", "4"]
