"""
Chat Tab -- main conversation interface with a live memory inspector sidebar.

Left column (3/4 width): standard chat UI using st.chat_message.
Right column (1/4 width): real-time view of all four memory tiers.

The display chat history (st.session_state["chat_history"]) is separate from
the sensory buffer. The display history keeps the full conversation for the UI;
the sensory buffer is a sliding window that compresses older messages.
"""

import streamlit as st


def render_chat_tab() -> None:
    """Render the chat interface and memory inspector."""
    mm = st.session_state.get("memory_manager")
    if mm is None:
        st.error("Memory manager not initialized.")
        return

    # Initialize display chat history if absent.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    left, right = st.columns([3, 1])

    with left:
        _render_chat(mm)

    with right:
        _render_memory_inspector(mm)


def _render_chat(mm) -> None:
    """Render the chat message list and input box."""
    # Display all messages from the full chat history.
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input.
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Show user message immediately.
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process through the full memory pipeline.
        with st.spinner("Thinking..."):
            response = mm.process_message(user_input)

        # Show and record assistant response.
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant"):
            st.markdown(response)

        st.rerun()


def _render_memory_inspector(mm) -> None:
    """Render the live memory tier inspector in the right column."""
    st.markdown("#### Memory Inspector")

    # Tier 1 -- Sensory buffer.
    with st.expander(
        f"Tier 1: Sensory ({mm.sensory.size}/{mm.sensory.max_messages})", expanded=False
    ):
        messages = mm.sensory.get_messages()
        if messages:
            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"][:80]
                st.text(f"{role}: {content}{'...' if len(msg['content']) > 80 else ''}")
        else:
            st.caption("Empty")

    # Tier 2 -- Compression log.
    with st.expander(
        f"Tier 2: Compressions ({len(mm.short_term.compression_log)})", expanded=False
    ):
        log = mm.short_term.compression_log
        if log:
            for i, entry in enumerate(log):
                st.text(
                    f"#{i+1}: {entry['original_tokens']} -> {entry['summary_tokens']} "
                    f"tokens (ratio: {entry['ratio']}x)"
                )
        else:
            st.caption("No compressions yet")

    # Tier 3 -- Retrieved summaries (from last query).
    with st.expander(
        f"Tier 3: Semantic ({mm.semantic.count()} stored)", expanded=False
    ):
        summaries = mm.last_retrieved_summaries
        if summaries:
            st.caption("Last retrieved:")
            for s in summaries:
                dist = f"{s['distance']:.3f}" if "distance" in s else "?"
                st.text(f"[dist={dist}] {s['document'][:100]}")
        else:
            st.caption("No summaries retrieved")

    # Tier 4 -- Recent episodic entries.
    with st.expander(
        f"Tier 4: Episodic ({mm.episodic.count()} total)", expanded=False
    ):
        entries = mm.last_episodic_entries
        if entries:
            for e in entries:
                st.text(f"[{e.timestamp[:19]}] {e.intent} -> {e.outcome}")
        else:
            st.caption("No episodic entries")
