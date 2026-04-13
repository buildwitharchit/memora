"""
Episodic Log Viewer Tab.

A sortable, filterable table of all episodic memory entries (Tier 4).
Supports filtering by session ID and intent substring.
"""

import pandas as pd
import streamlit as st


def render_episodic_tab() -> None:
    """Render the episodic log viewer with filters."""
    mm = st.session_state.get("memory_manager")
    if mm is None:
        st.error("Memory manager not initialized.")
        return

    st.subheader("Episodic Memory Log (Tier 4)")

    # Filter controls.
    col1, col2 = st.columns(2)
    with col1:
        session_filter = st.text_input(
            "Filter by Session ID", value="", key="episodic_session_filter"
        )
    with col2:
        intent_filter = st.text_input(
            "Filter by Intent (substring)", value="", key="episodic_intent_filter"
        )

    # Query episodic memory with filters.
    entries = mm.episodic.get_all(
        session_id=session_filter if session_filter else None,
        intent_pattern=intent_filter if intent_filter else None,
        limit=200,
    )

    if entries:
        # Convert pydantic models to a pandas DataFrame for display.
        data = [e.model_dump() for e in entries]
        df = pd.DataFrame(data)
        # Reorder columns for readability.
        column_order = ["id", "timestamp", "session_id", "user_id", "intent", "outcome"]
        df = df[[c for c in column_order if c in df.columns]]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(entries)} entries")
    else:
        st.info("No episodic entries found matching your filters.")
