"""
Sidebar memory controls for Memora.

Provides buttons to start a new session and reset individual memory tiers.
Rendered in the Streamlit sidebar.
"""

import streamlit as st


def render_memory_controls() -> None:
    """Render session and memory reset controls in the sidebar."""
    mm = st.session_state.get("memory_manager")
    if mm is None:
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Session")

    if st.sidebar.button("New Session", use_container_width=True):
        mm.new_session()
        st.session_state["chat_history"] = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Reset Memory Tiers")

    if st.sidebar.button("Reset Short-Term (Tier 2)", use_container_width=True):
        mm.reset_short_term()
        st.rerun()

    if st.sidebar.button("Reset Semantic (Tier 3)", use_container_width=True):
        mm.reset_semantic()
        st.rerun()

    if st.sidebar.button("Reset Episodic (Tier 4)", use_container_width=True):
        mm.reset_episodic()
        st.rerun()
