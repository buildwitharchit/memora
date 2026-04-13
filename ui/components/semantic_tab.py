"""
Semantic Memory Browser Tab.

Provides a manual ChromaDB similarity search interface. Enter any phrase
and see the most relevant stored summaries with their distance scores.
"""

import streamlit as st


def render_semantic_tab() -> None:
    """Render the semantic memory browser with search and browse modes."""
    mm = st.session_state.get("memory_manager")
    if mm is None:
        st.error("Memory manager not initialized.")
        return

    st.subheader("Semantic Memory Browser (Tier 3)")

    total = mm.semantic.count()
    st.caption(f"{total} summaries stored")

    # Search controls.
    query = st.text_input(
        "Search by similarity", placeholder="Type a phrase to find related memories..."
    )
    top_k = st.slider("Results to return", min_value=1, max_value=10, value=3)

    if query:
        results = mm.semantic.search(query, top_k=top_k)
        if results:
            for i, r in enumerate(results):
                distance = f"{r['distance']:.4f}"
                with st.expander(f"Result {i+1} (distance: {distance})", expanded=True):
                    st.write(r["document"])
                    if r.get("metadata"):
                        st.caption(f"Metadata: {r['metadata']}")
        else:
            st.info("No matching summaries found.")
    else:
        # Browse mode: show all stored summaries when no search query is entered.
        if total > 0:
            st.markdown("---")
            st.markdown("**All Stored Summaries**")
            all_summaries = mm.semantic.get_all()
            for i, s in enumerate(all_summaries):
                with st.expander(f"Summary {i+1}", expanded=False):
                    st.write(s["document"])
                    if s.get("metadata"):
                        st.caption(f"Metadata: {s['metadata']}")
