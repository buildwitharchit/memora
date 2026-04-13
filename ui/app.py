"""
Memora -- Streamlit Application Entrypoint.

Initializes the MemoryManager in session state and renders the three-tab UI:
  1. Chat: conversation with memory-augmented assistant
  2. Episodic Log: structured interaction history viewer
  3. Semantic Browser: manual similarity search over stored summaries
"""

import streamlit as st

from core.config import get_settings
from core.memory_manager import MemoryManager
from ui.components.chat_tab import render_chat_tab
from ui.components.episodic_tab import render_episodic_tab
from ui.components.memory_controls import render_memory_controls
from ui.components.semantic_tab import render_semantic_tab

# -- Page configuration --
st.set_page_config(
    page_title="Memora",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_memory_manager() -> None:
    """Create the MemoryManager once per Streamlit session."""
    if "memory_manager" not in st.session_state:
        settings = get_settings()
        st.session_state["memory_manager"] = MemoryManager(settings=settings)


def main() -> None:
    """Application entrypoint."""
    _init_memory_manager()

    # Sidebar: title and controls.
    st.sidebar.title("Memora")
    st.sidebar.caption("Multi-Tier Memory for AI Assistants")
    render_memory_controls()

    # Main content: three tabs.
    tab_chat, tab_episodic, tab_semantic = st.tabs(
        ["Chat", "Episodic Log", "Semantic Browser"]
    )

    with tab_chat:
        render_chat_tab()

    with tab_episodic:
        render_episodic_tab()

    with tab_semantic:
        render_semantic_tab()


if __name__ == "__main__":
    main()
