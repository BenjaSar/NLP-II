import streamlit as st
from streamlit_option_menu import option_menu

def configure_streamlit_page():
    """
    Configure the Streamlit page settings.
    """
    st.set_page_config(
        page_title="AI Engineer Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def display_sidebar_menu():
    """
    Display a sidebar menu with navigation options.
    Returns the selected menu option.
    """
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Chat", "Settings", "About"],
            icons=["chat-dots", "gear", "info-circle"],
            menu_icon="cast",
            default_index=0
        )
    return selected


def initialize_session_state():
    """
    Initialize session state variables for the chat application.
    """
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def display_chat_messages():
    """
    Display all chat messages from the conversation history.
    """
    for msg in st.session_state.conversation_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def display_message(role, content):
    """
    Display a single message in the chat.
    """
    with st.chat_message(role):
        st.markdown(content)
