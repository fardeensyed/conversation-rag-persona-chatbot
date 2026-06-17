from __future__ import annotations

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    raise SystemExit("Streamlit is not installed. Run: pip install -r requirements.txt")

from src.chatbot import chat


st.set_page_config(page_title="Conversation RAG Chatbot", layout="wide")
st.title("Conversation RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Try these")
    examples = [
        "What kind of person is User 1?",
        "What are User 2's habits?",
        "How does User 1 talk?",
        "What topics do they discuss most?",
        "What personal facts are mentioned?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            answer = chat(example)
            st.session_state.history.append(("user", example))
            st.session_state.history.append(("assistant", answer))

query = st.chat_input("Ask about the users or their conversations...")
if query:
    answer = chat(query)
    st.session_state.history.append(("user", query))
    st.session_state.history.append(("assistant", answer))

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)
