from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ── Model (cached — ek baar load hoga) ──────────────────────────
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        provider="auto",
        huggingfacehub_api_token=hf_token,  # ✅ yahan pass karo
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

# ── Session state init ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── UI ───────────────────────────────────────────────────────────
st.title("🤖 AI Chatbot")

# Sidebar: clear button
with st.sidebar:
    st.markdown("### Settings")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.caption(f"Context turns: {len(st.session_state.chat_history) // 2}")

# Display old messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# New input
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    messages = [SystemMessage(content="You are a helpful assistant. Maintain context across the conversation.")] \
               + st.session_state.chat_history

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = model.invoke(messages)
        st.write(result.content)

    st.session_state.chat_history.append(AIMessage(content=result.content))