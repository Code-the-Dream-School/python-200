from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

# Load environment and client
if load_dotenv():
    st.markdown("✅ Loaded environment variables from .env file")
else:
    st.markdown("⚠️ Environment variable failed to load. Check your .env file.")
    
client = OpenAI()

st.set_page_config(page_title="SalesBot", page_icon="👟")
st.title("SalesBot")
st.caption("A friendly sales assistant that will help you find the perfect shoes.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Session Controls")
    if st.button("End Chat"):
        st.session_state.clear()
        st.success("Chat ended. Click Start New Chat to begin again.")
    if st.button("Start New Chat"):
        st.session_state.clear()
        st.rerun()

# --- Initialize conversation ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system",
         "content": (
             "You are a friendly, competent sales assistant who helps customers find shoes "
             "that fit their needs. Ask questions to learn their shoe size, style preferences, "
             "and use case before recommending options."
         )},
        {"role": "assistant",
         "content": (
             "Hi there! I'm here to help you find the perfect pair of shoes. "
             "What kind of shoes are you looking for — running, casual, hiking, or something else?"
         )}
    ]

# --- Display chat history (skip system message) ---
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle new input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                max_completion_tokens=300
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
