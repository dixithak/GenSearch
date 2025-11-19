import streamlit as st
import time
from datetime import datetime
from gensearch.helper import get_pdf_text, get_text_chunks, get_vectorstore,conversational_chain
from gensearch.config import available_models, get_model_api_name


def user_input(query):
    # Ensure conversation exists
    if st.session_state.conversation is None:
        st.warning("Please upload and process documents first.")
        return

    # Call your conversational chain
    response = st.session_state.conversation({"question": query})
    st.session_state.chat_history = response['chat_history']

    # Make sure timestamps list exists and is a list
    if 'timestamps' not in st.session_state or st.session_state.timestamps is None:
        st.session_state.timestamps = []

    # Add timestamps for any new messages
    current_len = len(st.session_state.timestamps)
    needed = len(st.session_state.chat_history) - current_len
    if needed > 0:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.timestamps.extend([now_str] * needed)
    for i, message in enumerate(st.session_state.chat_history):
        role = st.session_state.user_name if i % 2 == 0 else "GenSearch"
        timestamp = st.session_state.timestamps[i]
        st.markdown(f"**{i+1}. {role} ({timestamp}):** {message.content}")

def main():
    st.set_page_config("GenSearch Application")
    st.header("GenSearch Application")
    st.write("Welcome to the GenSearch app!")
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "User"
    st.session_state.user_name = st.text_input("Enter your name:", st.session_state.user_name)

    if 'model_selected' not in st.session_state:
        st.session_state.model_selected = None
    st.session_state.model_selected = st.selectbox("Select model", available_models())
    st.write(f"You have selected: {get_model_api_name(st.session_state.model_selected)}")

    query = st.text_input("Enter your question here:")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = None
    if query:
        user_input(query)

    with st.sidebar:
        st.title("Navigation")
        docs = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"], accept_multiple_files = True)
        if st.button("Process Documents"):
            if docs:
                st.success(f"Uploaded {len(docs)} documents successfully!")
                with st.spinner("Processing documents..."):

                    text = get_pdf_text(docs)
                    text_chunks = get_text_chunks(text)
                    vector_embeddings = get_vectorstore(text_chunks)
                    st.session_state.conversation = conversational_chain(vector_embeddings, st.session_state.model_selected)
                    time.sleep(2)  # Simulate processing time
                st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one document.")
        



if __name__== '__main__':
    main()