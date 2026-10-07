import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

# Page configuration
st.set_page_config(page_title="LangChain QA Chatbot with Groq", page_icon="🚀")

# Title
st.title("Simple LangChain Chatbot with Groq")
st.markdown("Learn LangChain basics with Groq's ultra-fast inference capabilities! 🚀")

with st.sidebar:
    st.header("Settings")

    # Input for Groq API Key
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get your Free Groq API key from https://console.groq.com")

    # Model selection
    model_name = st.selectbox("Model", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"], index=0)

    # Clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the chat model
@st.cache_resource
def get_chain(groq_api_key, model_name):
    if not groq_api_key:
        return None

    # Initialize the ChatGroq model with the provided API key and model name
    llm=ChatGroq(
        api_key=groq_api_key,
        model=model_name,
        temperature=0.7,
        streaming=True,
    )

    # Create prompt template for the chatbot
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant powered by Groq. Answer the user's questions to the best of your ability."),
            ("human", "{input}"),
        ]
    )

    # Create chain
    chain=prompt | llm | StrOutputParser()
    return chain

# Get the chain
chain = get_chain(groq_api_key, model_name)

if not chain:
    st.warning("Please enter your Groq API Key in the sidebar to initialize the chatbot.")
    st.markdown("[Get your Free Groq API key here](https://console.groq.com)")

else:
    ## Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    ## User input
    if question := st.chat_input("Ask me anything!"):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Stream the response from Groq
                for chunk in chain.stream(input=question):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
            