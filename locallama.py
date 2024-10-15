import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

# Set environment variables for LangSmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Set up prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Set up the LLM
llm = Ollama(model="llama3.2:latest")

# Customize the Streamlit page
st.set_page_config(
    page_title="Offline LLM Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Apply custom CSS to style the UI elements
st.markdown("""
    <style>
        /* Set background color and default text color */
        body {
            background-color: #f0f2f6;
            color: #333333;
        }
        .main-title {
            font-size: 3rem;
            color: #4CAF50;
            text-align: center;
            font-family: 'Courier New', Courier, monospace;
        }
        .subheader {
            text-align: center;
            font-size: 1.5rem;
            margin-top: -20px;
            color: #888888;
        }
        .stTextInput {
            border-radius: 10px;
            font-size: 1.1rem;
            padding: 10px;
            margin-bottom: 10px;
        }
        .stTextInput input {
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 10px;
            color: #ffffff; /* Set text color to white */
            background-color: #000000; /* Set background color to black */
        }
        .submit-btn {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1.2rem;
            border-radius: 10px;
            cursor: pointer;
        }
        .submit-btn:hover {
            background-color: #45a049;
        }
        .response-box {
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-size: 1.1rem;
            color: #333333; /* Text color */
        }
    </style>
""", unsafe_allow_html=True)

# Create a custom header
st.markdown("<h1 class='main-title'>Offline LLM Chatbot 🤖</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subheader'>Ask any question and get intelligent responses</h2>", unsafe_allow_html=True)

# Create a form to allow the Enter key to trigger submission
with st.form(key="query_form"):
    input_text = st.text_input("What would you like to know?", placeholder="Type your question here...", key="input_text")
    submit_button = st.form_submit_button(label="Get Answer")

# When the form is submitted
if submit_button and input_text:
    try:
        # Format the prompt for the LLM
        formatted_prompt = prompt.format(question=input_text)
        
        # Get the response from the LLM
        llm_response = llm.invoke(formatted_prompt)
        
        # Check if the response is a string or structured data
        if isinstance(llm_response, str):
            response_content = llm_response
        else:
            response_content = "No valid response received."

        # Display the response in a styled container
        st.markdown(f"""
            <div class='response-box'>
                <strong>Response from Ollama:</strong><br>
                {response_content}
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
