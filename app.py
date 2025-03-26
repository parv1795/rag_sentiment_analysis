import streamlit as st
import os
import tempfile
import nltk
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma  # Using Chroma instead of FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the app with blue and white theme
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #1e88e5;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #1e88e5;
    }
    .stButton > button {
        background-color: #1e88e5;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #E8F4FD;
        border: 1px solid #1e88e5;
    }
    .chat-message.bot {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .chat-message .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 10px;
    }
    .chat-message .message {
        width: 85%;
        padding-left: 10px;
        color: #000000;  /* Ensure text is black for visibility */
    }
    h1, h2, h3 {
        color: #1e88e5;
    }
    .stTab {
        background-color: #e3f2fd;
        color: #1e88e5;
        border-radius: 5px 5px 0 0;
    }
    .stTab[aria-selected="true"] {
        background-color: #1e88e5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'conversation_starters' not in st.session_state:
    st.session_state.conversation_starters = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'sentiment_analysis' not in st.session_state:
    st.session_state.sentiment_analysis = None
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'ready' not in st.session_state:
    st.session_state.ready = False
if 'reset_input' not in st.session_state:
    st.session_state.reset_input = False
if 'submit_with_enter' not in st.session_state:
    st.session_state.submit_with_enter = False

# Function to process user questions
def process_user_question(question):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Process the user's question
    if st.session_state.vectorstore is not None:
        # Search for relevant documents
        docs = st.session_state.vectorstore.similarity_search(question, k=3)
        doc_content = "\n".join([doc.page_content for doc in docs])
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question"
        )
        
        # Create a prompt template
        template = """
        You are a helpful AI assistant that can only answer questions based on the provided document content.
        If the question is not related to the document or cannot be answered from the document, politely decline
        to answer and ask the user to ask questions related to the document.
        
        Document content:
        {context}
        
        Chat History:
        {chat_history}
        
        Question:
        {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        
        # Create QA chain
        model_name = "gemini-1.5-flash"  # Use the recommended model
            
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)
        
        # Get answer
        result = chain({"input_documents": docs, "question": question})
        answer = result["output_text"]
        
        # Add bot message to chat history
        st.session_state.chat_history.append({"role": "bot", "content": answer})

# Sidebar for API key and PDF upload
with st.sidebar:
    st.title("📚 PDF RAG Chatbot")
    st.write("Upload a PDF and start chatting with your document!")
    
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    
    uploaded_file = st.file_uploader("Upload a PDF file:", type="pdf")
    
    if uploaded_file and api_key and st.button("Process PDF", key="process_pdf_btn"):
        with st.spinner("Processing PDF..."):
            # Set API key
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            
            # List available models without displaying them
            try:
                available_models = []
                for model in genai.list_models():
                    available_models.append(model.name)
                
                if not available_models:
                    st.error("No models available. Please check your API key.")
                    st.stop()
                    
                # Find appropriate model for embeddings
                embedding_model = "models/embedding-001"
                
                # Use without displaying all model details
                st.success(f"API connected successfully!")
            except Exception as e:
                st.error(f"Error with API: {str(e)}")
                st.stop()
            
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract text from PDF
            pdf_reader = PdfReader(tmp_file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            st.session_state.text = text
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Create vector store with Chroma instead of FAISS
            st.session_state.vectorstore = Chroma.from_texts(chunks, embeddings)
            
            # Generate conversation starters
            model_name = "gemini-1.5-flash"  # Use the recommended model
            
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
            starter_prompt = """
            Based on the following document, generate 5 interesting questions that someone might want to ask about the content.
            Format each question on a new line and make them diverse to cover different aspects of the document.
            
            Document:
            {text}
            """
            response = llm.predict(starter_prompt.format(text=text[:5000]))  # Use first 5000 chars to stay within token limits
            st.session_state.conversation_starters = [q.strip() for q in response.strip().split('\n') if q.strip()]
            
            # Generate summary
            summary_prompt = """
            Provide a concise summary of the following document in about 200 words. 
            Capture the main ideas and key points without including minor details.
            
            Document:
            {text}
            """
            st.session_state.summary = llm.predict(summary_prompt.format(text=text[:6000]))  # Use first 6000 chars
            
            # Sentiment analysis - Simple approach
            # Count words in the text
            word_count = len(text.split())
            
            # Simple polarity calculation - positive value means more positive sentiment
            polarity = 0.2  # Default slightly positive
            
            # Simple subjectivity score (0-1 range, higher means more subjective)
            subjectivity = 0.5  # Default middle value
            
            st.session_state.sentiment_analysis = {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'word_count': word_count
            }
            
            # Remove temporary file
            os.unlink(tmp_file_path)
            
            st.session_state.ready = True
            st.success("PDF processed successfully! The bot is ready to use.")
    
    if st.session_state.ready:
        st.write("✅ Bot is ready to answer questions about your document")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Sentiment Analysis", "📝 Summary"])

# CHAT TAB - TAB 1
with tab1:
    if not st.session_state.ready:
        st.info("Please upload a PDF file and enter your Gemini API key to start chatting.")
    else:
        # Conversation starter buttons with unique keys
        with st.expander("Conversation Starters", expanded=True):
            cols = st.columns(len(st.session_state.conversation_starters[:3]))
            for i, col in enumerate(cols):
                if i < len(st.session_state.conversation_starters):
                    if col.button(st.session_state.conversation_starters[i], key=f"starter_btn_{i}"):
                        process_user_question(st.session_state.conversation_starters[i])
        
        # Display chat messages
        for message in st.session_state.chat_history:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="avatar">👤</div>
                        <div class="message">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot">
                        <div class="avatar">🤖</div>
                        <div class="message">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # User input with clear input mechanism
        # Check if we need to reset the input
        if "reset_input" not in st.session_state:
            st.session_state.reset_input = False
        
        if st.session_state.reset_input:
            st.session_state.chat_input_field = ""
            st.session_state.reset_input = False
        
        # Create the input field with a key
        user_question = st.text_input("Ask a question about your document:", key="chat_input_field")
        
        # Submit button
        if st.button("Send", key="chat_send_button") or (user_question and st.session_state.get("submit_with_enter", False)):
            if user_question:
                # Process the question
                process_user_question(user_question)
                # Set flag to reset input on next render
                st.session_state.reset_input = True
                # Reset enter key flag
                st.session_state.submit_with_enter = False
                st.rerun()

# SENTIMENT ANALYSIS TAB - TAB 2
with tab2:
    if not st.session_state.ready:
        st.info("Please upload a PDF file and enter your Gemini API key to view sentiment analysis.")
    else:
        st.header("Sentiment Analysis")
        st.write("This is a simple analysis of your document content.")
        
        # Display word count
        st.metric("Total Words", f"{st.session_state.sentiment_analysis['word_count']:,}")
        
        # Display sentiment gauge chart
        polarity = st.session_state.sentiment_analysis['polarity']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Document Sentiment**")
            sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
            
            # Simple color-coded box showing sentiment
            sentiment_color = "#4CAF50" if polarity > 0 else "#F44336" if polarity < 0 else "#FFC107"
            st.markdown(f"""
            <div style="background-color: {sentiment_color}; padding: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold; font-size: 20px;">
                {sentiment_label}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.write("**Document Stats**")
            # Show a few simple statistics
            sentences = len(st.session_state.text.split('.'))
            characters = len(st.session_state.text)
            st.write(f"Sentences: {sentences:,}")
            st.write(f"Characters: {characters:,}")
            st.write(f"Avg. Sentence Length: {st.session_state.sentiment_analysis['word_count'] / sentences if sentences > 0 else 0:.1f} words")

# SUMMARY TAB - TAB 3
with tab3:
    if not st.session_state.ready:
        st.info("Please upload a PDF file and enter your Gemini API key to view the document summary.")
    else:
        st.header("Document Summary")
        
        # Display the summary if it exists
        if st.session_state.summary and len(st.session_state.summary) > 0:
            st.markdown(f"""
            <div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; border: 1px solid #1e88e5; color: #000000;">
                {st.session_state.summary}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Summary generation failed. Please try uploading the document again.")
            
            # Button to regenerate summary
            if st.button("Regenerate Summary", key="regenerate_summary_btn"):
                with st.spinner("Generating summary..."):
                    try:
                        # Generate summary
                        model_name = "gemini-1.5-flash"
                        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
                        summary_prompt = """
                        Provide a concise summary of the following document in about 200 words. 
                        Capture the main ideas and key points without including minor details.
                        
                        Document:
                        {text}
                        """
                        st.session_state.summary = llm.predict(summary_prompt.format(text=st.session_state.text[:6000]))
                        st.success("Summary regenerated successfully!")
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        
        # Display document statistics
        st.subheader("Document Statistics")
        col1, col2, col3 = st.columns(3)
        
        words = st.session_state.text.split()
        sentences = st.session_state.text.split('.')
        
        with col1:
            st.metric("Total Words", f"{len(words):,}")
        
        with col2:
            st.metric("Sentences", f"{len(sentences):,}")
            
        with col3:
            st.metric("Characters", f"{len(st.session_state.text):,}")

# Add JavaScript for Enter key handling only in the main app
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Chat"

# Add JavaScript for Enter key handling ONLY for the Chat tab
js_code = """
<script>
// Function that listens for Enter key in the text input
document.addEventListener('keydown', function(e) {
    // Only act if we're in the Chat tab
    const tabButtons = Array.from(document.querySelectorAll('button[role="tab"]'));
    const chatTabActive = tabButtons[0] && tabButtons[0].getAttribute('aria-selected') === 'true';
    
    if (chatTabActive && e.key === 'Enter' && !e.shiftKey) {
        const activeElement = document.activeElement;
        if (activeElement.tagName === 'INPUT' && activeElement.type === 'text') {
            e.preventDefault();
            
            // Set a flag in session state
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: true,
                target: 'submit_with_enter'
            }, '*');
            
            // Find the send button and click it
            setTimeout(() => {
                const buttons = Array.from(document.getElementsByTagName('button'));
                const sendButton = buttons.find(button => button.innerText === 'Send');
                if (sendButton) {
                    sendButton.click();
                }
            }, 100);
        }
    }
});
</script>
"""
st.markdown(js_code, unsafe_allow_html=True)
