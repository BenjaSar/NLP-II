# %pip install llama-index llama-index-vector-stores-pinecone
# %pip install langchain-google-genai

# Package import
import os
import sys
import io
import logging
from pinecone import Pinecone
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_classic.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import textwrap
import streamlit as st
from utils.streamlit import initialize_session_state, display_chat_messages, display_message, configure_streamlit_page, display_sidebar_menu
# from langchain_community.llms import OpenAI
from PyPDF2 import PdfReader

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory

from langchain_classic.chains import ConversationalRetrievalChain
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Configuration
# openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY') # New variable for Google API key
pinecone_api_key = os.getenv('PINECONE_API_KEY')  # Renamed for consistency
PDF_FILE_PATH = os.getenv('PDF_FILE_PATH', 'ai_engineer_cv.pdf')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'quickstart')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))  # Increase default
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))  # Add overlap

# Validate required environment variables
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY environment variable is not set")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set")


# openai_client = OpenAI(api_key=openai_api_key)
# Initialze the opem embeddings
# embeddings = OpenAIEmbeddings(google_api_key=google_api_key)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

pc = Pinecone(api_key=pinecone_api_key)

# Inicializar cliente Pinecone
index = pc.Index(PINECONE_INDEX_NAME)

# Inicializar Pinecone usando langchain y pasando el embedding
pinecone_vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# Check if we already have vectors in the index to avoid re-uploading
# (A simple check is to see if the index has any stats, or use a session_state flag)
index_stats = index.describe_index_stats()

# Only ingest if the index is empty. 
# Alternatively, comment this whole block out after running it once.
if index_stats.total_vector_count == 0:
    print("Index is empty. Processing PDF and uploading...")

    try:
        if not os.path.exists(PDF_FILE_PATH):
            raise FileNotFoundError(f"PDF file not found: {PDF_FILE_PATH}")

        with open(PDF_FILE_PATH, 'rb') as f:
            pdf_content = f.read()
    except FileNotFoundError as e:
        st.error(f"Configuration error: {e}")
        st.stop()
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        st.error("An error occurred while processing the document.")
        st.stop()

    # Usar PdfReader para extraer el texto del PDF
    # Utilizamos BytesIO para procesar el contenido binario
    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Using the textsplitter of Langchaing for extracting of chunks of the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=0)

    # Create metadata with the file name for each fragment
    # metadata = [{"filename": 'cvFS.pdf'} for _ in range(len(text))]
    metadata = [{"filename": 'ai_engineer_cv.pdf'} for _ in range(len(text))]

    # Divide the text in fragments and assigne metadata to each frament
    documents = text_splitter.create_documents([text], metadatas=metadata)

    indices = [f"{'CV'}_{i+1}" for i in range(len(documents))]

    # Upload the embedding vectors to pinecone
    # pinecone_vectorstore.add_documents(documents=documents, ids=indices)

    # Definir el tamaño del lote y el retraso para respetar el límite de 100 RPM
    BATCH_SIZE = 50 
    DELAY_SECONDS = 3.5 # Wait 3.5 seconds between batches (safer than 5s, but faster)
    import time
    print(f"Uploading {len(documents)} documents to Pinecone in batches of {BATCH_SIZE}...")

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        batch_indices = indices[i:i + BATCH_SIZE]
        
        print(f"-> Adding batch {i//BATCH_SIZE + 1} (Documents {i+1} to {min(i + BATCH_SIZE, len(documents))})")

        try:
            pinecone_vectorstore.add_documents(documents=batch, ids=batch_indices)
        except Exception as e:
            print(f"🛑 Error during batch {i}: {e}. Stopping process.")
            # Print the error but allow a graceful exit
            break
            
        # Wait to avoid hitting the rate limit of 100 RPM
        if (i + BATCH_SIZE) < len(documents):
            print(f"Pausing for {DELAY_SECONDS} seconds...")
            time.sleep(DELAY_SECONDS) 

    print("\n✅ Document upload complete (or stopped due to error/limit).")
else:
    print("✅ Documents already indexed. Skipping upload.")

# Model
# Initilize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
# llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)

# Read the prompt file
with open("prompt.md", "r", encoding="utf-8") as f:
    prompt_text = f.read()
    # Defining the prompt of the system
    system_prompt =prompt_text

# Create the prompt with messages of the system and the user
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create a document processing chain that combines retrieved documents with the LLM
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template
)

# Configure the retriever from Pinecone
retriever = pinecone_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}
)

# Full question-answering chain with document retrieval and context injection
qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain
)

# Configure Streamlit page
configure_streamlit_page()

# Display sidebar menu
display_sidebar_menu()

# Initialize session state
initialize_session_state()


st.title("🤖 Chatbot with Gemini 2.5 Flash-Lite.")
st.subheader("¡Ask a question!")

# Show the last message
for msg in st.session_state.conversation_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Use the input of the chat
user_input = st.chat_input("Ask a question...")

if user_input:
    # Show the message to the user
    if len(user_input.strip()) == 0:
        st.warning("Please enter a valid question.")
        st.stop()
    
    if len(user_input) > 1000:  # Prevent extremely long inputs
        st.warning("Question is too long. Please limit to 1000 characters.")
        st.stop()

    st.chat_message("user").markdown(user_input)

    # Adding the user message to the record
    st.session_state.conversation_history.append(
        {"role": "user", "content": user_input})
    
    user_input = user_input.strip()
    # Getting the answer of the bot (Using LangChain create_retrieval_chain)
    response = qa_chain.invoke({"input": user_input})
    answer = response["answer"]

    # Show the answer of the bot
    st.chat_message("assistant").markdown(answer)

    # Adding the bot answer to the record
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": answer})
