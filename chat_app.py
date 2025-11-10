# %pip install llama-index llama-index-vector-stores-pinecone


import os
import sys
import io
import logging
from pinecone import Pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import textwrap
from pinecone.grpc import PineconeGRPC as Pinecone
import streamlit as st
from langchain_community.llms import OpenAI
from PyPDF2 import PdfReader

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')  # Renamed for consistency
PDF_FILE_PATH = os.getenv('PDF_FILE_PATH', 'cvFS.pdf')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'quickstart')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))  # Increase default
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))  # Add overlap
# pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
# pinecone_index = os.getenv('PINECONE_INDEX')

# Validate required environment variables
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set")


openai_client = OpenAI(api_key=openai_api_key)
# Initialze the opem embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


pc = Pinecone(api_key=pinecone_api_key)
# Inicializar cliente Pinecone
# index = pc.Index(pinecone_index)

index = pc.Index(PINECONE_INDEX_NAME)

# Inicializar Pinecone usando langchain y pasando el embedding
pinecone_vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

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

# Crear metadata con el nombre del archivo para cada fragmento
metadata = [{"filename": 'cvFS.pdf'} for _ in range(len(text))]

# Dividir el texto en fragmentos y asignar metadatos a cada fragmento
documents = text_splitter.create_documents([text], metadatas=metadata)

indices = [f"{'CV'}_{i+1}" for i in range(len(documents))]

# subo lo vectores de embeddings a pinecone
pinecone_vectorstore.add_documents(documents=documents, ids=indices)

# Model
# Initilize the model
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

# Defining the prompt of the system
system_prompt = (
    "Eres un bot que responde preguntas sobre documentos proporcionados. "
    "Usa únicamente el contexto dado para responder."
    "Si la respuesta no está en el contexto, di: "
    "'No te puedo proporcionar la información, ya que no existe en mi base de datos.'"
    "Sé preciso y conciso.\n\n"
    "Contexto: {context}"
)

# Create the prompt with messages of the system and the usar
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Encapsulate the LLM with the prompt in a string
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True
)

# Create a StuffDocumentsChain to combine documents into a single prompt for the LLM
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,               # The chain that handles prompt + LLM logic
    document_variable_name="context",  # The placeholder used in your prompt template
    # Optional: logs internal steps (good for debugging)
    verbose=True
)

# Configure the retriever from Pinecone
retriever = pinecone_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}
)


# Full question-answering chain with document retrieval and context injection
qa_chain = RetrievalQA(
    # Vector retriever (e.g. Pinecone, FAISS, etc.)
    retriever=retriever,
    # Chain that formats and feeds context to the LLM
    combine_documents_chain=stuff_chain,
    # Set to True if you want to display source docs
    return_source_documents=True,
    # Optional: logs intermediate steps for debugging
    verbose=True
)



# Initializes the conversation history in the session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


st.title("🤖 Chatbot with GPT-3.")
st.subheader("¡Ask a question!")

# Show the last message
for msg in st.session_state.conversation_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Use the input of the chat
user_input = st.chat_input("Ask a question...")

if user_input:
    # Mostrar mensaje del usuario
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
    # Getting the answer of the bot (Using LangChain RetrievalQA)
    response = qa_chain({"query": user_input})
    answer = response["result"]

    # Show the answer of the bot
    st.chat_message("assistant").markdown(answer)

    # Adding the bot answer to the record
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": answer})
