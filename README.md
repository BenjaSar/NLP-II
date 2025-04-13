
![header](doc/imgs/LogoHeader.png)

###### Author: FS

## LARGE LANGUAGE MODEL
-------------------------

### Assignment 1

#### GPT-3.5 Document QA Chatbot 🧠 

This project is an intelligent chatbot built with OpenAI's GPT-3.5-turbo, LangChain, and Streamlit. It allows users to ask questions and get answers based on the content of vectorized documents using semantic search.

##### Features 

✅ Chat interface with Streamlit

🔍 Document-based question answering

🤖 GPT-3.5-turbo integration

🧠 Contextual retrieval using vector stores (e.g. Pinecone or FAISS)

📚 Source document referencing (optional)

💾 Persistent conversation history (in session)

#### Installation  📦 

1. Clone the repo

`git clone https://github.com/yourusername/gpt-doc-chatbot.git
cd gpt-doc-chatbot`

2. Install dependencies

We recommend using a virtual environment:

`pip install -r requirements.txt`

3. Add your environment variables
Create a .env file or export the following:

```
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_environment
```

Or you can pass api_key directly in code if you're testing.

#### Running  the app ▶️ 

`streamlit run app.py`

#### Demo 📸 

Run the video showed in this repository.


#### License 
MIT License ©

![header](doc/imgs/LogoFooter.png)




