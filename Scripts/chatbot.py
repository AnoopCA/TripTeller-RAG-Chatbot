import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv(r"D:\ML_Projects\TripTeller-RAG-Chatbot\Keys\.env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
PDF_FILE = r"D:\ML_Projects\TripTeller-RAG-Chatbot\Data\munnar_highres.pdf"
FAISS_FOLDER = r"D:\ML_Projects\TripTeller-RAG-Chatbot\Vector_store\faiss"  # folder to save embeddings for persistence

loader = PyPDFLoader(PDF_FILE)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(FAISS_FOLDER):
    vectorstore = FAISS.load_local(FAISS_FOLDER, embeddings, allow_dangerous_deserialization=True)
    print(f"Loaded existing FAISS index from '{FAISS_FOLDER}'")
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_FOLDER)
    print(f"Created and saved FAISS index to '{FAISS_FOLDER}'")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

#direct_response = llm.invoke(test_question)

prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. 
        Use the following context to answer the user's question. 
        If the context does not contain enough information, answer the question using your general knowledge.
        Context:
        {context}
        Question: {input}
        Answer in a friendly, conversational way.
    """)

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

print("Chatbot ready with PDF knowledge! Type 'exit' to quit.\n")
while True:
    query = input("Ask a question: ")
    if query.lower() == "exit":
        break
    
    response = qa_chain.invoke({"input": query})
    print("\nAnswer:", response["answer"], "\n")
