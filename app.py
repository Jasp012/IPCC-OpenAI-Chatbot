import os
import streamlit as st
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings

# Streamlit page config
st.set_page_config(page_title="IPCC Chatbot", page_icon="🌍")
st.title("🌍 IPCC Chatbot (RAG Pipeline)")
st.write("Ask a question about the IPCC AR6 report. The answer is generated using OpenAI and sourced directly from the official report.")

# Load environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY environment variable not found.")
    st.stop()

# Load FAISS vector store
@st.cache_resource
def load_vectorstore():
    with open("faiss_store.pkl", "rb") as f:
        return pickle.load(f)

# Load vector DB and create retriever
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# Set up the LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Prompt template
prompt_template = """Use the following pieces of context from the IPCC report to answer the question at the end.
If you don't know the answer, just say you don't know. Do NOT make up an answer.

Context:
{context}

Question:
{question}
Helpful Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# RAG pipeline: LLMChain inside RetrievalQA
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
rag_pipeline = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever, return_source_documents=True)

# Input box
question = st.text_input("❓ Ask your question:")
if question:
    with st.spinner("🔎 Retrieving answer..."):
        result = rag_pipeline(question)
        st.success("✅ Answer:")
        st.write(result["result"])

        # Optional: Show sources
        with st.expander("📄 Show source documents"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content)
