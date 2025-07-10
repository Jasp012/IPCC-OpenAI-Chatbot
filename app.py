import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

# Chargement du vecteur index sauvegardé
@st.cache_resource
def load_vectorstore():
    with open("faiss_store.pkl", "rb") as f:
        return pickle.load(f)

st.title("🌍 IPCC Chatbot (GIEC + GPT)")
st.write("Pose une question sur le changement climatique d'après le rapport AR6 du GIEC.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("⚠️ Veuillez définir votre clé OPENAI_API_KEY dans les variables d’environnement.")
else:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = st.text_input("❓ Ta question :")
    if question:
        with st.spinner("Recherche..."):
            result = qa_chain.run(question)
        st.success("✅ Réponse :")
        st.write(result)


