# 🌍 IPCC + OpenAI Chatbot

An intelligent chatbot that allows users to ask natural language questions about the IPCC reports using generative AI and semantic search.

---

## 🚀 Project Goal

This project enables interactive exploration of complex scientific documents from the IPCC (specifically the AR6 Synthesis Report) through a conversational interface powered by GPT-3.5/4, LangChain, and FAISS.  
The user asks a question, and the chatbot answers with contextually relevant content extracted directly from the PDF.

---

## 🧠 Technologies Used

- **LangChain**: to orchestrate the retrieval pipeline
- **OpenAI API**: for generating responses
- **FAISS**: for semantic vector search
- **PyPDF2**: to read the IPCC PDF
- **Tiktoken**: for token-efficient text chunking
- **(Optional)** Streamlit for a web interface

---

## 🛠️ Installation

1. Clone the repository:

git clone https://github.com/Jasp012/IPCC-OpenAI-Chatbot.git
cd IPCC-OpenAI-Chatbot

2. Install dependencies:

pip install -r requirements.txt

3. Set your OpenAI key:

export OPENAI_API_KEY=your_api_key

4. Download the IPCC AR6 report PDF:
   
https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_FullReport.pdf
Rename it to 2023_AR6_SYR_FullReport.pdf and place it in the root of the project.

### Question Example 

question = "What are the key risks of climate change according to the IPCC?"

### Web interface (Optional)
streamlit run app.py