from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import openai
#from streamlit_chat import message

#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv(), override=True)
headers = {
    "authorization":st.secrets['OPENAI_API_KEY'],
    "content-type":"application/json"
    }
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title('Gaia chatbot')
question = st.text_input("Write a question about GAIA: ", key="input")

@st.cache_resource
def load_vectors():
    embedding_model = HuggingFaceEmbeddings()
    return FAISS.load_local("faiss_index", embedding_model)

vectorstore = load_vectors()

@st.cache_resource
def load_llm():
    return ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
llm = load_llm()
#question = 'Where is the GAIA spacecraft?'

#docs = vectorstore.similarity_search(question,k=5)

template = """Use the following pieces of context to answer the question at the end. Be helpful. Volunteer additional information where relevant, but keep it concise. Don't try to make up answers that are not supported by the context. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

if question:
    result = qa_chain({"query": question})
    st.write(result["result"])
    st.write('\n')
    st.write('Sources:')
    for num, rd in enumerate(result["source_documents"]):
        st.write(str(num+1)+') '+str(rd.metadata["title"]))
        section_info = []
        for item in rd.metadata:
            if item.startswith('Header'):
                section_info.append(rd.metadata[item])
        if rd.metadata.get("paragraph"):
            section_info.append('paragraph: '+rd.metadata["paragraph"])
        st.write('   (Section: '+', '.join(section_info)+')')
        st.write(rd.metadata["link"])
        st.write(rd)
        st.write('\n')
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])