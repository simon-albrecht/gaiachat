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
# headers = {
#     "authorization":st.secrets['OPENAI_API_KEY'],
#     "content-type":"application/json"
#     }
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title('Gaia chatbot')
question = st.text_input("Write a question about GAIA: ", key="input")

embedding_model = HuggingFaceEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embedding_model)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

#question = 'Where is the GAIA spacecraft?'

docs = vectorstore.similarity_search(question,k=5)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
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
    for rd in result["source_documents"]:
        st.write(rd.metadata)
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])