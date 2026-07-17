#module declaration
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#creating my promt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. Please respond to the Question asked"),
        ("user","Question:{question}")

    ]
)

#Streamlit framework
st.title("Shre AI")
input_text = st.text_input("Hey, how can I help you ? ")

#Ollama framework along with gemma2:latest LLM Model
llm = Ollama(model = "gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

#Gpt Output
if input_text:
    st.write(chain.invoke({"question":input_text}))

