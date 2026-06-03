from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from streamlit_chat import message
from langchain.messages import HumanMessage, AIMessage,SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()
gemini_model=ChatGoogleGenerativeAI(model="gemini-2.0-pro", temperature=0.7,api_key=os.getenv("API_KEY"))
groq_model=ChatGroq(model="gpt-4o", temperature=0.7, api_key=os.getenv("GROQ_API_KEY"))

#agents
def agent():
    agent=create_agent(
    model=gemini_model,
    system_message=SystemMessage(content="You are a helpful assistant that provides information about the latest news and trends in the field of artificial intelligence. You can also answer questions related to AI, provide insights on recent developments, and discuss the impact of AI on various industries. Your responses should be informative, concise, and engaging."),
    )
    return agent
st.title("Chatbot with Gemini and Groq")
msg=st.chat_input("Type your message here...")
if msg:
    message(msg, is_user=True)
    response=agent().run(HumanMessage(content=msg))
    message(response, is_user=False)
