import requests
from bs4 import BeautifulSoup
import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import GOOGLE_API_KEY

def app():

    st.title("Generative AI for Technical Interview Preparation from URL")
    data=st.text_input("Enter")

    def extract_data(url):
        if data=="":
            st.write("")
        else:
            with st.spinner('Extracting data...'):
                response = requests.get(url)
                response.raise_for_status()  
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                # st.write(paragraphs)
                for para in paragraphs:
                    return(para.get_text())
                code_elements = soup.find_all('code')
                for code in code_elements:
                    return(code.get_text())

    ex_data=extract_data(data)
    # st.write(ex_data)
    # user_input=st.text_input("content")
    st.write(ex_data)

    # # genai.configure(api_key='AIzaSyARRLeR7znnU8USoqfYiBPMuH_8Yafpiqk')
    # model = genai.GenerativeModel("gemini-1.5-flash")
    # chat = model.start_chat(
    #     history=[
    #         {"role": "user", "parts": "Hello"},
    #         {"role": "model", "parts":pt},
    #     ]
    # )
    # response = chat.send_message(in_data)
    # st.write(response.text)


    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Initialize the LLM with LangChain
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create a prompt template
    interview_prompt = PromptTemplate(
        input_variables=["content"],
        template="""You are a technical assistant designed to provide short and accurate answers to technical questions for interview preparation.
        Your goal is to deliver concise, clear, and precise responses that directly address the user's query without unnecessary detail.
        Focus on accuracy and brevity, ensuring the information is directly relevant to the question asked.
        The answers should be elaborated in point form.
        Help the user with the syntax to understand more effectively and generate example code for the topic.
        
        Content: {content}
        
        Generate 10 MCQ questions for the user with answers and explanations. Don't repeat the same question.
        Format as:
        
        Question 1:
        A. 
        B.
        C.
        D.
        Answer: 
        Explanation:
        
        Question 2:
        ...
        """
    )
    
    # Create a chain
    mcq_chain = LLMChain(
        llm=llm,
        prompt=interview_prompt,
        output_key="mcq_questions"
    )

    if data=="":
        st.write("Enter url")
    else:
        with st.spinner('Generating MCQ questions...'):
            try:
                # Run the chain
                if ex_data:
                    result = mcq_chain.invoke({"content": ex_data})
                    
                    st.header("MCQ Questions")
                    st.write(result["mcq_questions"])
                else:
                    st.warning("No content was extracted from the URL. Please try a different URL.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Try a different URL or refresh the page.")

