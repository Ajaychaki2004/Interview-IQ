import pdfplumber
import streamlit as st
import docx2txt
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import GOOGLE_API_KEY

def app():

    st.title("Generative AI for Technical Interview Preparation from PDF")

    def read_pdf_with_pdfplumber(file):
        with pdfplumber.open(file) as pdf:
            page = pdf.pages[0]
            return page.extract_text()
        
    st.subheader("DocumentFiles")
    def pdf_ex():
        docx_file = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'])
        if docx_file is not None:
            file_details = {"Filename": docx_file.name, "FileType": docx_file.type, "FileSize": docx_file.size}
            # st.write(file_details)

            if docx_file.type == "text/plain":
                raw_text = str(docx_file.read(), "utf-8")
                st.write(raw_text)

            elif docx_file.type == "application/pdf":
                try:
                    text = read_pdf_with_pdfplumber(docx_file)
                    st.write(text)
                except:
                    st.write("Error reading PDF file")

            elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                raw_text = docx2txt.process(docx_file)
                return(raw_text)


    pdf_data=pdf_ex()

    genai.configure(api_key=GOOGLE_API_KEY)

    # Initialize the LLM with LangChain
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create a prompt template
    pdf_prompt = PromptTemplate(
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
    pdf_mcq_chain = LLMChain(
        llm=llm,
        prompt=pdf_prompt,
        output_key="mcq_questions"
    )
    
    if not pdf_data:
        st.write("Upload File")
    else:
        with st.spinner('Generating MCQ questions from PDF...'):
            try:
                # Run the chain
                result = pdf_mcq_chain.invoke({"content": pdf_data})
                
                st.header("MCQ Questions")
                st.write(result["mcq_questions"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Try uploading a different file or refresh the page.")
