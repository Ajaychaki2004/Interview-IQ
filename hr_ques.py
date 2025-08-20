import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph
from typing import Dict, Any, TypedDict, Optional
from config import GOOGLE_API_KEY


# Define the state as a TypedDict
class HRState(TypedDict, total=False):
    resume_data: str
    hr_questions: str

def app():
    # Configure Genai with API key from environment
    genai.configure(api_key=GOOGLE_API_KEY)

    # Create a generative model instance
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Upload either an image or a PDF file
    uploaded_file = st.file_uploader("Upload an image or a PDF", type=["jpg", "jpeg", "png", "pdf"])
    
    # Initialize response to avoid UnboundLocalError
    response = None

    if uploaded_file:
        # Determine the file type based on the file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Prepare the prompt
        prompt = (
        "Extract the text from the provided {file_type}."
        " Present in a summarised point-by-point format:"
        )

        if file_extension in [".jpg", ".jpeg", ".png"]:
            # Save the uploaded image temporarily
            temp_file_path = "temp_image.jpg"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Upload the image to the generative model
            uploaded_file_path = genai.upload_file(temp_file_path)
            
            # Generate content based on the image and prompt
            response = model.generate_content([
                uploaded_file_path,
                "\n\n",
                prompt.format(file_type="image")
            ])

        elif file_extension == ".pdf":
            # Save the uploaded PDF temporarily
            temp_file_path = "temp_pdf.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Upload the PDF to the generative model
            uploaded_file_path = genai.upload_file(temp_file_path)
            
            # Generate content based on the PDF and prompt
            response = model.generate_content([
                uploaded_file_path,
                "\n\n",
                prompt.format(file_type="PDF")
            ])
    
    # Check if response contains data before accessing it
    if response:
        # st.write(response.text)
        pdf_data = response.text
    else:
        pdf_data = None
        st.write("No file was uploaded or there was an error processing the file.")
    
    # If pdf_data is not None, generate HR questions
    if pdf_data:
        google_llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            verbose=True,
            temperature=0.5,
            google_api_key=GOOGLE_API_KEY
        )

        # Define prompt template for HR questions
        hr_prompt = PromptTemplate(
            input_variables=["resume_data"],
            template="""You are an experienced Hiring Agent with over 10 years of experience in talent acquisition across various industries.
            
            Your goal is to conduct a structured interview based on extracted skills from a resume.
            
            Generate a list of HR interview questions for the following resume data:
            {resume_data}
            
            Focus on evaluating:
            1. Technical skills
            2. Relevant experience
            3. Cultural fit
            4. Problem-solving abilities
            5. Communication skills
            
            Format the output as a structured list of questions organized by category.
            """
        )
        
        # Create LLMChain for HR question generation
        hr_chain = LLMChain(llm=google_llm, prompt=hr_prompt, output_key="hr_questions")
        
        # Define initial state - use a dictionary instead of a function
        initial_state: HRState = {"resume_data": pdf_data}
        
        # Define the LangGraph workflow with a TypedDict schema
        workflow = StateGraph(HRState)
        
        # Define the node for our graph
        def generate_hr_questions(state: HRState) -> HRState:
            if not state.get("resume_data"):
                return {"resume_data": "", "hr_questions": ""}
            result = hr_chain.invoke({"resume_data": state["resume_data"]})
            return {"resume_data": state["resume_data"], "hr_questions": result["hr_questions"]}
        
        # Add node to graph
        workflow.add_node("hr_questions", generate_hr_questions)
        
        # Set the entry point
        workflow.set_entry_point("hr_questions")
        
        # Compile the graph
        hr_graph = workflow.compile()
        
        # Execute the graph
        try:
            result = hr_graph.invoke(initial_state)
            
            if "hr_questions" in result:
                st.header("HR Interview Questions")
                st.write(result["hr_questions"])
            else:
                st.warning("No questions could be generated. The resume might not contain enough information.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Try uploading a different file or refresh the page.")
    else:
        st.write("Please upload a valid file to begin the learning process.")
