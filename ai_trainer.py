from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph
import streamlit as st
from typing import Dict, Any, TypedDict, Optional
from config import GOOGLE_API_KEY


# Define the state as a TypedDict
class LearningState(TypedDict, total=False):
    topic: str
    explanation: str
    examples: str
    quiz: str
    summary: str

def app():
    # Initialize the LLM
    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        verbose=True,
        temperature=0.5,
        google_api_key=GOOGLE_API_KEY )

    topic = st.chat_input("Enter the topic you want to learn about:")
    st.info(topic)


    # Define the prompt templates for each task
    explain_prompt = PromptTemplate(
        input_variables=["topic"],
        template="""You are an expert Learning Specialist.
        
        Your goal is to teach users about a given topic in a simple, clear, and engaging manner.
        You aim to simplify complex concepts, use relatable examples, and ensure the user understands key points.
        
        Please explain the key concepts of the topic: {topic} in simple and clear terms.
        """
    )
    
    examples_prompt = PromptTemplate(
        input_variables=["topic", "explanation"],
        template="""You are an expert Learning Specialist.
        
        Based on this explanation: {explanation}
        
        Provide easy-to-understand examples related to {topic} to help illustrate key concepts.
        """
    )
    
    quiz_prompt = PromptTemplate(
        input_variables=["topic", "explanation", "examples"],
        template="""You are an expert Learning Specialist.
        
        Based on this explanation: {explanation}
        
        And these examples: {examples}
        
        Create an interactive quiz to reinforce learning about {topic}.
        Include 5 questions with answers that test understanding of {topic}.
        """
    )
    
    summary_prompt = PromptTemplate(
        input_variables=["topic", "explanation", "examples", "quiz"],
        template="""You are an expert Learning Specialist.
        
        Based on this information:
        Explanation: {explanation}
        Examples: {examples}
        Quiz: {quiz}
        
        Generate a simplified summary of {topic} to help reinforce learning.
        Cover only the most important points and keep it concise.
        """
    )
    
    # Create LLMChains for each task
    explain_chain = LLMChain(llm=llm, prompt=explain_prompt, output_key="explanation")
    examples_chain = LLMChain(llm=llm, prompt=examples_prompt, output_key="examples")
    quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt, output_key="quiz")
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

    # Define initial state - use a dictionary instead of a function
    initial_state: LearningState = {}
    if topic and topic.strip():
        initial_state = {"topic": topic}
    
    # Define the LangGraph workflow with the TypedDict schema
    workflow = StateGraph(LearningState)
    
    # Define the nodes for our graph
    def run_explanation(state: LearningState) -> LearningState:
        if not state.get("topic"):
            return {"topic": "", "explanation": "", "examples": "", "quiz": "", "summary": ""}
        result = explain_chain.invoke({"topic": state["topic"]})
        return {"topic": state["topic"], "explanation": result["explanation"], "examples": "", "quiz": "", "summary": ""}
    
    def run_examples(state: LearningState) -> LearningState:
        if not state.get("explanation"):
            return state
        result = examples_chain.invoke({"topic": state["topic"], "explanation": state["explanation"]})
        new_state = dict(state)
        new_state["examples"] = result["examples"]
        return new_state
    
    def run_quiz(state: LearningState) -> LearningState:
        if not state.get("examples"):
            return state
        result = quiz_chain.invoke({
            "topic": state["topic"], 
            "explanation": state["explanation"], 
            "examples": state["examples"]
        })
        new_state = dict(state)
        new_state["quiz"] = result["quiz"]
        return new_state
    
    def run_summary(state: LearningState) -> LearningState:
        if not state.get("quiz"):
            return state
        result = summary_chain.invoke({
            "topic": state["topic"], 
            "explanation": state["explanation"], 
            "examples": state["examples"],
            "quiz": state["quiz"]
        })
        new_state = dict(state)
        new_state["summary"] = result["summary"]
        return new_state
    
    # Add nodes to graph
    workflow.add_node("explanation", run_explanation)
    workflow.add_node("examples", run_examples)
    workflow.add_node("quiz", run_quiz)
    workflow.add_node("summary", run_summary)
    
    # Connect the nodes in sequence
    workflow.add_edge("explanation", "examples")
    workflow.add_edge("examples", "quiz")
    workflow.add_edge("quiz", "summary")
    
    # Set the entry point
    workflow.set_entry_point("explanation")
    
    # Compile the graph
    app_graph = workflow.compile()
    
    # Execute the graph only if a topic has been provided
    if topic:
        try:
            # Pass the initial state to invoke
            result = app_graph.invoke(initial_state)
            
            if "explanation" in result:
                st.header("Explanation")
                st.write(result["explanation"])
            
            if "examples" in result:
                st.header("Examples")
                st.write(result["examples"])
            
            if "quiz" in result:
                st.header("Quiz")
                st.write(result["quiz"])
            
            if "summary" in result:
                st.header("Summary") 
                st.write(result["summary"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Try a different topic or refresh the page.")
    else:
        st.write("Please enter a topic to begin the learning process.")


