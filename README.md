 # InterviewIQ AI

InterviewIQ AI is an advanced, AI-powered platform designed to streamline the recruitment process by automating resume screening, interview preparation, and candidate assessment. Utilizing machine learning and natural language processing (NLP), AI agents. InterviewIQ AI provides insights into candidate fit, reduces time-to-hire, and enhances the overall experience for both recruiters and candidates.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Prerequisites](#prerequisites)


## Features

- **Resume Parsing & Analysis**: Extracts key information from resumes and assesses candidate suitability based on job descriptions.
- **Interview Preparation**: Offers personalized interview preparation guidance to candidates, including common questions and best practices.
- **Candidate Fit Prediction**: Uses NLP to analyze candidate responses and job requirements, predicting suitability and alignment.
- **Automated Screening**: Automates initial candidate screening to save time and resources for HR teams.
- **Customizable AI Models**: Allows configuration to adapt to industry-specific requirements.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required libraries (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Ajaychaki2004/Interview-IQ.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Interview-IQ
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
     
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to run locally

1. Create a `.env` file in the project root directory with your Google API key:
   ```
   # Get your API key from: https://makersuite.google.com/app/apikey
   GOOGLE_API_KEY=your_api_key_here
   ```


2. Launch the Streamlit application:
   ```bash
   streamlit run main.py
   ```

3. Open your browser and go to:
   ```
   http://localhost:8501
   ```

5. The application has four main features:
   - **PDF**: Upload PDFs for analysis and question generation
   - **URL**: Enter URLs to extract and analyze content
   - **AI TRAINER**: Learn about any topic through AI-guided explanations
   - **HR QUESTION**: Upload resumes to generate targeted interview questions
