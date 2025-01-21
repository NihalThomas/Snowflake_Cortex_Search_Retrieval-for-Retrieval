import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import random
import plotly.express as px
import re  # Regular expressions to extract numeric score

# Snowflake session setup
session = get_active_session()
pd.set_option("max_colwidth", None)

# Role-to-source table mapping
roles_to_sources = {
    "Python Developer": "docs_python_table",
    "Network Engineer": "docs_network_table",
    "Data Engineer": "docs_dbms_table"
}

# Questions for each role
role_questions = {
    "Python Developer": [
        "What are Python decorators, and how do they work?",
        "Explain the difference between deep copy and shallow copy in Python.",
        "How does Python's garbage collection mechanism work?",
        "What is the difference between lists and tuples in Python?",
        "Can you explain Python's GIL (Global Interpreter Lock)?"
    ],
    "Network Engineer": [
        "What is the OSI model, and why is it important?",
        "Explain the difference between TCP and UDP.",
        "How do you configure a VLAN in a network switch?",
        "What is NAT, and why is it used?",
        "Describe the purpose of a subnet mask in networking."
    ],
    "Data Engineer": [
        "What are the benefits of using a distributed computing system like Hadoop?",
        "Explain ETL (Extract, Transform, Load) processes.",
        "What is the difference between OLAP and OLTP databases?",
        "How do you ensure data quality during data ingestion?",
        "What is the role of a data lake in a data engineering pipeline?"
    ]
}

# Function to generate ideal answers using RAG
def generate_ideal_answer(question, role):
    table = roles_to_sources.get(role)
    if not table:
        st.error(f"No document source found for the role: {role}")
        return None
    
    cmd = f"""
    WITH results AS (
        SELECT chunk,
            VECTOR_COSINE_SIMILARITY({table}.chunk_vec,
                SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?)) AS similarity
        FROM {table}
        ORDER BY similarity DESC
        LIMIT 3
    )
    SELECT chunk FROM results;
    """
    df_context = session.sql(cmd, params=[question]).to_pandas()
    context = " ".join(df_context['CHUNK'].tolist())
    
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer: "
    cmd2 = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response"
    df_response = session.sql(cmd2, params=["mistral-large", prompt]).collect()
    return df_response[0].RESPONSE

# Function to extract numeric score from RAG response
def extract_numeric_score(response):
    match = re.search(r'\d+(\.\d+)?', response)
    if match:
        return float(match.group(0))
    else:
        return 0.0

# Function to evaluate candidate response
def evaluate_response(candidate_answer, ideal_answer):
    prompt = f"Compare the following answers and assign a score between 0 and 10:\n\nIdeal Answer: {ideal_answer}\n\nCandidate Answer: {candidate_answer}\n\nScore: "
    cmd = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response"
    df_score = session.sql(cmd, params=["mistral-large", prompt]).collect()
    score = extract_numeric_score(df_score[0].RESPONSE.strip())
    return score

# Function to save response to the database
def save_response(application_id, name, role, question, candidate_answer, ideal_answer, score):
    cmd = f"""
    INSERT INTO candidates_table (application_id, name, role, question, candidate_answer, ideal_answer, score)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    session.sql(cmd, params=[application_id, name, role, question, candidate_answer, ideal_answer, score]).collect()

# Function to check if candidate has already attended the test
def check_if_already_attended(application_id):
    cmd = f"SELECT COUNT(*) AS count FROM candidates_table WHERE application_id = ?"
    df = session.sql(cmd, params=[application_id]).to_pandas()
    return df['COUNT'][0] > 0

# Main Streamlit app
st.title("RAG-Based Interview Application")

# Home Page
page = st.radio("Choose your role:", ["Candidate", "Recruiter"])

if page == "Candidate":
    st.header("Candidate Page")
    application_id = st.text_input("Enter your Application ID:")
    name = st.text_input("Enter your Name:")
    role = st.selectbox("Choose the role you are interviewing for:", list(roles_to_sources.keys()))
    
    if application_id:
        # Check if the candidate already attended the test
        if check_if_already_attended(application_id):
            st.warning("You already attended the test")
        else:
            if st.button("Start Interview") and name and role:
                st.session_state.role = role
                st.session_state.questions = random.sample(role_questions[role], 3)
                st.session_state.current_question_index = 0
                st.session_state.answers = [None] * 3

            # Initialize session state variables if not already initialized
            if "questions" in st.session_state and "current_question_index" in st.session_state:
                if st.session_state.current_question_index < len(st.session_state.questions):
                    question = st.session_state.questions[st.session_state.current_question_index]
                    st.write(f"**Question {st.session_state.current_question_index + 1}:** {question}")
                    
                    candidate_answer = st.text_area(f"Your Answer for Question {st.session_state.current_question_index + 1}:", key=f"answer_{st.session_state.current_question_index}")
                    
                    # Handle Next Question
                    if st.button("Next Question") and candidate_answer:
                        st.session_state.answers[st.session_state.current_question_index] = candidate_answer
                        if st.session_state.current_question_index < 2:
                            st.session_state.current_question_index += 1
                        else:
                            st.success("You have completed the interview. Please submit your answers.")

                # Only show the submit button when all answers are filled
                if st.session_state.current_question_index == 2 and all(st.session_state.answers):
                    if st.button("Submit Answers"):
                        for i in range(3):
                            question = st.session_state.questions[i]
                            candidate_answer = st.session_state.answers[i]
                            ideal_answer = generate_ideal_answer(question, st.session_state.role)
                            if ideal_answer:
                                score = evaluate_response(candidate_answer, ideal_answer)
                                save_response(application_id, name, st.session_state.role, question, candidate_answer, ideal_answer, score)
                        st.success("Your answers have been submitted successfully!")

elif page == "Recruiter":
    st.title("Recruiter Dashboard")
    st.write("Performance of Candidates:")
    # Dropdown to choose roles
    role_options = st.selectbox("Choose candidate role:", ("ALL", "Python Developer", "Network Engineer", "Data Engineer"))
    
    # SQL Query Logic
    if role_options == "ALL":
        query = "SELECT application_id, name, role, SUM(score) AS total_score FROM candidates_table GROUP BY application_id, name, role;"
    else:
        query = f"SELECT application_id, name, SUM(score) AS total_score FROM candidates_table WHERE role = '{role_options}' GROUP BY application_id, name;"
    
    # Execute query and convert to pandas dataframe
    candidates_df = session.sql(query).to_pandas()

    # Apply some custom styles
    st.dataframe(candidates_df)

        # Button to clear data
    if st.button("Clear Data"):
        try:
            session.sql("TRUNCATE TABLE candidates_table").collect()
            st.success("All data in 'candidates_table' has been cleared successfully!")
        except Exception as e:
            st.error(f"An error occurred while clearing the table: {e}")
