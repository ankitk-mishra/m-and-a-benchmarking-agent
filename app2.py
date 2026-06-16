import streamlit as st
import pandas as pd
import sqlite3
import re
import requests
import json
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="M&A Benchmarking Agent", layout="wide")
st.title("📊 Tech M&A Benchmarking AI")

# ==========================================
# 2. DATABASE UTILITIES & RELATIONSHIP HINTS
# ==========================================
@st.cache_resource
def get_database_connection():
    return sqlite3.connect(':memory:', check_same_thread=False)

conn = get_database_connection()

def get_schema():
    """Extracts the database schema and injects explicit relationship hints."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_str = ""
    for table in tables:
        table_name = table[0]
        schema_str += f"Table: {table_name}\nColumns: "
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()
        schema_str += ", ".join([f"{col[1]} ({col[2]})" for col in columns]) + "\n"
    
    # Inject explicit relationship mapping rules for the LLM
    schema_str += """
    \nDATABASE RELATIONSHIP HINTS (CRITICAL FOR JOINs):
    - Table 'primary_kpis' contains company demographic columns like: 'Sector', 'Revenue_m', 'Org_Size', 'IT_Team_Size', 'IT_Spend_m', 'IT_Standalone_Cost_m'.
    - Table 'cost_breakdown' contains granular budget columns like: 'IT_capex_m', 'IT_opex_m', 'IT_Personnel_Cost_m', 'Outsourcing_Cost_m', 'Licensing_Cost_m', 'Infrastructure_Cost_m'.
    - Table 'applications' contains software-level columns like: 'Application_Type', 'Application_Name', 'Vendor', 'Pricing_Model', 'Hosting_Type', 'Annual_Cost'.
    - To map 'Sector' or 'Revenue' to any column in 'cost_breakdown' or 'applications', you MUST execute a SQL JOIN:
      * JOIN cost_breakdown ON cost_breakdown.Project_Name = primary_kpis.Project_Name
      * JOIN applications ON applications.Project_Name = primary_kpis.Project_Name
    """
    return schema_str

def extract_sql(text):
    """Cleans up the text output from any LLM to pull out only the SQL query."""
    match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"SELECT(.*?);", text, re.DOTALL | re.IGNORECASE)
    if match:
        return f"SELECT {match.group(1).strip()};"
    text = text.replace("SQL Query:", "").strip()
    return text

# ==========================================
# 3. SIDEBAR: DATA INGESTION
# ==========================================
st.sidebar.header("1. Upload Project Data")
uploaded_file = st.sidebar.file_uploader("Upload Multi-sheet Excel", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
    sheet_names = xls.sheet_names
    
    with st.sidebar.expander("Loaded Data Models", expanded=True):
        for sheet in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
            
            # Clean whitespaces in text columns
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                
            # Make columns standard SQL compatible (no spaces or special chars)
            df.columns = [str(c).strip().replace(' ', '_').replace('(', '').replace(')', '').replace('#', 'num') for c in df.columns]
            
            table_name = str(sheet).replace(' ', '_').lower()
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            st.success(f"✅ {table_name} ({len(df)} rows)")

# ==========================================
# 4. ENGINE ROUTER: OLLAMA / GROQ / GEMINI / CLAUDE
# ==========================================
st.sidebar.header("2. AI Engine Configuration")
engine_type = st.sidebar.selectbox(
    "Choose AI Infrastructure",
    [
        "Local Ollama", 
        "Cloud API (Groq - Free)", 
        "Cloud API (Gemini - Free)", 
        "Cloud API (Anthropic Claude - Paid)"
    ]
)

# Render fields dynamically based on the selected engine
api_key = ""
model_name = ""

if engine_type == "Local Ollama":
    model_name = st.sidebar.text_input("Ollama Model Name", value="qwen2.5-coder:7b")
    
elif engine_type == "Cloud API (Groq - Free)":
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    model_name = st.sidebar.selectbox("Groq Model", ["llama-3.3-70b-versatile", "qwen/qwen3-32b"])
    st.sidebar.markdown("[Get a Free Groq API Key](https://console.groq.com/keys)")

elif engine_type == "Cloud API (Gemini - Free)":
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    model_name = "gemini-2.5-flash"
    st.sidebar.markdown("[Get a Free Gemini API Key](https://aistudio.google.com/)")

elif engine_type == "Cloud API (Anthropic Claude - Paid)":
    api_key = st.sidebar.text_input("Anthropic API Key", type="password")
    model_name = st.sidebar.selectbox("Claude Model", ["claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022"])
    st.sidebar.markdown("[Get an Anthropic API Key](https://console.anthropic.com/)")

def call_cloud_api(prompt_text):
    """Executes standard HTTP POST requests to bypass heavy SDK installations."""
    if engine_type == "Cloud API (Groq - Free)":
        if not api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API Error: {response.text}")
            
    elif engine_type == "Cloud API (Gemini - Free)":
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt_text}]
            }],
            "generationConfig": {
                "temperature": 0
            }
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise Exception(f"Gemini API Error: {response.text}")

    elif engine_type == "Cloud API (Anthropic Claude - Paid)":
        if not api_key:
            st.error("Please enter your Anthropic API Key in the sidebar.")
            return None
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": model_name,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            # Extract text elements from Claude response payload
            return response.json()["content"][0]["text"]
        else:
            raise Exception(f"Anthropic API Error: {response.text}")
            
    return None

def run_agentic_query(question, messages):
    schema = get_schema()
    if not schema:
        return "⚠️ Please upload an Excel file first so I have data to query.", None

    # Maintain context for up to the last 8 messages
    history_str = ""
    recent_msgs = messages[-9:-1] if len(messages) > 1 else []
    for m in recent_msgs:
        if m["role"] == "user" and m.get("content"):
            history_str += f"User: {m['content']}\n"
    
    if not history_str:
        history_str = "No previous context."

    prompt_template = """
    You are an expert Technology M&A data analyst. 
    Your goal is to write a strictly valid SQLite query to answer the user's question based on the provided database schema.
    
    CRITICAL RULES:
    1. Output ONLY the raw SQL query. 
    2. Do NOT add any explanations, markdown, or text before or after the query.
    3. Use exact column names provided in the schema. Do NOT invent column names. Double-check which table a column belongs to.
    4. For string filtering in WHERE clauses, ALWAYS use the LIKE operator (e.g., WHERE column_name LIKE '%keyword%') to avoid case-sensitivity and trailing space errors. Do NOT use strict equality (=) for strings.
    5. If data needs to be filtered by a column in one table, but you need to return a column from a DIFFERENT table, you MUST use a SQL JOIN connecting them on a common column (like Project_Name). Refer to the relationship hints.
    6. CONTEXT AWARENESS: Read the CHAT HISTORY. If the user asks a follow-up question (like "for the same" or "what about pricing?"), you MUST apply the filters (e.g., specific Sector or Project) mentioned in the previous messages.
    
    SCHEMA:
    {schema}
    
    CHAT HISTORY:
    {chat_history}
    
    USER QUESTION: {question}
    
    SQL QUERY:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    raw_response = "No response generated."
    
    try:
        formatted_prompt = prompt.format(schema=schema, chat_history=history_str, question=question)
        
        # Route to appropriate engine
        if engine_type == "Local Ollama":
            llm = Ollama(model=model_name, temperature=0)
            raw_response = llm.invoke(formatted_prompt)
        else:
            raw_response = call_cloud_api(formatted_prompt)
            if not raw_response:
                return "⚠️ API execution returned an empty response. Verify your API key.", None
                
        sql_query = extract_sql(raw_response)
        result_df = pd.read_sql_query(sql_query, conn)
        
        return "", result_df

    except Exception as e:
        return f"⚠️ Execution Error: {str(e)}\n\n*Raw LLM output was:* {raw_response}", None

# ==========================================
# 5. CHAT INTERFACE & AUTO-VISUALIZATION
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("df") is not None and not msg["df"].empty:
            st.dataframe(msg["df"])
            df = msg["df"]
            if len(df) > 1 and len(df.columns) >= 2:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    st.caption("Auto-Generated Visualization:")
                    st.bar_chart(data=df, x=df.columns[0], y=numeric_cols[0])

if prompt := st.chat_input("E.g., What is the average IT spend by organization size?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            text_response, dataframe = run_agentic_query(prompt, st.session_state.messages)
            if text_response:
                st.markdown(text_response)
            
            if dataframe is not None and not dataframe.empty:
                st.dataframe(dataframe)
                if len(dataframe) > 1 and len(dataframe.columns) >= 2:
                    numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        st.caption("Auto-Generated Visualization:")
                        st.bar_chart(data=dataframe, x=dataframe.columns[0], y=numeric_cols[0])
            elif dataframe is not None and dataframe.empty:
                st.info("The query ran successfully, but returned 0 rows.")
                
    st.session_state.messages.append({"role": "assistant", "content": text_response, "df": dataframe})
