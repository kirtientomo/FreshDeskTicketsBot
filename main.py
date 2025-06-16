import streamlit as st
import pandas as pd
import mysql.connector
from config import DB_CONFIG
from parser import extract_emp_id
import requests
from datetime import datetime
from requests.auth import HTTPBasicAuth
from dateutil import parser as dateparser
import pytz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama
import re
from pandas import json_normalize
 
# Load GGUF model
llm_local = Llama(
    model_path="c:\\Users\\KirtiS\\Downloads\\gemma-3-4b-it-Q4_0.gguf",
    verbose=False,
    n_ctx=15600
)
 
# Summarization
def summarize_with_gguf(description):
    prompt = f"Please summarize the following ticket description in one short sentence:\n\n{description}"
    output = llm_local(prompt, max_tokens=100)
    summary = output['choices'][0]['text'].strip()
    if summary.lower().startswith("please summarize") or summary.lower().startswith(description[:20].lower()):
        summary = summary.split("\n")[-1].strip()
    return summary
 
# Generate Response
def generate_response_from_kb(subject, description, kb_entry_text, kb_metadata):
    prompt = f"""
You are a support assistant. Based on the following ticket and the provided knowledge base entry and metadata, generate a helpful and accurate response.
 
⚠️ Important: Do not use external knowledge. Only refer to the information provided in the KB entry and metadata.
 
Ticket Subject: {subject}
Ticket Description: {description}
 
Knowledge Base Entry:
{kb_entry_text}
 
Metadata:
{kb_metadata}
 
Response:"""
 
    output = llm_local(prompt, max_tokens=300)
    return output['choices'][0]['text'].strip()
 
# Constants
API_KEY = "ufozNVMnHxpnX6j35G"  # Add your API key
EXCEL_FILE_PATH = "C:\\Users\\KirtiS\\Downloads\\Ticket_updated_responses (1).xlsx"
 
# Load Knowledge Base
def load_knowledge_base(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df
 
# Vectorize KB
def vectorize_knowledge_base(df, model):
    vector_texts = (
        df['Subject'].fillna('') + ' ' +
        df['Description'].fillna('') + ' ' +
        df['Responses'].fillna('') + ' ' +
        df['Remarks'].fillna('') + ' ' +
        df['Category'].fillna('') + ' ' +
        df['Ticket type'].fillna('')
    ).tolist()
 
    embeddings = model.encode(vector_texts, convert_to_tensor=True)
    return embeddings
 
# Extract Metadata
def extract_metadata(df):
    metadata_cols = [
        'Select Client', 'Ticket ID', 'Status', 'Priority',
        'Tag name', 'Group name', 'Created date', 'Automatable'
    ]
    return df[metadata_cols].copy()
 
# Store vectors in FAISS
def store_vectors_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
 
# UTC to IST conversion
def format_datetime(utc_str):
    try:
        utc_dt = dateparser.isoparse(utc_str)
        ist = pytz.timezone("Asia/Kolkata")
        ist_dt = utc_dt.astimezone(ist)
        return ist_dt.strftime("%B %d, %Y at %I:%M %p IST")
    except Exception:
        return utc_str
 
# Fetch tickets from Freshdesk
def fetch_todays_tickets():
    try:
        url = "https://entomo.freshdesk.com/api/v2/tickets?updated_since=2025-06-10&include=description"
        response = requests.get(url, auth=HTTPBasicAuth(API_KEY, "X"))
        response.raise_for_status()
        data = response.json()
 
        today = datetime.utcnow().date()
        filtered = [
            {
                "ID": ticket.get("id"),
                "Subject": ticket.get("subject"),
                "Type": ticket.get("type"),
                "Priority": ticket.get("priority"),
                "Updated At": format_datetime(ticket.get("updated_at")),
                "Description": ticket.get("description_text"),
                "custom_fields": ticket.get("custom_fields", {}) 
            }
            for ticket in data
            if datetime.strptime(ticket.get("updated_at", ""), "%Y-%m-%dT%H:%M:%SZ").date() == today
        ]
        return filtered
    except Exception as e:
        return f"Error: {str(e)}"
 
# KB retrieval
def retrieve_relevant_entries(query, model, index, kb_df, metadata_df):
    query_embedding = model.encode([query], convert_to_tensor=True)
    D, I = index.search(query_embedding, k=5)
    matched_kb = kb_df.iloc[I[0]].copy()
    matched_metadata = metadata_df.iloc[I[0]].copy()
    return matched_kb, matched_metadata
 
# Clean Description
def clean_description(text):
    if not text:
        return ""
    patterns = [
        r"CAUTION:.*?(?=\n|$)",
        r"Disclaimer:.*?(?=\n|$)",
        r"This e-mail.*?intended solely for.*?(?=\n|$)",
        r"This is an auto generated mail.*?(?=\n|$)",
        r"The information in this email is confidential.*?(?=\n|$)",
        r"This email and any files transmitted.*?(?=\n|$)"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()
 
# Cache function for processing ticket
@st.cache_data(show_spinner=False)
def get_ticket_summary_and_response(ticket):
    cleaned_description = clean_description(ticket['Description'])
    summary = summarize_with_gguf(cleaned_description)
    query_text = ticket['Subject'] + ' ' + cleaned_description
    matched_kb, matched_metadata = retrieve_relevant_entries(query_text, model, faiss_index, kb_df, metadata_df)
    kb_entry_text = matched_kb.to_string()
    metadata_text = matched_metadata.to_string()
    generated_response = generate_response_from_kb(ticket['Subject'], cleaned_description, kb_entry_text, metadata_text)
    return cleaned_description, summary, generated_response, metadata_text
 
# Streamlit UI starts
st.set_page_config(page_title="Data Engineering Bot", layout="centered")
st.title("🤖 Data Engineering Ticket Bot")
 
# Load KB once
model = SentenceTransformer('all-MiniLM-L6-v2')
kb_df = load_knowledge_base(EXCEL_FILE_PATH)
metadata_df = extract_metadata(kb_df)
kb_embeddings = vectorize_knowledge_base(kb_df, model)
faiss_index = store_vectors_in_faiss(kb_embeddings)

st.subheader("📦 Today's Tickets")

tickets_data = fetch_todays_tickets()

if isinstance(tickets_data, str):
  st.error(tickets_data)
  elif tickets_data:
      tickets_df = pd.DataFrame(tickets_data)
      tickets_df = json_normalize(tickets_data)

      # Extract unique filter values
      clients = tickets_df['custom_fields.select_client'].dropna().unique()
      types = tickets_df['Type'].dropna().unique()
      priorities = tickets_df['Priority'].dropna().unique()

      # Filter UI
      with st.expander("🔍 Filter Tickets", expanded=True):
          col1, col2, col3 = st.columns(3)

          with col1:
                selected_client = st.selectbox("Select Client", options=["All"] + list(clients))

          with col2:
              selected_type = st.selectbox("Select Type", options=["All"] + list(types))

          with col3:
              selected_priority = st.selectbox("Select Priority", options=["All"] + list(priorities))

          search_query = st.text_input("Search by Subject:")

      # Apply filters
      filtered_df = tickets_df.copy()

      if selected_client != "All":
          filtered_df = filtered_df[filtered_df['custom_fields.select_client'] == selected_client]

      if selected_type != "All":
          filtered_df = filtered_df[filtered_df['Type'] == selected_type]

      if selected_priority != "All":
          filtered_df = filtered_df[filtered_df['Priority'] == selected_priority]

      if search_query:
          filtered_df = filtered_df[filtered_df['Subject'].str.contains(search_query, case=False, na=False)]

      if not filtered_df.empty:
          # Define a mapping of old column names to new ones
          column_renames = {
              "ID": "Ticket ID",
              "Subject": "Ticket Subject",
              "Type": "Service Type",
              "Priority": "Urgency Level",
              "Updated At": "Last Updated",
              "Description": "Ticket Description",
              "custom_fields.select_client": "Select Client",
              "custom_fields.cf_assign_to_team_member": "Assign to Team Member",
              "custom_fields.resolution_details": "Resolution Details",
              "custom_fields.cf_ticket_is_about": "Ticket is About",
              "custom_fields.cf_client_response": "Client Response"
          }

          # Apply renaming
          renamed_df = filtered_df.rename(columns=column_renames)

          # Display the renamed DataFrame
          st.dataframe(renamed_df, use_container_width=True)

          #st.dataframe(filtered_df, use_container_width=True)

          selected_ticket_id = st.selectbox("Select Ticket ID:", filtered_df['ID'])
          selected_ticket = filtered_df[filtered_df['ID'] == selected_ticket_id].iloc[0]

          with st.spinner("Processing Ticket..."):
              cleaned_description, summary, generated_response, metadata_text = get_ticket_summary_and_response(selected_ticket)

          tab1, tab2, tab3 = st.tabs(["🎟 Ticket Info", "📝 Summary", "🤖 AI Response"])

          with tab1:    
              st.dataframe({
                  "ID": selected_ticket["ID"],
                  "Subject": selected_ticket["Subject"],
                  "Type": selected_ticket["Type"],
                  "Priority": selected_ticket["Priority"],
                  "Assign to Team Member": selected_ticket.get("custom_fields", {}).get("cf_assign_to_team_member", "Not Assigned"),
                  "Select Client": selected_ticket.get("custom_fields", {}).get("select_client"),
                  "Updated At": selected_ticket["Updated At"],
                  "Description": cleaned_description
              }, use_container_width=True)

          with st.expander("🧾 Metadata Details", expanded=False):
              if isinstance(metadata_text, dict):
                  metadata_df = pd.DataFrame(metadata_text.items(), columns=["Key", "Value"])
                  st.table(metadata_df)
              else:
                  st.markdown(f"```]\n{metadata_text}\n```")

          with tab2:
              st.write("Actual Description:", cleaned_description)
              st.write("Summary:", summary.lstrip("_ ").strip() if summary.lstrip("_ ").strip() else "No summary available.")

          with tab3:
              # Initialize session state for modified response
              if "modified_response" not in st.session_state:
                  st.session_state.modified_response = generated_response
              if "editing_response" not in st.session_state:
                  st.session_state.editing_response = False

              # Display current response
              if not st.session_state.editing_response:
                  st.write(st.session_state.modified_response)

                  col1, col2 = st.columns(2)
                  with col1:
                      if st.button("✅ Approve"):
                          st.success("Response approved.")
                  with col2:
                      if st.button("✏️ Modify"):
                          st.session_state.editing_response = True
              else:
                  # Show editable text area
                  new_response = st.text_area("Modify the AI-generated response:", value=st.session_state.modified_response, height=200)
                  if st.button("💾 Save Changes"):
                      st.session_state.modified_response = new_response
                      st.session_state.editing_response = False
                      st.success("Response updated.")
      else:
          st.warning("No tickets match the selected filters.")
  else:
      st.warning("No tickets found today.")
