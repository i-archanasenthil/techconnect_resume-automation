import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import tempfile
import os
import re
import io
import pandas as pd
import numpy as np
from PIL import Image
import docx
from docx import Document

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

st.set_page_config(layout = "wide")

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

#Creating the scope of the access to the drive
#Limiting the access to read-only, no edits,deletes or modification can be done to the file

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
client_config = {
    "installed":{
        "client_id": st.secrets["google"]["client_id"],
        "project_id": st.secrets["google"]["project_id"],
        "auth_uri": st.secrets["google"]["auth_uri"],
        "token_uri": st.secrets["google"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["google"]["auth_provider_x509_cert_url"],
        "client_secret": st.secrets["google"]["client_secret"],
        "redirect_uris": [st.secrets["google"]["redirect_uris"]]
    }
}

#To avoid for reauthentication everytime caches the result
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def authenticate_user():
    """
    Initializing the OAuth fow using the client_secret.json file
    Opens the user's browser for authentication and send back the credentials
    creates a drive API client for access
    """
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0)
    return build('drive', 'v3', credentials=creds)

model = load_model()

def extract_id_from_drive_url(drive_url):
    """
    Trying to find the file ID or folder ID from the URL patterns
    """
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    st.error("Invalid Google Drive URL format,")
    return None

def list_files_in_folder(drive_service, folder_id):
    """
    Lists files inside a Google Drive Folder
    """
    query = f"'{folder_id}' in parents and trashed = false"
    response = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return response.get("files", [])

def download_file(drive_service, file_id):
    """
    This function reads the file from the Google drive and returns content in bytes
    """
    request = drive_service.files().get_media(fileId = file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extracts text from the PDF file given as bytes (in-memory)
    Converts raw bytes into a file like object
    """
    pdf_stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_docx_bytes(docx_bytes):
    docx_stream = io.BytesIO(docx_bytes)
    doc = Document(docx_stream)
    return "\n".join([para.text for para in doc.paragraphs])

def send_email_with_results(df):
    sender = st.secrets["email"]["sender"]
    password = st.secrets["email"]["password"]
    receiver = st.secrets["email"]["receiver"]

    smtp_server = st.secrets["email"]["smtp_server"]
    smtp_port = st.secrets["email"]["smtp_port"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Resume Matching Results"
    msg["From"] = sender
    msg["To"] = ",".join(receiver)

    html_table = df.to_html(index=False, escape=False)
    body = f"""
    <html>
        <body>
            <p>Hello Team,<br><br>
            Here are the results from the resume matching app:</p>
            {html_table}
            <br><p>Best Regards,<br>Tech Connect Alberta</p>
        </body>
    </html>
    """
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")


logo = Image.open("assets\logo.jfif")

col1, col2, col3 = st.columns([1,3,1])
with col1:
    st.image(logo, width=100)
with col2:
    st.title("Tech Connect Alberta Resume Sorter")

st.divider()

c1, c_divider, c2 = st.columns([5, 0.5, 5])

with c1:
    job_description = st.text_area("Paste Job Description")
    folder_link = st.text_area("Google Drive Resume Links", placeholder="https://drive.google.com/drive/folders/...")

    if st.button("Run Match"):
        if not job_description or not folder_link.strip():
            st.warning("Please provide both the job description and a valid Drive folder link.")

        else:
            st.info("Authenticating with Google...")
            drive_service = authenticate_user()
            folder_id = extract_id_from_drive_url(folder_link)

            if not folder_id:
                st.error("Could not extract folder ID from the link.")
            else:
                st.info("Fetching PDF files from the Drive Folder")
                files = list_files_in_folder(drive_service, folder_id)

                if not files:
                    st.warning("No PDF files found in the folder")
                else:
                    jd_embedding = model.encode(job_description, convert_to_tensor = True) 
                    results = []

                    for f in files:
                        try:
                            file_bytes = download_file(drive_service, f["id"])

                            if f["name"].lower().endswith(".pdf"):
                                text = extract_text_from_pdf_bytes(file_bytes)
                            elif f["name"].lower().endswith(".docx"):
                                text = extract_text_from_docx_bytes(file_bytes)
                            else:
                                st.warning(f"Unsupported file type: {f['name']}")
                                continue

                            if not text.strip():
                                st.warning(f"No text found in file : {f['name']}")
                                continue

                            resume_embedding = model.encode(text, convert_to_tensor=True)
                            score = util.cos_sim(jd_embedding, resume_embedding).item()
                            file_link = f"https://drive.google.com/file/d/{f['id']}/view?usp=sharing"

                            results.append({
                                "Result Name": f['name'], 
                                "Similarity Score": round(score, 4),
                                "Download Link": file_link
                            })

                        except Exception as e:
                            st.error(f"Error processing {f['name']}: {e}")

                    if results:

                        with c2:
                            st.subheader("Match Results")
                            df = pd.DataFrame(results)
                            df = df.sort_values(by = "Similarity Score", ascending = False)

                            df['Download Link'] = df['Download Link'].apply(lambda url: f"[Download Resume]({url})")
                            st.write(df.to_markdown(index= False), unsafe_allow_html=True)
                            st.session_state["match_results_df"] = df
    
    if st.button("Send Results as Email"):
        if "match_results_df" in st.session_state:
            send_email_with_results(st.session_state["match_results_df"])
        else:
            st.warning("Please run the match first before sending the email")
                        

    