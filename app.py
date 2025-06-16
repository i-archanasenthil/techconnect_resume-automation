import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import tempfile
import os
import re
import io

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

#Creating the scope of the access to the drive
#Limiting the access to read-only, no edits,deletes or modification can be done to the file

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CLIENTS_SECRET_FILE = st.secrets["google"]["client_secret_path"]
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
    flow = InstalledAppFlow.from_client_secrets_file(CLIENTS_SECRET_FILE, SCOPES)
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

def list_files_in_folder(drive_service, folder_id, mime_filter="application/pdf"):
    """
    Lists files inside a Google Drive Folder
    """
    query = f"'{folder_id}' in parents and trashed = false"
    if mime_filter:
        query += f" and mimeType = '{mime_filter}'"
    response = drive_service.files().list(q=query, fields="files(id, name)").execute()
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

st.title("Resume Matching Application")

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
                        text = extract_text_from_pdf_bytes(file_bytes)

                        if not text.strip():
                            st.warning(f"No text found in file : {f['name']}")
                            continue

                        resume_embedding = model.encode(text, convert_to_tensor=True)
                        score = util.cos_sim(jd_embedding, resume_embedding).item()
                        results.append((f['name'], score))

                    except Exception as e:
                        st.error(f"Error processing {f['name']}: {e}")

                st.subheader("Match Results")
                for name, score in sorted(results, key = lambda x: x[1], reverse = True):
                    st.markdown(f" - **{name}** - **Similarity Score**: `{score:.4f}`")
                
