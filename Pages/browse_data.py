import streamlit as st
import os
from local_loader import list_pdf_files
import PyPDF2


def get_document_text(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def main():
    paths = list(list_pdf_files())
    file_path = st.selectbox("Select a data file to view", paths, index=None)

    if file_path:
        if is_pdf(file_path):
            pdf_text = get_document_text(file_path)
            st.text(pdf_text)
        else:
            with open(file_path, "r") as f:
                md_content = f.read()
                st.markdown(md_content)


def is_pdf(file_path):
    return file_path.endswith('.pdf') and os.path.isfile(file_path)


if __name__ == "__main__":
    main()
