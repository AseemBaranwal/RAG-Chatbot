import os
from pathlib import Path

from pypdf import PdfReader
from langchain.docstore.document import Document


def list_pdf_files(data_dir="./data"):
    paths = Path(data_dir).glob('**/*.pdf')
    for path in paths:
        yield str(path)


def load_pdf_files(data_dir="./data"):
    docs = []
    paths = list_pdf_files(data_dir)
    for path in paths:
        print(f"Loading {path}")
        doc = get_document_text(open(path, "rb"))
        docs.extend(doc)
    return docs


# Use with result of file_to_summarize = st.file_uploader("Choose a file") or a string.
# or a file like object.
def get_document_text(uploaded_file, title=None):
    docs = []
    f_name = uploaded_file.name
    if not title:
        title = os.path.basename(f_name)
    if f_name.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={
                'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text
        doc_text = uploaded_file.read().decode()
        docs.append(doc_text)

    return docs
