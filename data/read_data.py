from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.schema import Document
import pdfplumber


def read_pdf_to_langchain_docs(pdf_path: str, skip: int = 0) -> list[Document]:
    """
    Reads a PDF file and returns Langchain documents.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list[Document]: A list of Langchain documents.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents[skip:]


def read_web_based_data(url: str) -> list[Document]:
    """
    Reads a web based data and returns Langchain documents.

    Args:
        url (str): The url to the web based data.

    Returns:
        list[Document]: A list of Langchain documents.
    """
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents


def read_complex_pdf_to_langchain_docs(pdf_path: str,skip:int=5) -> list[Document]:
    """
    Reads a PDF file and returns Langchain documents.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list[Document]: A list of Langchain documents.
    """
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        
        for page in pdf.pages[5:]:
            extracted_data.append({
                "text": page.extract_text()
            })

            tables = page.extract_tables()
            # print(tables)
            if tables:
                for table in tables:
                    table_text = "\n".join([", ".join(
                        [str(cell) if cell is not None else "" for cell in row]) for row in table])
                    extracted_data.append({
                        "text": table_text
                    })
    documents = [Document(page_content=doc["text"]) for doc in extracted_data]
    return documents
