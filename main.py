from data.read_data import read_pdf_to_langchain_docs, read_web_based_data, read_complex_pdf_to_langchain_docs
from data.vectorize_data import OpenAIVectorizeData, HuggingFaceVectorizeData
from data.data_processor import LangChainDocumentProcessor
import os
from dotenv import load_dotenv
import json
import re
from langchain.schema import Document
from data.summarization import summarize_with_segment_rag
load_dotenv()


def process_aml_data(pdf_path: str, skip: int = 15, model: str = "huggingface"):
    """
    Process the Base-III data from the PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
        skip (int): The number of pages to skip at the beginning of the PDF file.

    Returns:
        list[Document]: A list of Langchain documents.
    """
    documents = read_pdf_to_langchain_docs(pdf_path, skip=skip)
    print(len(documents), flush=True)
    # # Save documents to a JSON file

    output_file = "aml_documents.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([doc.model_dump()
                  for doc in documents], f, ensure_ascii=False, indent=4)
    print(f"Documents saved to {output_file}", flush=True)
    document_chunker = LangChainDocumentProcessor(documents)
    aml_chunks = document_chunker.semantic_chunking(model="huggingface")
    # print(len(aml_chunks), flush=True)
    # Save chunks to a JSON file
    output_file = "chunks_aml.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([doc.model_dump()
                  for doc in aml_chunks], f, ensure_ascii=False, indent=4)
    # print(f"Documents saved to {output_file}", flush=True)
    if model == "huggingface":
        vectorizer = HuggingFaceVectorizeData(name="AML-HuggingFace")
    else:
        vectorizer = OpenAIVectorizeData(name="AML-OpendAI")
    vector_store = vectorizer.vectorize_data(aml_chunks)
    return vector_store


def process_base_iii(pdf_path: str, skip: int = 15, model: str = "huggingface"):
    """
    Process the Base-III data from the PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
        skip (int): The number of pages to skip at the beginning of the PDF file.

    Returns:
        list[Document]: A list of Langchain documents.
    """
    documents = read_pdf_to_langchain_docs(pdf_path, skip=skip)
    print(len(documents), flush=True)
    # # Save documents to a JSON file

    output_file = "base_iii.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([doc.model_dump()
                  for doc in documents], f, ensure_ascii=False, indent=4)
    print(f"Documents saved to {output_file}", flush=True)
    document_chunker = LangChainDocumentProcessor(documents)
    base_chunks = document_chunker.semantic_chunking(model="huggingface")
    # print(len(aml_chunks), flush=True)
    # Save chunks to a JSON file
    output_file = "chunks_base_iii.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([doc.model_dump()
                  for doc in base_chunks], f, ensure_ascii=False, indent=4)
    # Filter out chunks that are just URLs or less than 5 characters
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    base_chunks = [chunk for chunk in base_chunks if not url_pattern.match(
        chunk.page_content) and len(chunk.page_content) >= 5]
    # print(f"Documents saved to {output_file}", flush=True)
    if model == "huggingface":
        vectorizer = HuggingFaceVectorizeData(name="Base-III-HuggingFace")
    else:
        vectorizer = OpenAIVectorizeData(name="Base-III-OpendAI")
    vector_store = vectorizer.vectorize_data(base_chunks)
    return vector_store


def read_complex_document_for_summarization(pdf_path):
    """
    Read a complex document for summarization.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The text of the document.
    """
    documents = read_complex_pdf_to_langchain_docs(pdf_path)
    document_chunker = LangChainDocumentProcessor(documents)
    chunks = document_chunker.recursive_chunk(
        chunk_size=3000, min_chunk_size=800)
    vector_store = HuggingFaceVectorizeData(name="Base-III-HuggingFace")
    print(len(chunks), flush=True)
    store = vector_store.get_vector_store()
    final_summary = summarize_with_segment_rag(chunks, store)
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(final_summary)
    print("Summary saved to summary.txt", flush=True)


if __name__ == "__main__":
    # pdf_path = "datasets/aml_document.pdf"
    # vector_store = process_aml_data(pdf_path)
    # print("AML data processed")
    # pdf_path = "datasets/base_iii_data.pdf"
    # vector_store = process_base_iii(pdf_path, skip=5)
    read_complex_document_for_summarization("datasets/supervisory_report.pdf")
