from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


def summarize_documents(documents: list[Document], context: str = "") -> str:
    # Initialize the Groq LLM with Llama-3.2
    llm = ChatGroq(
        # Llama 3.2 specific model (adjust if needed)
        model="llama-3.2-90b-vision-preview",
    )
    base_template_str = f'Summarize the following text concisely in the context of {context}:\n'
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=base_template_str + "{text}",
    )

    # Create the LLM chain
    summarization_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Summarize each document
    summaries = []
    for doc in documents:
        summary = summarization_chain.run(text=doc.page_content)
        summaries.append(summary)

    # Merge the summaries into a single coherent summary
    merged_summary = " ".join(summaries)

    final_summary = summarization_chain.run(text=merged_summary)

    return final_summary


def summarize_with_segment_rag(
    segments: list[Document],
    vector_store: any,
    context: str = "",
    top_k: int = 2
) -> str:
    """
    Summarize segments with relevant RAG-enriched data from a vector store.

    Args:
        segments (list[Document]): Segments (chunks) to summarize.
        vector_store (VectorStore): The populated vector store for RAG.
        context (str): Additional context to guide the summary.
        top_k (int): Number of relevant chunks to retrieve for each segment.

    Returns:
        str: Final summary combining all enriched segment summaries.
    """
    # 1. Initialize the ChatGroq model
    llm = ChatGroq(model="llama-3.2-90b-vision-preview")

    # 2. Define the summarization prompt template
    base_template_str = (
        "You are a domain-specific assistant specializing in summarizing financial and compliance documents, "
        "particularly Basel III regulations and Anti-Money Laundering (AML) frameworks.\n\n"
        "### Instructions:\n"
        "1. Summarize the provided document and retrieved context concisely.\n"
        "2. Focus on the most critical points relevant to the topic.\n"
        "3. Prioritize key regulations, frameworks, rules, and thresholds.\n"
        "4. Remove redundant or irrelevant details.\n"
        "5. Use clear and professional language suitable for financial reports.\n"
        "6. If applicable, organize the summary into structured bullet points.\n\n"
        "### Context:\n{context}\n\n"
        "### Input Document:\n{text}\n\n"
        "### Output Format:\n"
        "- A concise summary of key points.\n"
        "- Highlight important thresholds, rules, or frameworks.\n"
        "- Avoid duplication of ideas."
    )
    prompt_template = PromptTemplate(
        input_variables=["context", "text"], template=base_template_str)

    # 3. Prepare the LLM chain
    summarization_chain = prompt_template | llm

    # 4. Process each segment: Retrieve relevant chunks and summarize
    summaries = []
    print(segments[:3])
    for segment in segments:
        # print(f"Processing segment: {segment.metadata}", flush=True)
# 
        # Retrieve relevant chunks from the vector store
        relevant_docs = vector_store.similarity_search(
            segment.page_content, k=top_k)
        retrieved_text = "\n\n".join(
            [doc.page_content for doc in relevant_docs])

        # Combine the segment with retrieved context
        combined_text = f"{segment.page_content}\n\n--- Retrieved Context ---\n\n{retrieved_text}"

        # Summarize the combined text
        summary = summarization_chain.invoke({"context": context, "text": combined_text})
        summaries.append(summary)

    # 5. Combine all summaries into a final summary
    final_summary = "\n".join(summaries)

    return final_summary
