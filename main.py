import getpass
import os

from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

from ragssitant.chuncker.text_chuncker import TextChunker
from ragssitant.db.persistant_db import VectorDB


def answer_research_question(query: str, vectordb: VectorDB, llm):
    """
    Generate an answer based on retrieved research.
    """
    relevant_chunks = vectordb.search(query, top_k=3)
    context = "\n\n".join([f"From {chunk['title']}:\n{chunk['content']}" for chunk in relevant_chunks])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following research findings, answer the researcher's question:

Research Context:
{context}

Researcher's Question: {question}

Answer: Provide a comprehensive answer based on the research findings above.
""",
    )
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, relevant_chunks


def main() -> None:
    documents_path = "./data"
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    publications = chunker.load_documents(documents_path)
    chunked_publications = chunker.process_documents(publications)
    print(f"\nTotal chunked documents: {len(chunked_publications)}")

    vectordb = VectorDB(db_path="./research_db", collection_name="ml_publications")
    vectordb.insert_documents(chunked_publications)
    print("Documents inserted into vector database.")

    if "MISTRAL_API_KEY" not in os.environ:
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

    llm = ChatMistralAI(name="mistral-small")
    answer, sources = answer_research_question(
        "What are effective techniques for handling class imbalance?", vectordb, llm
    )
    print("AI Answer:", answer)
    print("\nBased on sources:")
    for source in sources:
        print(f"- {source['title']}")


if __name__ == "__main__":
    main()
