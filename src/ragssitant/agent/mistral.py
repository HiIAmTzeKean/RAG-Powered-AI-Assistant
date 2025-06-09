import importlib.resources
import os
from typing import Optional

import dotenv
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel

from ragssitant.db.persistant_db import VectorDB


class Mistral(BaseModel):
    llm: Optional[ChatMistralAI] = None

    def __init__(self, **data) -> None:
        super().__init__(**data)
        dotenv.load_dotenv(".env")
        if "MISTRAL_API_KEY" not in os.environ:
            raise ValueError("MISTRAL_API_KEY environment variable is not set. Please set it in your .env file.")
        self.llm = ChatMistralAI(name="mistral-small")

    def answer_question(self, query: str, vectordb: VectorDB):
        """
        Generate an answer based on retrieved research.
        """
        relevant_chunks = vectordb.search(query, top_k=3)
        context = "\n\n".join([f"From {chunk['title']}:\n{chunk['content']}" for chunk in relevant_chunks])
        with importlib.resources.open_text("ragssitant.config", "prompt_template.txt") as f:
            template_str = f.read()

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template_str,
        )
        prompt = prompt_template.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        return response.content, relevant_chunks
