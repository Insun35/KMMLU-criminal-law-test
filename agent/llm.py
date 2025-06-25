import os
from dotenv import load_dotenv
from openai import OpenAI
from .prompt_tpl import PROMPT_TPL
from .retriever import Retriever

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Agent:
    def __init__(self, api_key: str, k: int = 3):
        self.llm = OpenAI(api_key=api_key)
        self.retriever = Retriever(api_key=api_key, k=k)

    def answer(self, question: str, choices: dict[str, str]) -> str:
        context_chunks = self.retriever.retrieve(question)
        context = "\n\n".join(context_chunks)
        prompt = PROMPT_TPL.format(
            question=question,
            A=choices["A"],
            B=choices["B"],
            C=choices["C"],
            D=choices["D"],
            context=context,
        )
        resp = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
        )
        return resp.choices[0].message.content.strip().upper()
