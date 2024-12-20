from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def get_openai_llm(model_name: str = "gpt-4o-mini",
                   max_tokens=512,
                   temp=0.3):
    llm = ChatOpenAI(api_key=api_key,
                     model_name=model_name,
                     max_tokens=max_tokens,
                     temperature=temp,
                     max_retries=2)
    return llm
