from langchain_openai import ChatOpenAI

def get_openai_llm(model_name: str = "gpt-3.5-turbo",
                   max_tokens = 512,
                   temp = 0.3):
    llm = ChatOpenAI(api_key="",
                     model_name=model_name,
                     max_tokens=max_tokens,
                     temperature=temp,
                     max_retries = 2)
    return llm