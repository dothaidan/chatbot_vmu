import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

template = """Bạn là một người quản lý công tác sinh viên của trường đại học, 
hãy sử dụng kiến thức đã có sẵn để  trả lời câu hỏi của sinh viên.
Nếu bạn không biết hãy tìm địa chỉ văn phòng công tác sinh viên và chỉ dẫn sinh viên đến đó.
Hãy viết đầy đủ nội dung.
Context: {context}
"""

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("system", "Context: {context}"),
    ("human", "Question: {question}")
])


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer:\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = custom_prompt
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }
        rag_chain = (
                input_data
                | self.prompt
                | self.llm
                | self.str_parser
        )
        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
