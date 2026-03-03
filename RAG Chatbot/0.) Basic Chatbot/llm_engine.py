import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class SimpleChatManager:
    def __init__(self):
        self.model_name = "gpt-4o-mini"
        self.llm = ChatOpenAI(model=self.model_name, temperature=0.7)

    def get_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 친절하고 재치 있는 AI 조수입니다. 사용자와 일상적인 대화를 나눕니다."),
            MessagesPlaceholder(variable_name="history"), # 대화 맥락 유지용
            ("human", "{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain