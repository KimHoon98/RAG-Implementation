import streamlit as st
from llm_engine import SimpleChatManager
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="데일리 챗봇", page_icon="💬")
st.title("💬 일상 대화용 기본 챗봇")

# 엔진 초기화
@st.cache_resource
def get_chat_manager():
    return SimpleChatManager()

chat_manager = get_chat_manager()
chain = chat_manager.get_chain()

# 세션 상태 초기화 (대화 기록 저장)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 1. 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 어시스턴트 답변 생성
    with st.chat_message("assistant"):
        # 대화 맥락(history) 생성
        history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))

        # RAG 없이 LLM 답변 호출
        response = chain.invoke({
            "question": prompt,
            "history": history
        })
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})