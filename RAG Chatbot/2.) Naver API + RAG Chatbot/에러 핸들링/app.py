from rag_engine import RAGManager
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import streamlit.components.v1 as components
from urllib.parse import quote

# 1. 엔진 가동
@st.cache_resource
def get_rag():
    # RAGManager 클래스의 인스턴스를 생성하여 반환
    return RAGManager()

rag = get_rag()
chain = rag.get_chain()

st.set_page_config(page_title="한국 식당 리뷰 챗봇", layout="wide", page_icon="🍽️")
st.title("🍽️ 식당 리뷰 챗봇")
st.caption("네이버 지역 검색 + 블로그 리뷰를 실시간으로 검색해서 답변합니다.")

# 2. 채팅 메시지 세션 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 저장된 식당 목록이 있으면 지도로 복원
        if message["role"] == "assistant" and message.get("restaurants"):
            pass

# 3. 사용자 입력 및 답변 생성
if prompt := st.chat_input("예) 서울역 근처 맛집 추천해줘"):
    # 사용자 질문 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 히스토리 변환
    chat_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        # 엔진을 통해 답변 생성
        with st.spinner("네이버에서 식당 정보를 검색 중입니다..."):
            result = chain(prompt, chat_history=chat_history)
        
        response = result["answer"]
        restaurants = result.get("restaurants", [])
        
        st.markdown(response)

        # 대화 기록 저장
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "restaurants": restaurants,
            })
        
    st.rerun()