# LangChain Orchestra를 활용한 가장 기본적인 LLM 개발

이 프로젝트는 LangChain의 대화 체인(Chain)과 Streamlit UI를 결합하여 만든 가장 기본적인 형태의 AI 챗봇입니다. 별도의 외부 문서 참조(RAG) 없이, 사용자와 주고받은 대화 맥락을 기억하며 자연스러운 일상 대화를 나눌 수 있습니다.


---

## 주요 특징

- **대화 맥락 유지 (Memory):** 
MessagesPlaceholder를 통해 이전 대화 내용을 기록하고 이를 LLM에 전달하여 흐름이 끊기지 않는 대화가 가능합니다.

- **경량화된 구조:** 
RAG 방식과 달리 벡터 데이터베이스나 문서 로딩 과정이 없어 응답 속도가 매우 빠릅니다.

- **재치 있는 페르소나:** 
시스템 프롬프트를 통해 "친절하고 재치 있는 AI 조수"라는 성격이 부여되어 있습니다.

---

---

## 🛠 기술 스택
- **Frontend: Streamlit**

- **Orchestration: LangChain**

- **LLM: OpenAI gpt-4o-mini**

- **Language: Python 3.11+**

---

---

## 📂 파일 구조 및 역할
app.py: 사용자 화면(UI)을 구성하고 세션 상태(st.session_state)를 통해 대화 기록을 관리합니다.

llm_engine.py: ChatOpenAI 모델과 프롬프트 템플릿을 결합하여 실제 답변 생성 로직을 담당합니다.

.env: OpenAI API Key 등 보안이 필요한 환경 변수를 관리합니다.

---
