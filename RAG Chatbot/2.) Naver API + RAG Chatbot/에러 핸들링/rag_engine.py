import os
import yaml
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 환경 변수 로드: API 및 폴더 경로
load_dotenv()

class NaverSearchClient:
    """네이버 검색 API 호출 담당 클라이언트"""

    BASE_URL = "https://openapi.naver.com/v1/search"

    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")

        # API 설정이 안되었을때
        if not self.client_id or not self.client_secret:
            raise ValueError(
                ".env 파일에 NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET 설정이 되어있는지 확인하세요"
            )

        # HTTP 요청의 헤더로 실어 보냄으로써, 네이버 서버가 허가된 사용자인지 판단할 수 있음
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }


    def search_local(self, query: str, display: int = 5) -> list[dict]:
        """네이버 지역 검색 API → 식당 기본 정보 반환"""
        params = {"query": query, "display": display, "sort": "comment"}
        response = requests.get(
            f"{self.BASE_URL}/local",
            headers=self.headers,
            params=params,
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("items", [])

    def search_blog(self, query: str, display: int = 5) -> list[dict]:
        """네이버 블로그 검색 API → 식당 리뷰 반환"""
        params = {"query": f"{query} 리뷰 맛집", "display": display, "sort": "sim"}
        response = requests.get(
            f"{self.BASE_URL}/blog",
            headers=self.headers,
            params=params,
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("items", [])
    
class RAGManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self.settings = config["rag_settings"]

        self.model_name = self.settings["model_name"]
        naver_cfg = self.settings.get("naver_search", {})
        self.local_display = naver_cfg.get("local_display", 5)
        self.blog_display = naver_cfg.get("blog_display", 5)

        self.naver = NaverSearchClient()

    def _history_search_query(self, current_question: str, chat_history: list) -> str:
        """
        대화 히스토리를 참고해서 실제로 검색할 키워드 추출
        예) "거기 메뉴 알려줘" -> 거기가 어디인지 이전 대화에서 예측
        """
        if not chat_history:
            return current_question
        
        llm = ChatOpenAI(model=self.model_name, temperature=0.1)
        history_text = "\n".join(
            f"{'사용자' if isinstance(m, HumanMessage) else '어시스턴트'}: {m.content}" for m in chat_history[-6:] # 최대 최근 6개 메시지까지 참조
        )

        history_prompt = f"""
지금까지의 대화 내용입니다: {history_text}

새로운 질문: "{current_question}"

위 대화 맥락을 보고, 새로운 질문이 어떤 식당/장소에 대한 것인지 파악해서
네이버 검색에 사용할 구체적인 검색어를 한 줄로만 출력하세요.
예) "서울 일상정원 서울역점 메뉴 가격 리뷰"
설명 없이 검색어만 출력하세요."""
        
        result = llm.invoke(history_prompt)
        return result.content.strip()
    
    def _fetch_context(self, search_query:str) -> tuple[str, dict]:
        """네이버 지역 검색 + 블로그 검색 결과를 컨텍스트 문자열로 변경 + 원본 데이터로 반환"""
        context_parts = []
        raw_data = {"local": [], "blog": []}

# 식당의 위치와 명당 -> local
# 식당의 실제 평판과 리뷰 -> blog
        try:
            local_items = self.naver.search_local(search_query, self.local_display)
            raw_data["local"] = local_items
            if local_items:
                context_parts.append("=== 네이버 지역 검색 결과 (식당 정보) ===")
                for item in local_items:
                    name = item.get("title", "").replace("<b>", "").replace("</b>", "")
                    address = item.get("roadAddress") or item.get("address", "")
                    category = item.get("category", "")
                    description = item.get("description", "")
                    context_parts.append(
                        f"- 식당명: {name}\n"
                        f"  카테고리: {category}\n"
                        f"  주소: {address}\n"
                        f"  설명: {description}"
                    )
        except Exception as e:
            context_parts.append(f"[지역 검색 오류: {e}]")
        
        try:
            blog_items = self.naver.search_blog(search_query, self.blog_display)
            raw_data["blog"] = blog_items
            if blog_items:
                context_parts.append("\n=== 네이버 블로그 리뷰 ===")
                for item in blog_items:
                    title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                    desc = item.get("description", "").replace("<b>", "").replace("</b>", "")
                    blog_name = item.get("bloggername", "")
                    context_parts.append(
                        f"- 제목: {title}\n"
                        f"  블로거: {blog_name}\n"
                        f"  내용 요약: {desc}"
                    )
        except Exception as e:
            context_parts.append(f"[블로그 검색 오류: {e}]")

        context_str = "\n".join(context_parts) if context_parts else "검색 결과가 없습니다."
        return context_str, raw_data
    
    def get_chain(self):
        """대화 히스토리 반영 RAG 체인 반환"""
        llm = ChatOpenAI(model=self.model_name, temperature=0.3)

        # MessagesPlaceholder로 대화 히스토리를 프롬프트 중간에 삽입
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
당신은 식당 리뷰 전문 AI 어시스턴트입니다.
아래 네이버 검색을 통해 실시간으로 수집된 식당 정보와 블로그 리뷰,
그리고 지금까지의 대화 내용을 함께 참고해서 답변해 주세요.

식당에 대한 정보를 물었을때 아래와 같은 답변 형식을 따라주세요:

1. 추천 식당이 있다면 식당명과 주소를 명확히 언급
2. 리뷰에서 언급된 메뉴, 분위기, 가격대 등을 구체적으로 설명
3. 정보가 부족한 경우 솔직하게 안내

다만 정보가 부족한 경우 솔직하게 안내

#수집된 정보 (네이버 실시간 검색):
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"), # 대화 히스토리 설정
            ("human", "{question}"),
        ])

        # 체인 생성
        chain = prompt | llm | StrOutputParser()

        def rag_chain_with_history(question: str, chat_history: list = []) -> dict:
            # 1. 히스토리를 보고 실제 검색어 결정
            search_query = self._history_search_query(question, chat_history)

            # 2. 네이버 API로 컨텍스트 수집
            context, raw_data = self._fetch_context(search_query)

            # 3. LLM 호출
            answer = chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": question,
            })

            # 4. 지역 검사 결과에서 식당 이름+주소 추출
            restaurants = []
            for item in raw_data.get("local", []):
                name = item.get("title", "").replace("<b>", "").replace("</b>", "")
                address = item.get("roadAddress") or item.get("address", "")
                if name:
                    restaurants.append({"name": name, "address": address})
            
            return {
                "answer": answer,
                "restaurants": restaurants,
            }
        
        return rag_chain_with_history