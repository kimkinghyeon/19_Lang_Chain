from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec

# FastAPI 애플리케이션 생성
app = FastAPI()

# 요청 모델: 사용자의 대화 요청을 위한 데이터 모델
class ChatRequest(BaseModel):
    user_id: str           # 사용자를 구별하기 위한 ID
    conversation_id: str    # 대화를 구별하기 위한 ID
    question: str           # 사용자가 질문하는 내용

# 응답 모델: AI의 응답 데이터를 포함하는 모델
class ChatResponse(BaseModel):
    answer: str             # AI가 생성한 응답

# 채팅 체인을 설정하는 함수 정의
def setup_chat_chain():
    # 세조의 말투를 사용하는 대화 프롬프트 설정
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "너는 조선의 국왕 세조야 아주 무자비하고 딱딱한 사극말투를 사용하지"),
            MessagesPlaceholder(variable_name='chat_history'),   # 대화 기록을 여기에 추가
            ('human', "{question}")                              # 사용자의 질문을 {question}에 할당
        ]
    )

    # ChatOpenAI 체인 구성
    chain = prompt | ChatOpenAI(temperature=0, model="gpt-4o") | StrOutputParser()

    # 대화 기록을 데이터베이스에서 조회하는 함수
    def get_chat_history(user_id, conversation_id):
        return SQLChatMessageHistory(
            table_name=user_id,                      # user_id를 테이블 이름으로 사용해 사용자별 기록 저장
            session_id=conversation_id,              # session_id로 대화 구분
            connection="sqlite:///sqlite.db"         # SQLite 데이터베이스 연결 문자열
        )

    # 구성 가능한 필드 정의: user_id와 conversation_id를 공유 가능한 값으로 설정
    config_field = [
        ConfigurableFieldSpec(id="user_id", annotation=str, is_shared=True),
        ConfigurableFieldSpec(id="conversation_id", annotation=str, is_shared=True)
    ]

    # 대화 체인 반환
    return RunnableWithMessageHistory(
        chain,                                      # 설정된 ChatOpenAI 체인
        get_chat_history,                           # 대화 기록 조회 함수
        input_messages_key="question",              # 사용자의 질문을 받는 키
        history_messages_key="chat_history",        # 대화 기록을 저장할 키
        history_factory_config=config_field         # 대화 기록 조회 시 참고할 파라미터 설정
    )

# 채팅 체인 초기화: AI의 응답 생성을 위한 체인을 미리 설정
chat_chain = setup_chat_chain()

# 채팅 응답 엔드포인트: POST 요청으로 사용자의 질문에 대한 AI 응답을 반환
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # config 설정: 사용자의 ID와 대화 ID를 사용해 대화 기록을 조회
        config = {
            "configurable": {
                'user_id': request.user_id,
                "conversation_id": request.conversation_id
            }
        }
        # ChatOpenAI 체인을 통해 AI 응답 생성
        response = chat_chain.invoke({"question": request.question}, config)
        
        # AI의 응답 반환
        return ChatResponse(answer=response)
    except Exception as e:
        # 예외 발생 시 500 오류 반환
        raise HTTPException(status_code=500, detail=str(e))

# 대화 기록 조회 엔드포인트: GET 요청으로 특정 사용자와 대화 ID에 따른 대화 기록을 반환
@app.get('/chat_history/{user_id}/{conversation_id}')
async def get_history(user_id: str, conversation_id: str):
    try:
        # 사용자의 대화 기록 조회
        history = SQLChatMessageHistory(
            table_name=user_id,                      # 사용자의 ID로 테이블 이름 설정
            session_id=conversation_id,              # 대화 ID로 특정 대화 구분
            connection="sqlite:///sqlite.db"         # SQLite 데이터베이스 연결
        )

        # 대화 기록을 JSON 형식으로 반환
        return {
            "message": [
                {
                    "role": "user" if msg.type == 'human' else 'assistant',  # 사용자인지 AI인지 구분
                    'content': msg.content                                   # 메시지 내용
                }
                for msg in history.messages                                  # 대화 기록의 각 메시지 조회
            ]
        }
    except Exception as e:
        # 예외 발생 시 500 오류 반환
        raise HTTPException(status_code=500, detail=str(e))
