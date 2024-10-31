from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec


app = FastAPI()

# API 요청 / 응답
class ChatReuquest(BaseModel):
    user_id:str
    conversation_id:str
    question:str
    
class ChatResponse(BaseModel):
    answer:str

# 채팅 체인 설정 
def setup_chat_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "너는 조선의 국왕 세조야 아주 무자비하고 딱딱한 사극말투를 사용하지"),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', "{question}")
        ]
    )
    
    # 체인
    chain = prompt | ChatOpenAI(temperature=0,model="gpt-4o")| StrOutputParser()
    
    # 채팅 히스토리 가져오는 함수
    def get_chat_history(user_id,conversation_id):
        return SQLChatMessageHistory(
            table_name=user_id,
            session_id=conversation_id,
            connection="sqlite:///sqlite.db"
        )
        
    config_field =[
        ConfigurableFieldSpec(id="user_id", annotation=str,is_shared=True) ,
        ConfigurableFieldSpec(id="conversation_id", annotation=str,is_shared=True)
    ]
        
    return RunnableWithMessageHistory(
        chain,
        get_chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        history_factory_config=config_field
    )
    
# 체인 초기화
chat_chain = setup_chat_chain()

@app.post("/chat",response_model=ChatResponse)
async def chat(request:ChatReuquest):
    try:
        config = {
            "configurable" : 
                {
            'user_id':request.user_id,
            "conversation_id":request.conversation_id
                }
            }   
        response = chat_chain.invoke({"question":request.question},config)
        
        return ChatResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.get('/chat_history/{user_id}/{conversation_id}')
async def get_history(user_id:str,conversation_id:str):
    try:
        history = SQLChatMessageHistory(
            table_name=user_id,
            session_id=conversation_id,
            connection="sqlite:///sqlite.db"
        )
        
        return {"message":[
            {"role": "user" if msg.type == 'human' else 'assistance','content': msg.content}
            for msg in history.messages
        ]}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))