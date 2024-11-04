from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class PDFRAG:
    def __init__(self, file_path: str, llm):
        self.file_path = file_path
        self.llm = llm

    # 문서 로드
    def load_documents(self):
        loader = PyMuPDFLoader(self.file_path)
        docs = loader.load()
        return docs
    
    # 문서 분할
    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)
        return split_documents
    
    # 임베딩
    def create_vectorstore(self, split_documents):
        embeddings = OpenAIEmbeddings()

        # DB 새성
        vectorstore = FAISS.from_documents(
            documents= split_documents, 
            embedding=embeddings
        )

        return vectorstore
    
    # 검색기생성
    def create_retriever(self):
        vectorstore = self.create_vectorstore(
            self.split_documents(self.load_documents())
        )

        retriever = vectorstore.as_retriever()
        return retriever
    
    def create_chain(self, retriever):
        # 프롬프트
        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
        )

        # 체인생성
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain