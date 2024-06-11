from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from memory import create_memory_chain
from rag_chain import make_rag_chain


def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    system_prompt = """You are a helpful AI assistant that is an auto insurance expert. You are provided with prior 
    information on the policy documents and you have to use the following context and the users' chat history to help 
    the user. Do not hesitate to ask for clarification if you need more information. Also, please format the response 
    in a proper manner with headings, bullet points, numbered lists, and tables as needed. If you don't know the 
    answer, just say that you don't know.
    
    Context: {context}
    
    Question: """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    return response
