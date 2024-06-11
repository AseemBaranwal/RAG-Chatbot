from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(question_input):
    if not question_input:
        return None
    elif isinstance(question_input, str):
        return question_input
    elif isinstance(question_input, dict) and 'question' in question_input:
        return question_input['question']
    elif isinstance(question_input, BaseMessage):
        return question_input.content
    else:
        raise Exception(
            "string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(model, retriever, rag_prompt=None):
    # We will use a prompt template from langchain hub.
    if not rag_prompt:
        rag_prompt = hub.pull("rlm/rag-prompt")

    # And we will use the LangChain RunnablePassthrough to add some custom processing into our chain.
    rag_chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | model
    )

    return rag_chain
