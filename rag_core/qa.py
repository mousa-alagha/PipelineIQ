
from __future__ import annotations

from typing import List, Tuple
import re

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, Document


def load_index(store_dir: str = "vectorstore") -> FAISS:
    """Load the FAISS index from disk (trusting your own pickle)."""
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        store_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


from langchain.chains import ConversationalRetrievalChain

def load_conv_chain(store_dir: str = "vectorstore", temperature: float = 0.0):
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    index = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(temperature=temperature)
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=index.as_retriever(),
        return_source_documents=True,   
    )


import re

def answer_and_summarize(question: str, index: FAISS, k: int = 3, temperature: float = 0.0):
    llm = ChatOpenAI(temperature=temperature)
    qa = load_qa_with_sources_chain(llm, chain_type="stuff")

    docs = index.similarity_search(question, k=k)
    result = qa({"input_documents": docs, "question": question})
    answer = result["output_text"]

    
    answer = re.sub(r"\n?SOURCES:.*$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()

    summary_msg = llm([HumanMessage(content=f"Summarize this answer in 3 bullet points:\n\n{answer}")])
    return answer, summary_msg.content, docs
