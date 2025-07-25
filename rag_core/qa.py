from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage


def load_index(store_dir: str = "vectorstore"):
    """Load the FAISS index from disk (trusting your own pickle)."""
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        store_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )


def load_conv_chain(
    store_dir: str = "vectorstore",
    temperature: float = 0.0
) -> ConversationalRetrievalChain:
    """
    Load a conversational retrieval chain that carries chat history forward,
    so follow-up pronouns are resolved correctly.
    """
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    index = FAISS.load_local(
        store_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    llm = ChatOpenAI(temperature=temperature)
    conv = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=index.as_retriever(),
        return_source_documents=False,
    )
    return conv


def answer_and_summarize(
    question: str,
    index: FAISS,
    k: int = 3,
    temperature: float = 0.0,
):
    """Run a QA chain on the top-k docs and then summarize the answer."""
    # First, do standard QA with sources
    llm = ChatOpenAI(temperature=temperature)
    qa = load_qa_with_sources_chain(llm, chain_type="stuff")

    docs = index.similarity_search(question, k=k)
    result = qa({"input_documents": docs, "question": question})
    answer = result["output_text"]

    # Then create a 3-bullet summary
    summary_msg = llm([
        HumanMessage(content=f"Summarize this answer in 3 bullet points:\n\n{answer}")
    ])
    return answer, summary_msg.content