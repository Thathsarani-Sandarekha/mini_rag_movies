import os
import json
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def chunk_docs(df: pd.DataFrame,
               chunk_chars: int = 1500,
               overlap_chars: int = 150) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = [
        Document(page_content=row.Plot, metadata={"title": row.Title})
        for _, row in df.iterrows()
    ]
    return splitter.split_documents(docs)


def build_vectorstore(chunks: list[Document],
                      embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Chroma:
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    client = chromadb.EphemeralClient()
    vstore = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        collection_name="movie_plots",
        client=client,
    )
    return vstore


def make_llm(model: str = "openai/gpt-oss-20b", temperature: float = 0.0) -> ChatGroq:
    if not api_key:
        raise EnvironmentError("Please set GROQ_API_KEY in your environment.")
    return ChatGroq(model=model, temperature=temperature)


def format_context_snippets(docs: list[Document], max_chars: int = 400) -> list[str]:
    """
    Keep snippets compact; prefix with title for traceability.
    """
    snippets = []
    for d in docs:
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        title = d.metadata.get("title", "Unknown")
        snippets.append(f"{title} — {text}")
    return snippets


def ask_llm_for_answer_and_reasoning(llm, question, context_snippets):
    system_msg = (
        "You are a helpful assistant that answers questions about movie plots in natural, full-sentence form. "
        "Base your answer ONLY on the provided context snippets. "
        "If the answer is not found, respond with a JSON object stating so. "
        "Return output strictly as valid JSON with fields: "
        '{"answer": "<natural language answer>", "reasoning": "<short explanation>"}'
    )

    user_msg = (
        f"Question:\n{question}\n\nContext snippets:\n"
        + "\n".join(f"- {s}" for s in context_snippets)
        + "\n\nReturn a valid JSON object with fields "
        '{"answer": "...", "reasoning": "..."} only.'
    )

    resp = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg)
    ])

    text = resp.content if hasattr(resp, "content") else str(resp)
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or "answer" not in data or "reasoning" not in data:
            raise ValueError("JSON missing required keys.")
        return {"answer": data["answer"], "reasoning": data["reasoning"]}
    except Exception:
        return {
            "answer": text.strip(),
            "reasoning": "Used top retrieved context snippets to form the answer."
        }

def run_pipeline(csv_path, question,
                 top_k=4,
                 chunk_chars=1500,
                 overlap_chars=150,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 groq_model="openai/gpt-oss-20b",
                 temperature=0.0):

    # 1. Load dataset
    df = load_df(csv_path)

    # 2. Chunk plots
    chunks = chunk_docs(df, chunk_chars, overlap_chars)

    # 3. Build vector store (Chroma)
    vstore = build_vectorstore(chunks, embedding_model)
    retriever = vstore.as_retriever(search_kwargs={"k": top_k})
    retrieved = retriever.invoke(question)

    # 4. Prepare snippets
    context_snippets = format_context_snippets(retrieved, max_chars=400)

    # 5. Query LLM (Groq)
    llm = make_llm(model=groq_model, temperature=temperature)
    llm_out = ask_llm_for_answer_and_reasoning(llm, question, context_snippets)

    # 6. Output structured JSON
    result = {
        "answer": llm_out["answer"],
        "contexts": context_snippets,
        "reasoning": llm_out["reasoning"]
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    csv_path = "data/movies_data_subset_clean.csv"
    print("Type your question below (or type 'exit' to quit)\n")

    while True:
        question = input("Enter your question: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Exiting... 👋")
            break

        if question:
            run_pipeline(csv_path, question)
            print("\n---\n")
