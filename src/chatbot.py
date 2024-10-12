from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from embed_fn import get_embedding_function

CHROMA_PATH = "./volleybot_doc_db"
CHROMA_COLLECTION_NAME = 'volleyball'

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    query_text = "what is volleyball"
    resp = query_rag(query_text)
    print(resp)


def query_rag(query_text):

    db = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)


    model = OllamaLLM(model="nemotron-mini")
    response_text = model.invoke(prompt)

    # debug
    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    
    return response_text


if __name__ == '__main__':
    main()