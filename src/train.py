DATA_PATH = "./data"
CHROMA_PATH = "./volleybot_doc_db"
CHROMA_COLLECTION_NAME = 'volleyball'

from embed_fn import get_embedding_function


from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()



from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)



def calculate_chunk_ids(doc_pages: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for doc_page in doc_pages:
        source = doc_page.metadata.get("source")
        page = doc_page.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        doc_page.metadata["id"] = chunk_id

    return doc_pages

from langchain_chroma.vectorstores import Chroma

def add_to_chroma(chunks: list[Document]):
    chunks_ids = [chunk.metadata["id"] for chunk in chunks]
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    vector_store.reset_collection()
    vector_store.add_documents(documents=chunks, ids=chunks_ids)


def main():
    documents = load_documents()
    pages_chunk = split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(pages_chunk)
    add_to_chroma(chunks_with_ids)

if __name__ == '__main__':
    main()