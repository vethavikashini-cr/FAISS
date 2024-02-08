import streamlit as st
import sys
import csv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Load CSV data
agams_loader = CSVLoader(file_path="11_agams_english.csv")  # change this according to ur location
agams_data = agams_loader.load()

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # the character length of the chunk
    chunk_overlap=100,  # the character length of the overlap between chunks
    length_function=len,  # the length function - in this case, character length (aka the python len() fn.)
)
agams_documents = text_splitter.transform_documents(agams_data)
combined_documents = agams_documents

# Embeddings and Vector Store
store = LocalFileStore("./cache/")

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

core_embeddings_model = HuggingFaceEmbeddings(
    model_name=embed_model_id
)

embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings_model, store, namespace=embed_model_id
)

vector_store = FAISS.from_documents(combined_documents, embedder)

# Streamlit app
def main():
    st.title('Document Similarity Search')

    query = st.text_input('Enter your query:')
    if st.button('Search'):
        embedding_vector = core_embeddings_model.embed_query(query)
        docs = vector_store.similarity_search_by_vector(embedding_vector, k=4)
        results = [page.page_content for page in docs]

        st.header(f'Search Results for "{query}"')
        if results:
            for result in results:
                st.write(result)
        else:
            st.write('No results found.')

if __name__ == '__main__':
    main()
