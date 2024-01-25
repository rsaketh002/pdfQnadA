# Import necessary libraries
import streamlit as st
import os

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index import ServiceContext, set_global_service_context, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM

# Install required packages (you might not need these as Streamlit may have them pre-installed)
# !pip install -q cassandra-driver
# !pip install -q cassio>=0.1.1
# !pip install -q gradientai --upgrade
# !pip install -q llama-index
# !pip install -q pypdf
# !pip install -q tiktoken==0.4.0

# Set environment variables
GRADIENT_ACCESS_TOKEN = st.secrets["GRADIENT_ACCESS_TOKEN"]
GRADIENT_WORKSPACE_ID = st.secrets["GRADIENT_WORKSPACE_ID"]


# Load Streamlit app
def main():
    st.title("Streamlit App")

    # Load documents
    documents = SimpleDirectoryReader("docs").load_data()
    st.write(f"Loaded {len(documents)} document(s).")

    # Initialize GradientBaseModelLLM and GradientEmbedding
    llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
    embed_model = GradientEmbedding(
        gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
        gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
        gradient_model_slug="bge-large",
    )

    # Initialize ServiceContext
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=256)
    set_global_service_context(service_context)

    # Initialize VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    # Get user input
    user_query = st.text_input("Enter your query:")

    # Perform query when button is clicked
    if st.button("Submit"):
        response = query_engine.query(user_query)
        st.write("Response:", response.response)

if __name__ == "__main__":
    main()
