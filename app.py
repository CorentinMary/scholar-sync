import datetime as dt
import logging

import streamlit as st
import yaml
from dotenv import load_dotenv

from src.data_preprocessing import download_documents
from src.inference import InferencePipeline

# Logging configuration
datetime_str = dt.datetime.now().strftime(format="%Y-%m-%d_%H%M")
log_file_name = f"./logs/app_{datetime_str}.log"
logging.basicConfig(
    filename=log_file_name,
    level=logging.INFO,
)


@st.cache_resource
def initialize_app():
    """Initializes the app in a cached way so that each step is executed only once when launching the app."""
    # Loading environment variables and configuration
    logging.info("Loading the environment variables and configuration")
    load_dotenv()
    config = yaml.safe_load(open("./config.yaml", "r"))

    # Download the documents from the bucket and initialize retriever and summarizer objects
    logging.info("Downloading the documents")
    download_documents(
        bucket_name=config["bucket_name"],
        destination=config["local_docs_path"],
    )
    logging.info("Initializing the inference pipeline")
    pipeline = InferencePipeline(
        document_folder=config["local_docs_path"],
        retriever_name=config["retriever_name"],
        summarizer_name=config["summarizer_name"],
        summarizer_kwargs={"max_summary_tokens": config["max_summary_tokens"]},
    )
    return config, pipeline


config, pipeline = initialize_app()

# Application layout
st.markdown("<h1 style='text-align: center;'>ScholarSync</h1>", unsafe_allow_html=True)
st.write(
    """
    **ScholarSync is designed to help you update your paper based on the latest novelties in the field.**
    \n**To get started, fill the field below with an input paragraph and click on the button.**
    """
)
input = st.text_input(label="Input paragraph")

if input:
    st.write(
        f"""
        **Your input paragraph:**
        \n{input}
        """
    )

if st.button("Start Engine", type="primary"):
    with st.spinner("Retrieving similar papers and summarizing..."):
        logging.info(f"Starting inference with input = '{input}'")
        output = pipeline.predict(input_text=input, n_sim_docs=config["n_sim_docs"])
        logging.info("Inference complete")
    st.write("**Helpful papers:**")
    for doc in output:
        st.header(doc["title"])
        st.write(doc["summary"])
