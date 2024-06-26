# ScholarSync

## Project Description

ScholarSync is an app designed to help researchers update their papers based on the latest novelties in the field.

It leverages NLP methods such as vectorization, retrieval and summarization to:

- retrieve from a research papers database, a set of documents similar to an input paragraph
- summarize the retrieved documents.

An evaluation pipeline to assess the quality and accuracy of the retrieval and summarization tasks is also provided.

The repository is structured as follows:

```
├───data/                                      # storage location of documents
├───logs/                                      # app.py and evaluation_runner.py logs
├───mlruns/                                    # storage location of evaluations results (with mlflow)
├───src/
│   ├───engine/
│   │   │   prompt.py                          # prompt templates for summarization
│   │   │   retrieval.py                       # classes for retrieval task
│   │   │   summarization.py                   # classes for summarization task
│   │   │   vectorization.py                   # classes for vectorization task
│   │   └───__init__.py
│   ├───tests/                                 # unit tests
│   │   │   pytest.ini                         # pytest configuration file
│   │   │   test_01_utils.py
│   │   │   test_02_data_preprocessing.py
│   │   │   test_03_vectorization.py
│   │   │   test_04_retrieval.py
│   │   │   test_05_summarization.py
│   │   │   test_06_inference.py
│   │   │   test_07_evaluation.py
│   │   └───__init__.py
│   ├───data_preprocessing.py                  # function to download documents
│   ├───evaluation.py                          # functions to evaluate and generate an evaluation dataset
│   ├───inference.py                           # inference pipeline
│   ├───utils.py                               # preprocessing and other useful functions
│   └───__init__.py
├───.env                                       # environment variables
├───.flake8                                    # flake8 configuration file
├───.gitignore
├───app.py                                     # application runner
├───config.yaml                                # application configuration file
├───Dockerfile
├───evaluation_runner.py                       # evaluation pipeline runner
├───LICENSE
├───pyproject.toml
├───README.md
└───requirements.txt
```

The following sections describe:

- how to run the app
- how to run the evaluation pipeline
- the details of the approach
- some ideas of improvement

## Running the app

The app can be run locally in a Docker container by executing the following commands:

```
docker build -t scholar-sync .
docker run -p 8501:8501 scholar-sync
```

Once the container is running, you can access the app in a browser at the address **localhost:8501**

### Configuration

Several parameters can be configured by modifying the _config.yaml_ file to run the app at your convenience.

These parameters include :

- _n_sim_docs_: number of documents to retrieve (3 by default)
- _max_summary_tokens_: maximum number of tokens of the summaries (200 by default)
- _retriever_name_: name of the retrieval model to use (”similarity” by default, see the Retrieval section below for more details on available models)
- _summarizer_name_: name of the summarization model to use (”prompt” by default, see the Summarization section below for more details on available models)
- _bucket_name_ : name of the public repository where the documents to use are stored (”llm-technical-test-data” by default).
  NB: beware that additional changes in the code might be required if the bucket has a different structure from llm-technical-test-data
- _local_docs_path_: folder where the documents should be locally stored (defaults to “./data”)

## Running the evaluation pipeline

To run the evaluation pipeline, you will first need to create a virtual environment and install the dependencies:

```
conda create -n scholar-sync python=3.12 -y
conda activate scholar-sync
pip install -r requirements.txt
```

Once the environment is setup, the pipeline can be run with

```
python evaluation_runner.py
```

The experiment results will be stored in the _mlruns/_ folder, under the “dev” experiment by default.

The following arguments can be used to run the evaluation pipeline at your convenience:

- _--retriever-name_: name of the retrieval model to use (uses the _retriever_name_ value from the config.yaml file by default, see the Retrieval section below for more details on available models)
- _--summarizer-name_: name of the summarization model to use (uses the _summarizer_name_ value from the _config.yaml_ file by default, see the Summarization section below for more details on available models)
- _--n-sim-docs_: number of documents to retrieve (uses the _n_sim_docs_ value from the _config.yaml_ file by default)
- _--max-summary-tokens_: maximum number of tokens of the summaries (uses the _max_summary_tokens_ value from the _config.yaml_ file by default)
- _--metrics_: name(s) of the metrics to use for evaluation (defaults to "similarity,accuracy,rouge_score", see the Evaluation section below for more details on available metrics)
- _--mode_: name of the mode to run the evaluation in (defaults to "self-supervised", see the Evaluation section below for more details on available modes)
- _--size_: when running in self-supervised mode, size of the dataset to generate (defaults to 10)
- _--n-sample-sentences_: when running in self-supervised mode, number of sentences to sample from sampled documents' abstracts to generate the input paragraph (defaults to 2)
- _--dataset-path_: when running in supervised mode, path to the dataset to use for evaluation (defaults to None)
- _--experiment-name_: name of the mlflow experiment to store the evaluation results under (defaults to "dev")

## Approach

This section provides more details on the implementation of each task.

### Data Preprocessing

The data provided for this project consists in a set of documents stored in a public GCP bucket. We have access to both the raw pdf files and the extracted texts from these documents.

For time limitation reasons, I decided to focus on the extracted text data.

Thus the data preprocessing merely consists in dowloading the data from the bucket, which is done with the **dowload_documents** function from _data_preprocessing.py_.

### Vectorization

Once the documents are locally downloaded, we need to create a vector database in order to be able to query these documents when given an input paragraph.

In this case, I used langchain’s DirectoryLoader, RecursiveCharacterTextSplitter and OpenAIEmbeddings to create a Chroma vectorstore.

The **Vectorizer** class from _engine/vectorization.py_ is responsible for the creation of this database.

### Retrieval

Once the vectorstore is created, we can query it using its **similarity_search_with_relevance_score** method which allows to retrieve the chunks of documents that are most similar to an input text. The similarity is defined as the cosine similarity between the chunk’s embeddings and the input text’s embeddings. The advantage of this method is that it also returns the similarity score which can be used to evaluate the performance of the retrieval task.

One particularity of this step is that, the initial documents being stored under multiple chunks, querying directly for 3 similar documents can lead to several chunks of the same document being returned. Hence the approach chosen is to iteratively query for the most similar chunk and at each iteration, removing all chunks from the documents that have already been retrieved.

The **SimilarityRetriever** class from _engine/retrieval.py_ is responsible for this task. For configuration, this model is accessible under the name “similarity”.

### Summarization

The last task is to summarize the content of the retrieved documents.

Given that we are working with research papers, a simple approach is to consider that the abstract is already a summary of the paper. Hence we can return the abstract section as summary. This approach is implemented in the **DummySummarizer** class of _engine/summarization.py_. For configuration, this model is accessible under the name “dummy”.

However, we can also imagine that the user might want a shorter summary of the paper. In this case, the abstract may be longer than the requested summary length so we need another approach.

Note that when the _max_summary_tokens_ parameter is smaller than the abstract length, DummySummarizer will return a truncated version of the abstract.

Summarization is one of the tasks at which Large Language Models models excel, thus one could be tempted to ask a LLM to summarize each retrieved research paper. However, the paper’s length might exceed the context size of the model so a more elaborate approach such as **langchain’s MapReduce chain** is needed. This approach consists in iteratively summarizing several chunks of the initial paper (map step) and producing a final summary from these intermediate summaries (reduce step).

The **PromptSummarizer** class from _engine/summarization.py_ is responsible for this task. For configuration, this model is accessible under the name “prompt”.

### Inference

The inference is the combination of the vectorization, retrieval and summarization tasks.

The **InferencePipeline** class from _inference.py_ has been created to run all these tasks from a simple call on its predict method.

### Evaluation

The main limitation to the evaluation of retrieval and summarization is the lack of a ground truth against which to evaluate a prediction.

Assuming that a ground truth is provided, I chose the following metrics to evaluate the tasks:

- retrieval:
  - **similarity**: the similarity score returned by the retriever
  - **retrieval_accuracy**: the number of documents that are correctly retrieved over the total number of documents retrieved. <br> Ex: say that documents A, B and C should be retrieved, if the retriever provides documents A, B and X, the retrieval accuracy is 2/3. <br> The **retrieval_accuracy** function from _evaluation.py_ implements this metric. <br>Note that we could also take into account the order in which the documents are retrieved.
- summarization: **ROUGE-1 score**, that is the overlap of words between prediction and ground truth.

The evaluation over these metrics is performed via the **evaluate** function from _evaluation.py_. In the _evaluation_runner.py_ arguments these metrics are refered to as “similarity”, “accuracy” and “rouge_score” respectively.

When a ground truth is not provided, the approach chosen is to generate such a dataset by sampling the documents provided. The idea is the following:

- randomly sample _n_sim_docs_ documents and take their abstract section,
- from these abstract sections, sample _n_sample_sentences_ sentences to create an artificial input paragraph
- by iterating on the two previous steps _size_ times, we create a dataset where the input paragraph is the output of step 2, the retrieval ground truth is the documents sampled and the summarization ground truth is the abstracts of these documents.

The **generate_dataset** and **generate_mixture_input** functions from _evaluation.py_ implement this approach.

When running the evaluation pipeline, you can choose whether to:

- use your own ground truth dataset (with --mode supervised--dataset_path {path/to/your/dataset}.json arguments)
- use a randomly generated ground truth (with--mode self-supervised and abritrary values of _size_ and _n_sample_sentences_).

NB: when using your own ground truth dataset, it should be a json file containing a list of dictionaries with the following structure:

```
{
    "input_text": ...,
    "expected_output": [{
        "title": ...,
        "summary": ...
    }, ...]
}
```

## Next Steps

Finally, below is a list of potential improvements to be made:

- using the raw pdf files with langchain's pdf document loader
- experimenting with different text splitters and embeddings for vectorization
- using Maximum marginal relevance search for retrieval
- summarizing the papers' abstract rather than the whole document
- prompt engineering for summarization
- evaluating the robustness of predictions with small input variations
- caching of summaries in the app to avoid re-computing them
- adding a pdf dowload button to the app
- adding input boxes for the user to vary the number of documents retrieved and/or the length of summaries
