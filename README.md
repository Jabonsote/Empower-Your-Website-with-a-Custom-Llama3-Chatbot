# Empower-Your-Website-with-a-Custom-Llama3-Chatbot
Create your own personalized chatbot experience directly within your web page using Llama3. This Streamlit-based application allows you to interact with a chatbot trained on your specified web content, providing tailored responses to your queries.


### Key Features:

- **Customized Interaction:** Enter the URL of your web page to scrape relevant content for chatbot training.
- **Personalized Questions:** Engage with the chatbot by asking custom questions tailored to your specific interests.
- **Dynamic Responses:** Experience real-time responses generated by the Llama3 model based on the provided input.
- **Conversation History:** Keep track of your interactions with the chatbot through a convenient conversation history display.

### How to Use:

1. Enter the URL of the web page you want to interact with in the provided text input field.
2. Type your custom question or inquiry in the designated "You" text input area.
3. Click the "Send" button to initiate the chatbot interaction.
4. Explore the chatbot's responses and engage in meaningful conversations tailored to your interests.

Enhance your website's user experience and foster engaging interactions with your audience by integrating a personalized Llama3 chatbot using this intuitive Streamlit application.
## Requeriments
   Python
   ```bash
     pip install langchain langchain_community streamlit
   ```
   Conda
   ```bash
     conda install langchain langchain_community streamlit -c conda-forge
   ```

## Steps to Set Up:

1. **Install Ollama**: Download Ollama for your operating system:
   - **Linux:**
   ```bash
     curl -fsSL https://ollama.com/install.sh | sh
   ```
   - **Windows:** *(Note: Windows installation method may vary)*

2. **Fetch Your Model**: Replace "llama3" with your desired model by running the following command:
   ```bash
   ollama pull llama3
     ```
3. **Run the Application**: Execute the following command to start the Streamlit application:
   ```bash
   /bin/python3 -m streamlit run chat.py
     ```
4. **Stop the Service**: To stop the Ollama service, use the following command:
   ```bash
   sudo service ollama stop

     ```
Note:
Make sure to replace "chat.py" with the filename of your Streamlit application if it's different.
Ensure that your system meets the requirements for running Ollama and Streamlit.


# What is RAG?
RAG is a technique for augmenting LLM knowledge with additional data.

LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on. If you want to build AI applications that can reason about private data or data introduced after a model's cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

LangChain has a number of components designed to help build Q&A applications, and RAG applications more generally.

Note: Here we focus on Q&A for unstructured data. If you are interested for RAG over structured data, check out our tutorial on doing question/answering over SQL data.

## Concepts
A typical RAG application has two main components:

Indexing: a pipeline for ingesting data from a source and indexing it. This usually happens offline.

Retrieval and generation: the actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

The most common full sequence from raw data to answer looks like:

### Indexing

- Load: First we need to load our data. This is done with DocumentLoaders.

- Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and for passing it in to a model, since large chunks are harder to search over and won't fit in a model's finite context window.

- Store: We need somewhere to store and index our splits, so that they can later be searched over. This is often done using a VectorStore and Embeddings model.
index_diagram

![](https://python.langchain.com/v0.2/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png)

- Retrieval and generation
- Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
- Generate: A ChatModel / LLM produces an answer using a prompt that includes the question and the retrieved data

![](https://python.langchain.com/v0.2/assets/images/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png)
