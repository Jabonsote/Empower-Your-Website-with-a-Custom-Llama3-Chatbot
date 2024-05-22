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
