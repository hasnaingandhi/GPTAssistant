from flask import Flask, render_template, request
import random
import os
import datetime
from werkzeug.datastructures import MultiDict
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
os.environ["OPENAI_API_KEY"] = "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"
#openai.api_key = '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'
conversation = []

@app.route('/')
def home():
    return render_template('index.html',messages=conversation)

@app.route('/submit', methods=['POST'])
def submit():
    input_text = request.form.get('input-field')
    current_time = datetime.datetime.now().strftime("%I:%M %p")

    #form_data = MultiDict(request.form)
    user_message = {
        'type': 'user-message',
        'text': input_text,
        'sentTime': current_time
    }
    conversation.append(user_message)
    #form_data['input-field'] = ''
    response = chatbot_response(input_text)
    chatbot_message = {
        'type': 'chatbot',
        'text': response,
        'sentTime': current_time
    }
    conversation.append(chatbot_message)
    #return render_template('index.html', input_text=input_text, response=response, current_time=current_time)
    return render_template('index.html', messages=conversation)
    

def chatbot_response(user_input):
    answer = ""
    
    # doc = textract.process("./SyllabusChat.pdf")

    # with open('SyllabusChat.txt', 'w') as f:
    #     f.write(doc.decode('utf-8'))

    with open('SyllabusChat_5Jul.txt', 'r') as f:
        text = f.read()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )

    #chunks = text_splitter.create_documents([textL])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(text_splitter.create_documents([text]),embeddings)
    chain = load_qa_chain(OpenAI(temperature=0,model_name='text-davinci-003'), chain_type="stuff")
    query = user_input
    '''
    prompt = f"""
        Your task is to find the answer of the query by using the given context.
        Steps to follow: 
        1. Use the given file to answer and ignore any hallucination as well as outside data except the given context.

        Information is given in the following format 
        Question:
        ```
        question here
        ```

        answer here
        ```

        Question:
        ```
        {query}

        ```

        Answer
    """
    '''
    prompt = query
    
    docs = db.similarity_search(query)

    answer = chain.run(input_documents=docs, question=query)
    print(query)
    return answer


if __name__ == '__main__':
    app.run()