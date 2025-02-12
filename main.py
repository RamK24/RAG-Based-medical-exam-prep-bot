from flask import Flask, request, jsonify, render_template
import torch
import random
import gc
from context import Embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel, TextStreamer
from generate import generate, process_response, to_retrieve
from models import db, Docs


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///docs.db'
# init the db
db.init_app(app)

# with app.app_context():
#     db.create_all()
    
# docs = load_dataset("MedRAG/textbooks")
# docs = docs['train']

# with app.app_context():
#     db.session.query(Docs).delete()
#     db.session.commit()
#     for idx, row in enumerate(docs):
#         entry = Docs(doc_id=idx, textbook=row['title'], content=row['content'])
#         db.session.add(entry)
#     db.session.commit()

role = """You are Qwen, created by Alibaba Cloud. You are a US Medical License exam prep assistant. \n 
        **Instructions**:

1. You will receive a **medical-related query**. Along with the query, you will be provided with **context** which contains both relevant and irrelevant passages.
2. You will generate an Answer which **must** be inferred from the context, explanation and lastly a list containing the relevant passage numbers from which you inferred from. The context is a bunch of passages each provided with a passage number. eg: passage 1: information in passage 1. Use these numbers only.
3. In certain scenarios, follow up questions will be asked to your response, in this case no context will be provided to you. You will use relevant history from previous questions to infer context and generate a response. 
    For **follow up** questions, i.e., medical questions without context, you will only provide an explanation enclosed in <exp> </exp> tags and **no likely answer or relevant passages information** should be provided. here's an example
    Input: user: Here's a question: How to treat hypertension ? here's the context: some context will be provided to you.
            system: A relevant answer is generated.
            user: Here's a question: can you explain further please?
            system: <exp>some explanantion of previous question.</exp>
3. If the query is **not medical-related**, respond with:  
   *"I do not have enough information to answer your query. I can only provide assistance with US Medical License exam preparation."* and nothing else.
4. Wrap likey answer and explanation in <ans> </ans> and <exp></exp> tags.
   Output example:  

<ans> The answer is </ans> \n
<exp> Here is some explanantion. here's some more explanation</exp> \n
<p> [1, 4, 18] </p>
"""

messages = [
    {"role": "system", "content": role}
     ]
def load_models_on_startup():
    embedding_model_id = "BMRetriever/BMRetriever-7B"
    embedding_model = AutoModel.from_pretrained(embedding_model_id, device_map="auto", load_in_4bit=True)
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
    generator_model_id = 'Qwen/Qwen2.5-7B-Instruct-1M'
    generator_model = AutoModelForCausalLM.from_pretrained(generator_model_id, device_map='auto', load_in_4bit=True)
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_id)
    NER_pipeline = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
    streamer = TextStreamer(generator_tokenizer, skip_prompt=True)

    return embedding_tokenizer, embedding_model, generator_tokenizer, generator_model, NER_pipeline, streamer


embedding_tokenizer, embedding_model, generator_tokenizer, generator_model, NER_pipeline, streamer = load_models_on_startup()
emb = Embeddings(embedding_tokenizer, embedding_model)
def create_context(doc_objects):
    context = [f'passage {doc.doc_id}: {doc.content}' for doc in doc_objects]
    context = '\n'.join(context)
    return context


def get_relevant_docs(doc_ids):
    with app.app_context():
        relevant_docs = Docs.query.filter(Docs.doc_id.in_(doc_ids)).all()
        return relevant_docs 

def get_response(query, retrieve, emb_obj, messages=messages):
    supporting_info = ''
    context_str = ''
    if retrieve:
        indices = emb_obj.get_context(query)
        indices = indices.flatten().tolist()
        doc_objects = get_relevant_docs(indices)
        context_str = create_context(doc_objects)

    messages, relevant_ids = generate(query, context_str, generator_model, generator_tokenizer, streamer, messages)
    # additional passages to reduce bias.
    if relevant_ids:
        ids = set(indices).difference(set(relevant_ids))
        relevant_ids += random.sample(list(ids), 2)
        relevant_context = get_relevant_docs(relevant_ids)
        supporting_info = '\n\n'.join([f'{row.content} <b>(from {row.textbook})</b>' for row in relevant_context[:-2]])
        relevant_context = create_context(relevant_context)
        prompt = f"here's the query: {query} \n here's the context: {relevant_context}"
        messages[-2] = {'role': 'user', 'content': prompt}
    curr_response = messages[-1]['content']
    return curr_response, supporting_info

# Serve the frontend
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle POST requests from the frontend
@app.route('/generate', methods=['POST'])
def handle_generate():
    try:
        # Get JSON data from the request
        data = request.get_json()
        user_input = data.get('message')
        retrieve = to_retrieve(user_input, NER_pipeline)
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Generate a response using your function
        print('input sending:', user_input)
        print('*'*100)
        response, supporting_info = get_response(user_input, retrieve, emb)
        print('response received:', response)
        print('*'*100)
        response = process_response(response, supporting_info)
        # Return the response as JSON
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode for development