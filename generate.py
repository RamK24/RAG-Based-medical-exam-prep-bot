
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import re
import ast
import gc
from flask import Response

# docs = load_dataset("MedRAG/textbooks")
# query = """ 
# . An 11-year-old boy is brought to the office by his parents for a well-child examination and routine vaccinations. His medical
# history is unremarkable and he receives no medications. Vaccinations were up-to-date at his most recent examination 1 year
# ago. He is not sexually active, and the family is affiliated with a religion that teaches abstinence. The parents express
# reluctance about administering the human papillomavirus (HPV) vaccine today because of the potential adverse effects
# associated with the vaccine. They ask if their son really needs this vaccine today, given that he will not become sexually
# active until he is much older and married. The patient’s vital signs are within normal limits. Physical examination shows no
# abnormalities. Which of the following is the most appropriate response to the parents?
# """

# context = emb.get_context(query, docs['train'])
# del emb.model
# emb.model = None
# del emb

# gc.collect()
# torch.cuda.empty_cache()




def create_prompt(query, context):
    return f"here's the query: {query} \n here's the context: {context}"

def extract_lists(response):
    pattern = r'\[[^\[\]]+\]'
    # Find the first match
    match = re.search(pattern, response)
    if match:
        return ast.literal_eval(match.group(0))
    
def to_retrieve(query,  pipe=None):
    # pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
    result = pipe(query)
    # del pipe
    # torch.cuda.empty_cache()
    # gc.collect()
    return True if result else False

def generate(query, context_str, model, tokenizer, history=[]):
    if context_str:
        prompt = create_prompt(query, context_str)
    else:
        prompt = query
    history.append({'role': 'user', 'content': prompt})
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    generated_ids = model.generate(**inputs, temperature=0.1, max_new_tokens=1500)
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    relevant_context_ids = extract_lists(response)
    print('relevance:', relevant_context_ids)
    history.append({'role': 'system', 'content': response})

    # return Response(generate_tokens(), content_type="text/plain"), history, relevant_context_ids
    
    return history, relevant_context_ids


def process_response(response, supporting_info):
    offset = len('<ans>')
    answer_start = response.find('<ans>') 
    answer_end = response.find('</ans>')
    explanantion_start = response.find('<exp>') 
    explanantion_end = response.find('</exp')
    # print('extracted', [answer_start, answer_end, explanantion_end, explanantion_start])

    if -1 not in [explanantion_start, explanantion_end, answer_start, answer_end]:
        return (f'<b>Likely answer</b>: <b>{response[answer_start + offset: answer_end].strip()}</b> \n\n'
                
                f'<b>Explanation </b>: {response[explanantion_start + offset: explanantion_end].strip()} \n\n'

                f'<b>Supporting information</b>: {supporting_info}')
    else:
        if explanantion_start != -1 and explanantion_end != -1:
            return response[explanantion_start + offset: explanantion_end].strip()
        else:
            return response