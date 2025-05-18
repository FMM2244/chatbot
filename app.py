# from flask import Flask, request, jsonify, render_template
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# from huggingface_hub import whoami
# from prompt_utils import create_prompt, detect_domain

# app = Flask(__name__)

# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.get_json()
#     user_question = data.get('question', '')

#     # Automatically choose relevant domains
#     selected_domains = detect_domain(user_question)
#     prompt = create_prompt(user_question, selected_domains)

#     inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
#     output = model.generate(
#         **inputs,
#         max_length=250,
#         temperature=0.7,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     answer = response.split("Assistant:")[-1].strip()

#     return jsonify({'answer': answer})

# if __name__ == '__main__':
#     app.run(debug=True)














from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from huggingface_hub import whoami
from prompt_utils import create_prompt, detect_domain

info = whoami()
print("Connected to Hugging Face as:", info["name"])

app = Flask(__name__)

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# this is a decorator in python, which is a design pattern that allows you to modify the
# functionality of a function by wrapping it in another function.
# 
# basicly when some enter the root of the website it executes the functon below
@app.route('/')
def home():
    return render_template('index.html')

# this function executes after the user clicks the ask button
# it fetches data from the user formats is and create a proper prompt
# and then sends it to the model
# after it recives the models respons the
# the answer is send to be cleaned and then send back to
# be displayed to the user
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('question', '')
    selected_domain = detect_domain(user_question)
    prompt = create_prompt(user_question, selected_domain)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=512,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "User: " in response:
    	answer = response.split("User: ")[0].strip()
    else:
    	answer = response.replace(prompt, '').strip() or "sorry you provided no question"
    return jsonify({'answer': answer})

# def create_prompt(user_question, domain):
#     prompt = (
#         "You are an assistant that answers questions about either \"Jordan\" or \"42Amman.\"\n\n"
#         f"Here is the user question:\n{user_question}\n\n"
#         "Use the following information to generate a short, concise, and polite answer:\n\n"
#         f"Information about 42Amman:\n{domain}\n\n"
#         "Please answer politely and keep it brief."
#     )
#     return prompt

if __name__ == '__main__':
    app.run(debug=True)

