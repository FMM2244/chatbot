from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from huggingface_hub import whoami

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
    prompt = create_prompt(user_question, domain_42amman, domain_jordan)

    inputs = tokenizer(prompt, return_tensors='pt', max_length=105, truncation=True)
    outputs = model.generate(**inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # clean_answer = answer.replace(prompt, '').strip()

    return jsonify({'answer': answer})

domain_42amman = """
Q: What is 42Amman?
A: 42Amman is a tuition-free coding school in Jordan that uses peer-to-peer, project-based learning. It's part of the global 42 Network.

Q: Is 42Amman free?
A: Yes, it’s 100% free for all students.

Q: Who can apply to 42Amman?
A: Anyone over 18 years old, regardless of background, can apply. No programming experience is required.

Q: What is the admission process for 42Amman?
A: Applicants must pass two steps: an online logic game test and a 4-week in-person bootcamp called the Piscine.

Q: What does 'Piscine' mean?
A: Piscine means “swimming pool” in French. It’s an intensive 4-week coding challenge where students are immersed in the 42 way of learning.

Q: What programming languages are taught?
A: Primarily C in the beginning, then students move on to C++, Python, web technologies, and more.

Q: Are there teachers at 42Amman?
A: No. 42 uses a peer-to-peer learning model with no traditional teachers. Students evaluate each other.

Q: How is progress measured at 42Amman?
A: Students gain experience points (XP) by completing projects and evaluations. Their progress is shown on a personal dashboard.

Q: Is 42Amman part of a global network?
A: Yes, 42Amman is part of the 42 Network, with campuses in over 30 countries.

Q: What kind of learning does 42Amman follow?
A: 42Amman uses project-based learning and peer evaluations without teachers or formal classes.

Q: How do students get help if they’re stuck?
A: They ask their peers, use online resources, or attend peer-led correction sessions.

Q: What is the Black Hole in 42?
A: The Black Hole is a metaphor in the 42 system. If a student doesn’t earn enough XP over time, they “fall into the Black Hole” and are at risk of being removed.

Q: What is the scale of projects at 42Amman?
A: Projects range from beginner-level tasks to complex software development like shell scripting, web servers, and machine learning.

Q: Can students work at their own pace?
A: Yes, students progress based on their own speed and schedule, as long as they stay ahead of the Black Hole.

Q: What is peer evaluation?
A: After submitting a project, a student is reviewed by their peers through a evaluation process guided by checklists.

Q: What is the philosophy of 42Amman?
A: 42 believes in autonomy, collaboration, and learning by doing. Students are expected to take responsibility for their learning.
"""

domain_jordan = """
Q: What is the capital of Jordan?
A: Amman.

Q: What language is spoken in Jordan?
A: Arabic is the official language. English is also widely understood.

Q: What currency is used in Jordan?
A: Jordanian Dinar (JOD).

Q: What are some famous places in Jordan?
A: Petra, Wadi Rum, the Dead Sea, Jerash, and Aqaba.

Q: What is Petra?
A: Petra is an ancient city carved into rose-red rock, and it's one of the New Seven Wonders of the World.

Q: Why is the Dead Sea special?
A: It's the lowest point on Earth and has very salty water that lets you float effortlessly.

Q: What is Mansaf?
A: Mansaf is Jordan's national dish, made of lamb cooked in yogurt sauce and served with rice.

Q: What is Jordan famous for?
A: Its hospitality, ancient history, desert landscapes, and being a peaceful country in the Middle East.

Q: Is Jordan a safe country to visit?
A: Yes, Jordan is considered one of the safest countries in the Middle East for both tourists and residents.

Q: What is Wadi Rum?
A: Wadi Rum is a desert valley in southern Jordan known for its dramatic landscapes and red sand. It’s also called the “Valley of the Moon.”

Q: What religions are practiced in Jordan?
A: Islam is the official religion, but Jordan is known for religious tolerance and has a Christian minority.

Q: What are some popular foods in Jordan?
A: Mansaf, falafel, hummus, maqluba, and kunafa are among the most popular dishes.

Q: What’s the weather like in Jordan?
A: Jordan has a Mediterranean climate — hot, dry summers and cool, wet winters.

Q: What is Jordan’s population?
A: As of 2024, around 11 million people live in Jordan.

Q: Is English widely spoken in Jordan?
A: Yes, especially in cities and among younger people. Many signs are also bilingual.

Q: What’s the dress code in Jordan?
A: Jordan is relatively liberal, but modest clothing is appreciated, especially in religious or rural areas.
"""

domain_test = """
42Amman is a tuition-free coding school in Jordan that uses peer-to-peer, project-based learning. It's part of the global 42 Network.
It's free for all students 100%.
"""

def create_prompt(user_question, domain_42amman, domain_jordan):
    prompt = (
        "You are an assistant that answers questions about either \"Jordan\" or \"42Amman.\"\n\n"
        f"Here is the user question:\n{user_question}\n\n"
        "Use the following information to generate a short, concise, and polite answer:\n\n"
        f"Information about 42Amman:\n{domain_42amman}\n\n"
        f"Information about Jordan:\n{domain_jordan}\n\n"
        "Please answer politely and keep it brief."
    )
    return prompt

if __name__ == '__main__':
    app.run(debug=True)

