from data import domain_qa_data

def create_prompt(user_question, selected_domains):
    intro = "You are a helpful assistant. Answer based only on the information provided previously.\n\n"
    knowledge = ""

    domains_to_use = selected_domains or domain_qa_data.keys()

    for domain in domains_to_use:
        knowledge += f"## {domain} Info:\n"
        for pair in domain_qa_data.get(domain, []):
            knowledge += f"Q: {pair['q']}\nA: {pair['a']}\n"
        knowledge += "\n"

    return f"{knowledge} \nUser: {user_question}{intro}\n"

def detect_domain(question):
    q = question.lower()

    domain_keywords = {
        "42Amman": [
            "42", "piscine", "black hole", "peer", "xp", "correction", "evaluation",
            "project-based", "coding school", "tuition-free", "campus", "cursus", "cluster"
        ],
        "Jordan": [
            "jordan", "amman", "mansaf", "petra", "wadi rum", "dead sea", "aqaba",
            "arabic", "jerash", "maqluba", "kunafa", "middle east"
        ]
    }

    matched_domains = []

    for domain, keywords in domain_keywords.items():
        if any(keyword in q for keyword in keywords):
            matched_domains.append(domain)

    # If no keyword matches, return all domains as fallback
    return matched_domains if matched_domains else ["42Amman", "Jordan"]