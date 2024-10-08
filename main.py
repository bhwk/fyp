import taskgen


def llm(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    response = client.chat.completions.create(
        model="openhermes",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    else:
        return ""


sentence_style = taskgen.Function(
    fn_description="Give the user a recommendation based on the <user_input>.",
    output_format={"output": "sentence"},
    fn_name="sentence_with_user_recommendation",
    llm=llm,
)


def jaccard_similarity(query, document):
    query = query.lower().split(" ")
    document = document.lower().split(" ")
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def provide_recommendation_based_on_user_input(
    shared_variables, user_input: str
) -> str:
    """Matches user_input against corpus of documents and returns the best recommendation"""
    corpus_of_documents = shared_variables["Corpus"]
    similarities = []
    for doc in corpus_of_documents:
        similarity = jaccard_similarity(user_input, doc)
        similarities.append(similarity)
    return corpus_of_documents[similarities.index(max(similarities))]


corpus_of_documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new.",
    "Attend a live music concert and feel the rhythm.",
    "Go for a hike and admire the natural scenery.",
    "Have a picnic with friends and share some laughs.",
    "Explore a new cuisine by dining at an ethnic restaurant.",
    "Take a yoga class and stretch your body and mind.",
    "Join a local sports league and enjoy some friendly competition.",
    "Attend a workshop or lecture on a topic you're interested in.",
    "Visit an amusement park and ride the roller coasters.",
]


my_agent = taskgen.Agent(
    "Activity assistant",
    "You are an agent that makes gives the user the best recommendation based on their query",
    shared_variables={"Corpus": corpus_of_documents},
    global_context="Corpus: <Corpus>",
    llm=llm,
)
my_agent.assign_functions([provide_recommendation_based_on_user_input])
output = my_agent.run("I like to hike")
output = my_agent.reply_user()
