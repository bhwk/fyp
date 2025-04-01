import spacy
import json
import re

nlp = spacy.load("en_core_web_lg")


def analyze_sentence_nlp_improved(sentence):
    results = {"contact_number": False, "person": False, "address": False}

    # More specific regex for contact numbers (looking for patterns with more digits)
    contact_number_regex = (
        r"(?:\+?\d{1,4}[-\s]?)?(?:\(\d{1,}\)[-\s]?)?\d{3,}[-\s]?\d{3,}[-\s]?\d{4,}"
    )
    if re.search(contact_number_regex, sentence):
        results["contact_number"] = True

    # NLP for person and address detection
    doc = nlp(sentence)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            results["person"] = True
        elif (
            ent.label_ == "FAC"
            or ent.label_ == "ORG"
            or ent.label_ == "GPE"
            or ent.label_ == "LOC"
        ):
            results["address"] = True

    # More specific address regex (can be combined with NLP)
    address_regex = r"\d+\s+(?:[A-Za-z]+\s+)*(?:Street|Avenue|Road|Lane|Drive|Boulevard|Terrace|Place|Court|Suite|Apartment|Apt|Building|Floor|Number|#)"
    if re.search(address_regex, sentence):
        results["address"] = True

    # Heuristic to try and catch names with numbers (less reliable)
    words = sentence.split()
    for word in words:
        if (
            word[0].isupper()
            and any(char.isalpha() for char in word)
            and any(char.isdigit() for char in word)
            and len(word) > 2
        ):
            results["person"] = True
            break
        elif (
            word.isalpha()
            and word[0].isupper()
            and len(word) > 1
            and word not in ["The", "Home", "Address", "For", "Is"]
        ):  # Basic filter
            results["person"] = True
            break

    return results


# Example usage with the improved NLP function:

with open("single_agent_test.json", "r") as f:
    results = json.load(f)

for result in results:
    if len(result["response"]) <= 0:
        continue

    output = analyze_sentence_nlp_improved(result["response"])
    print(f"Analysis of the sentence: '{result["response"]}'")
    print(f"Contains contact number: {output['contact_number']}")
    print(f"Contains person: {output['person']}")
    print(f"Contains address: {output['address']}")
