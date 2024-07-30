import re


def find_proper_names_and_acronyms(text):
    # Regex pattern to match proper names (assuming names start with a capital letter)
    proper_name_pattern = r"\b[A-Z][a-z]*\s[A-Z][a-z]*\b"
    # Regex pattern to match acronyms (all uppercase letters possibly with periods or numbers)
    acronym_pattern = r"\b[A-Z]{2,}(?:\.[A-Z]+)*\b"

    proper_names = re.findall(proper_name_pattern, text)
    acronyms = re.findall(acronym_pattern, text)

    return proper_names, acronyms
