import re


def remove_stopword(proper_name, stopwords):
    for stopword in stopwords:
        if proper_name.startswith(stopword):
            proper_name = proper_name[len(stopword) :]
    return proper_name


def find_proper_names_and_acronyms(text):
    # Regex pattern to match proper names (two or more capitalized words)
    proper_name_pattern = r"\b([A-Z][a-z]+(?:[\s\-]+[A-Z][a-z]+)+)\b"

    # Regex pattern to match acronyms (all uppercase letters possibly with periods or numbers)
    acronym_pattern = r"\b[A-Z]{2,}(?:\.[A-Z]+)*\b"

    proper_names = re.findall(proper_name_pattern, text)

    stopwords = [
        "Le ",
        "La ",
        "Les ",
        "L'",
        "L'",
        "De ",
        "Du ",
        "Des ",
        "Au ",
        "Un ",
        "Une ",
        "Aux ",
        "Pour ",
        "Contre ",
        "Ce ",
        "Ces ",
    ]
    proper_names = [remove_stopword(it, stopwords) for it in proper_names]
    acronyms = re.findall(acronym_pattern, text)
    return proper_names, acronyms
