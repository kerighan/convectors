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


def find_numbers(text):
    # Regex patterns to capture numbers with context
    patterns = [
        r"(\$?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?:M|K|B)?)",  # Plain numbers, monetary units, scientific notation
        r"(\d+(?:\.\d+)?%)",  # Percentages
        r"(\d+(?:,\d{3})*(?:\.\d+)?\s?(?:kg|g|m|km|lb|oz|L|ml|t|°C|°F))",  # Units (e.g., weight, distance, temperature)
        r"(\d+\s?(?:to|–|-)\s?\d+)",  # Ranges (e.g., "10-20", "100 to 200")
        r"(\d{4})",  # Years (e.g., 2023)
        r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)",  # Dates (e.g., "10/12/2024")
        r"(\$\d+(?:,\d{3})*(?:\.\d+)?)",  # Dollar amounts
        r"(€\d+(?:,\d{3})*(?:\.\d+)?)",  # Euro amounts
        r"(USD \d+(?:,\d{3})*(?:\.\d+)?)",  # USD-prefixed amounts
    ]

    # Combine all patterns into a single regex
    combined_pattern = "|".join(patterns)

    # Find all matches with context
    matches = re.findall(combined_pattern, text)

    # Flatten and clean up results (filter empty groups)
    results = [match for group in matches for match in group if match]

    # Classify the results based on the context
    classified_results = []
    for result in results:
        if "%" in result:
            classified_results.append({"type": "percentage", "value": result})
        elif re.search(r"kg|g|m|km|lb|oz|L|ml|t|°C|°F", result):
            classified_results.append({"type": "unit", "value": result})
        elif re.search(r"to|–|-", result):
            classified_results.append({"type": "range", "value": result})
        elif re.search(r"\$\d+|€\d+|USD", result):
            classified_results.append({"type": "currency", "value": result})
        elif re.match(r"\d{4}$", result):
            classified_results.append({"type": "year", "value": result})
        elif re.match(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", result):
            classified_results.append({"type": "date", "value": result})
        else:
            classified_results.append({"type": "plain_number", "value": result})

    # return classified_results
    return [it["value"] for it in classified_results]
