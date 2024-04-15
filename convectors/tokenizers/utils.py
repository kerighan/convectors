import re


# =============================================================================
# Functions
# =============================================================================


def words_from(text):
    return re.findall(
        r"([\w]+|[\U0001F300-\U0001F5FF|\U0001F600-\U0001F64F|"
        r"\U0001F680-\U0001F6FF|\u2600-\u26FF\u2700-\u27BF])",
        text,
        re.UNICODE,
    )


def words_and_punctuation_from(text):
    return re.findall(
        r"([\w]+|[\U0001F300-\U0001F5FF|\U0001F600-\U0001F64F|"
        r"\U0001F680-\U0001F6FF|\u2600-\u26FF\u2700-\u27BF]|"
        r"[,?;.:\/!()\[\]'\"’\\><+-=])",
        text,
        re.UNICODE,
    )


def sentences_from(text, split_lines=False):
    if split_lines:
        sentences = re.split(r"[\n\r]", text)
        res = []
        for sentence in sentences:
            res.extend(
                re.split(
                    r"(?<!\w[\t\r!:.?|•]\w.)(?<![A-Z][a-z][.])(?<=[\t\r!:.?|•…])\s",
                    sentence,
                )
            )
        return res
    return re.split(
        r"(?<!\w[\t\r!:.?|•]\w.)(?<![A-Z][a-z][.])(?<=[\t\r!:.?|•…])\s", text
    )


def tokenize(
    text,
    stopwords=None,
    strip_accents=False,
    strip_punctuation=True,
    sentence_tokenize=False,
    word_tokenize=True,
    lower=True,
    split_lines=False,
):
    text = str(text)
    # turn lowercase
    if lower:
        text = text.lower()
    # remove accents
    if strip_accents:
        from unidecode import unidecode

        text = unidecode(text)
    if not sentence_tokenize:
        if strip_punctuation:
            # tokenize by removing punctuation
            words = words_from(text)
        else:
            # tokenize by keeping punctuation
            words = words_and_punctuation_from(text)
    elif word_tokenize:
        words = []
        sentences = sentences_from(text, split_lines=split_lines)
        if strip_punctuation:
            words = [words_from(s) for s in sentences]
        else:
            words = [words_and_punctuation_from(s) for s in sentences]
    else:
        return sentences_from(text, split_lines=split_lines)

    # remove stopwords
    if stopwords is not None:
        if not sentence_tokenize:
            words = [w.strip() for w in words if w.strip() not in stopwords]
        else:
            words = [
                [w.strip() for w in sentence if w.strip() not in stopwords]
                for sentence in words
            ]
    return words
