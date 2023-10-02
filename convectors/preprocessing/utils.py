import re


NAME_TO_REGEX = {
    "url": r"https?://[\w\d:#@%/;$()~_?\+-=\\\.&]+(?<!\.)",
    "emoji": re.compile("(["                     # .* removed
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "])", flags=re.UNICODE),
    "flag": re.compile(
        u'(\U0001F1F2\U0001F1F4)|'       # Macau flag
        u'([\U0001F1E6-\U0001F1FF]{2})'  # flags
        "+", flags=re.UNICODE),
    "hashtag": r"(\#[a-zA-Z0-9_]+\b)",
    "mention": r"(\@[a-zA-Z0-9_]+\b)",
    "email": r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
    "coin": r"(\$[a-zA-Z]+\b)"
}
