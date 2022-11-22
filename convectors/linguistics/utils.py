import re

name2regex = {
    "url": r"((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)",
    "emoji": re.compile("(["                     # .* removed
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "])", flags=re.UNICODE),
    "hashtag": r"(\#[a-zA-Z0-9_]+\b)",
    "mention": r"(\@[a-zA-Z0-9_]+\b)",
    "email": r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
    "coin": r"(\$[a-zA-Z]+\b)"
}
