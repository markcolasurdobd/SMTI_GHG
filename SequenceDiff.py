from difflib import SequenceMatcher



text1 = "I like dogs"

text2 = "I like dog"

matcher = SequenceMatcher(None, text1, text2)

ratio = matcher.ratio()

print(ratio)--