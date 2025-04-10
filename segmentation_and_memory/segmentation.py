import nltk
import re

nltk.download('punkt', quiet=True)

def segment_by_sentences(text):
    """Segment the input text into sentences using NLTK."""
    return nltk.sent_tokenize(text)

def segment_by_paragraphs(text):
    """Segment the input text into paragraphs by splitting at newlines."""
    return text.split('\n')

def segment_by_fixed_length(text, max_length=100):
    """Segment the text into chunks based on a maximum character length."""
    segments = []
    while len(text) > max_length:
        segment = text[:max_length]
        segments.append(segment)
        text = text[max_length:]
    if text:  # Add the remaining part if there's any
        segments.append(text)
    return segments

def segment_by_custom_delimiters(text, delimiter=","):
    """Segment the input text by a custom delimiter (default: comma)."""
    return [segment.strip() for segment in text.split(delimiter)]

def segment_text(text, method="sentences", max_length=10, delimiter=","):
    """Segment the text based on the chosen method."""
    if not text:
        return []

    if method == "sentences":
        return segment_by_sentences(text)
    elif method == "paragraphs":
        return segment_by_paragraphs(text)
    elif method == "length":
        return segment_by_fixed_length(text, max_length)
    elif method == "delimiters":
        return segment_by_custom_delimiters(text, delimiter)
    else:
        raise ValueError("Invalid segmentation method chosen.")
