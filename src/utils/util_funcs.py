import spacy
from scipy.spatial import distance


def load_nlp_model():
    nlp = spacy.load("en_core_web_md")
    return nlp


def clean_tokens(token, del_stopwords=True, del_punct=True, del_references=True, 
    del_non_alpha_start_end=True, lemmatize=True, lowercase=True, del_no_vector=True):
    
    if del_stopwords and token.is_stop:
        return None

    if del_punct and token.is_punct:
        return None

    if del_references and token.like_email:
        return None

    if del_references and token.like_url:
        return None

    if del_non_alpha_start_end and not token.is_alpha:
        return None

    if del_no_vector and token.vector_norm < 0.001:
        return None

    return token


def get_tokens(text: str, nlp, vectors=True, del_stopwords=True, del_punct=True, del_references=True, 
    del_non_alpha_start_end=True, lemmatize=True, lowercase=True, has_vector=True):

    final_tokens = []
    for token in nlp(text):
        clean_token = clean_tokens(token)
        if clean_token:
            if not vectors:
                final_tokens.append(clean_token.text)
            else:
                final_tokens.append(clean_token.vector)
    return final_tokens


def get_similarity(v1, v2):
    return 1 - distance.cosine(v1, v2)