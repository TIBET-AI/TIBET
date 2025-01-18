import numpy as np
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

def get_syn(word, ant=False):
  words = []
  for syn in wordnet.synsets(word):
    for i in syn.lemmas():
      if ant:
        if i.antonyms():
          words.append(i.antonyms()[0].name().lower())
      else:
        words.append(i.name().lower())
  if word == 'male':
    words.append('man')
  if word == 'female':
    words.append('woman')

  return list(set(words))

def clean_concepts_syn(C_init, C_cf=None):

    if C_cf is None:
      C_cf = C_init

    # Create a vocabulary
    vocab = set([c[0] for c in C_init] + [c[0] for c in C_cf])

    # get the synonyms for each word from wordnet
    syns = {}
    for v in vocab:
        syns[v] = get_syn(v)

    # if any concept is a synonym of any other concept, then they are the same. Get tuples of synonyms for the concepts
    syns_tuples = []
    for v1 in vocab:
        for v2 in vocab:
            if v1 in syns[v2] or v2 in syns[v1] and (v2, v1) not in syns_tuples:
                syns_tuples.append((v1, v2))

    # in C_init and C_cf, replace the synonyms with the first word in the tuple, and aggregate the scores of synonyms.
    # For example if C_init = [('book', 0.4), ('books', 0.3), ('beach', 0.3)] and syns_tuples = [('book', 'books')], then C_init_new = [('book', 0.7), ('beach', 0.3)]
    C_init_new = []
    for c in C_init:
        for s in syns_tuples:
            if c[0] in s:
                c = (s[0], c[1])
        C_init_new.append(c)

    C_cf_new = []
    for c in C_cf:
        for s in syns_tuples:
            if c[0] in s:
                c = (s[0], c[1])
        C_cf_new.append(c)

    num_idx = len(C_init_new)
    for idx in range(num_idx):
        c = C_init_new[idx]
        for idx2 in range(idx+1, num_idx):
            c2 = C_init_new[idx2]
            if c[0] == c2[0]:
                C_init_new[idx] = (c[0], c[1] + c2[1])
                C_init_new[idx2] = (c[0], 0)
    C_init_new = [c for c in C_init_new if c[1] != 0]

    num_idx = len(C_cf_new)
    for idx in range(num_idx):
        c = C_cf_new[idx]
        for idx2 in range(idx+1, num_idx):
            c2 = C_cf_new[idx2]
            if c[0] == c2[0]:
                C_cf_new[idx] = (c[0], c[1] + c2[1])
                C_cf_new[idx2] = (c[0], 0)
    C_cf_new = [c for c in C_cf_new if c[1] != 0]

    # If a concept is present in C_init but not in C_cf, then add it to C_cf with score 0.0, and vice versa
    for c in C_init_new:
        if c[0] not in [c2[0] for c2 in C_cf_new]:
            C_cf_new.append((c[0], 0.0))

    for c in C_cf_new:
        if c[0] not in [c2[0] for c2 in C_init_new]:
            C_init_new.append((c[0], 0.0))

    # order concepts by score
    C_init_new = sorted(C_init_new, key=lambda x: x[0])
    C_cf_new = sorted(C_cf_new, key=lambda x: x[0])

    return C_init_new, C_cf_new

def compute_association(C_init, C_cf, funcx):

    C_init, C_cf = clean_concepts_syn(C_init, C_cf)

    p = [c[1] for c in C_init]
    q = [c[1] for c in C_cf]

    # Compute HistIOU distance
    h = funcx(p,q)
    
    return h