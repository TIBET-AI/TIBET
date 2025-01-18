import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from typing import Dict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the text into words
    words = word_tokenize(text)

    # tags the words with parts of speech for the lemmatization
    #words_pos = nltk.pos_tag(words)

    # Remove stopwords
    stopwords_list = set(stopwords.words('english'))
    stopwords_list.add('image')
    # words_pos = [(word, pos) for word, pos in words_pos if word not in stopwords_list]
    words = [word for word in words if word not in stopwords_list]

    # Lemmatize the words
    #lemmatizer = WordNetLemmatizer()
    #words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in words_pos]

    # Join the words back into a preprocessed text
    preprocessed_text = ' '.join(words)

    return preprocessed_text

def word_frequency(text, num_images):
    """
    Calculate the frequency of words in the given text and return a dictionary of word frequencies.

    Parameters:
        text (str): The input text.

    Returns:
        Dict[str, float]: A dictionary containing word frequencies with words as keys and their corresponding normalized frequencies as values.
    """
    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    new_tokens = []
    for t in tokens:
        if not (t == 'yes' or t == 'no' or t == 'none'):
            new_tokens.append(t)
    tokens = new_tokens


    # Calculate the frequency distribution of the words
    fdist = FreqDist(tokens)

    # total_word_count = 0
    # for word, frequency in fdist.items():
    #     total_word_count += frequency

    word_dict: Dict[str, float] = {}
    for word, frequency in fdist.items():
        word_dict[word] = frequency / num_images  # normalization

    return word_dict

def get_freq_dict(vqa_answers, num_images, threshold=0):
    dict_base = word_frequency(vqa_answers, num_images)
    C = [(c, dict_base[c]) for c in dict_base.keys() if dict_base[c] >= threshold]
    return C

def merge_clean_VQAans(VQAns_list, VQA_idx=None):
    # This function takes a list of VQA answers and gives clean concepts
    merged_concepts = ''
    for image_VQAns in VQAns_list:
      if VQA_idx is not None:
        processed_VQAns = preprocess_text(image_VQAns[VQA_idx])
        merged_concepts += processed_VQAns + ' '
      else:
        processed_VQAns = ""
        for i in range(0, len(image_VQAns)):
            processed_VQAns = processed_VQAns + " " + preprocess_text(image_VQAns[i])
        merged_concepts += processed_VQAns + ' '

    return merged_concepts


def merge_clean_concepts(VQAns_list, concept_idx=None):
    merged_concepts = ''
    concept_list = []
    for image_VQAns in VQAns_list:
      if concept_idx is not None:
        processed_VQAns = preprocess_text(image_VQAns[concept_idx])
        merged_concepts += processed_VQAns + ' '
        concept_list.append(processed_VQAns)
      else:
        processed_VQAns = ""
        for i in range(0, len(image_VQAns)):
          processed_VQAns = processed_VQAns + " " + preprocess_text(image_VQAns[i])
        merged_concepts += processed_VQAns + ' '
        concept_list.append(processed_VQAns)

    return merged_concepts, concept_list

def get_concepts(p_dict, bias_axis=None, cf_idx=0, VQA_idx=None):
    # This function takes a dictionary of annotations and returns a list of concepts
    # bias_axis is the axis of bias
    # cf_idx is the index of the counterfactual
    # VQA_idx is the index of the VQA question

    if bias_axis is None:
        concepts = merge_clean_VQAans(p_dict['concepts_initial'], VQA_idx=VQA_idx)
        concepts = get_freq_dict(concepts, len(p_dict['concepts_initial']))
    else:
        concepts = merge_clean_VQAans(p_dict['concepts_cf'][bias_axis][cf_idx], VQA_idx=VQA_idx)
        concepts = get_freq_dict(concepts, len(p_dict['concepts_cf'][bias_axis][cf_idx]))

    return concepts