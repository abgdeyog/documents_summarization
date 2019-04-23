"""
    You need to have /data dirrectory with data from lab8
    You need to have glove.6B.100d.txt dataset to test this code
    You can get it here https://nlp.stanford.edu/data/glove.6B.zip
    ran this file
"""

import nltk
import numpy as np
import networkx
from sklearn.metrics.pairwise import cosine_similarity



stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
ps = nltk.stem.PorterStemmer()


# tokenize text using nltk lib
def tokenize(text):
    return nltk.word_tokenize(text)


# checks if word is appropriate - not a stop word and isalpha
def is_apt_word(word):
    return word not in stop_words and word.isalpha()


# combines all previous methods together
def preprocess(text):
    tokenized = tokenize(text.lower())
    return [w for w in tokenized if is_apt_word(w)]



def build_dict(query):
    '''
    build dictionary from query
    :param query: []
    :return: {term:frequency}
    '''
    dict = {}
    for word in query:
        try:
            dict[word] += 1
        except KeyError:
            dict[word] = 1
    return dict


def cosine_scoring(dict1, dict2):
    """
    :param dict1: dictionary - term:frequency
    :param dict2: dictionary - term:frequency
    :return: score
    """
    score = 0
    for term in dict1:
        try:
            score += dict1[term] * dict2[term]
        except KeyError:
            pass
    if len(dict1)*len(dict2) == 0:
        return 0
    return score/(len(dict1)*len(dict2))


def text_rank(document, n):

    """
    TextRank algorithm
    Document summarization method based on graph algorithm page rank
    :param document: document to process
    :param n: number of sentences to return
    :return [Top k sorted by score array of senteces,
            Sorted by order in text array of top k by score sentences]
    """
    sentences = nltk.tokenize.sent_tokenize(document)
    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]
    glove = open('data/glove.6B.100d.txt', encoding='utf-8')
    word_embedding_length = 100
    word_vectors = {}
    for line in glove:
        token = line.split()
        word_vectors[token[0]] = [float(token[i]) for i in range(1, word_embedding_length + 1)]
    glove.close()
    sentence_vectors = []
    for sentence in preprocessed_sentences:
        mean = [0] * word_embedding_length
        if len(sentence) != 0:
            for i in range(word_embedding_length):
                sum = 0
                count = 0
                for word in sentence:
                    try:
                        sum += word_vectors[word][i]
                        count += 1
                    except KeyError:
                        pass
                mean[i] = sum/len(sentence)
            sentence_vectors.append(mean)
    similarity = {}
    for i in range(len(sentences)):
        similarity[i] = {}
        for j in range(len(sentences)):
            if i != j:
                try:
                    val = cosine_similarity(np.array(sentence_vectors[i]).reshape(1, -1),
                                                     np.array(sentence_vectors[j]).reshape(1, -1))[0][0]
                except:
                    val = 0
                similarity[i][j] = {"weight": val}
    graph = networkx.from_dict_of_dicts(similarity)
    scores = networkx.pagerank(graph)
    top = sorted(scores, key=scores.get, reverse=True)[0:n]
    top_sorted = sorted(top)
    return [sentences[number] for number in top]\
        #, [sentences[number] for number in top_sorted]

file_in = open("data/test.txt", "r")
file_out = open("data/test_out.txt", "w+")
file_out.write(' '.join(text_rank(file_in.read(), 6)))