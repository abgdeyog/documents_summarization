import sklearn.metrics.pairwise as sklearn
import sys
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from k_medoids import KMedoids
import utils
import scores


def clean(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words('english')

    # remove stopwords from the sentences
    clean_sentences = [' '.join([i for i in r.split() if i not in stop_words]) for r in clean_sentences]
    return clean_sentences


def cos_sim(vector1, vector2):
    """
    Cosine similarity scoring
    :param vector1: vector of terms
    :param vector2: vectors of terms
    :return: score
    """
    v1 = vector1.reshape(1, 100)
    v2 = vector2.reshape(1, 100)
    return 1 - sklearn.cosine_similarity(v1, v2)[0, 0]


def generate_sentence_vectors(sentences, embedding_file):
    # split text into list of sentences
    n = len(sentences)
    clean_sentences = clean(sentences)
    # Extract word vectors
    word_embeddings = {}
    for line in embedding_file:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefficients

    # Compute sentence vector as average vector of word-vectors
    sentence_vectors = []
    for i in clean_sentences:
        zeros = np.zeros((100,))
        if len(i) == 0:
            v = zeros
        else:
            length = len(i.split())
            v = sum([word_embeddings.get(w, zeros) for w in i.split()]) / length

        sentence_vectors.append(v)

    sentence_vectors = np.array(sentence_vectors)
    return sentence_vectors


def main():

    docs = utils.parse_docs('/Users/mac/Downloads/snickebod/duc2004/docs')
    print(sys.getsizeof(docs))
    print(len(docs))
    for docno in docs:
        print(docno)

        sentences = sent_tokenize(docs[docno])
        with open('glove.6B.100d.txt', encoding='utf-8') as file:
            sentence_vectors = generate_sentence_vectors(sentences, file)

        model = KMedoids(n_clusters=5, dist_func=cos_sim)
        centers = model.fit(sentence_vectors, plotit=True, verbose=True)

        sent_scores = scores.sentence_importance_score(centers, sentences)

        centers.sort(key=lambda x: sent_scores[x], reverse=True)
        for i in range(len(sent_scores)):
            if sent_scores[i] != 0:
                print(f'Score of {i}th sentence: {sent_scores[i]}')
        print(centers)
        for _id in centers:
            print(sentences[_id].replace('\n', ''))

    file.close()

if __name__ == '__main__':
    main()
