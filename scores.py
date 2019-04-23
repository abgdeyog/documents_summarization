from sklearn.feature_extraction.text import TfidfVectorizer


def jaccard_similarity(sent1, sent2):
    """

    :param sent1: str, sentence1
    :param sent2: str, sentence2
    :return: Jaccard coefficient
    """
    set1 = set(sent1)
    set2 = set(sent2)
    return len(set1.intersection(set2)) / len(set1.union(set2))


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
    if len(dict1) * len(dict2) == 0:
        return 0
    return score / (len(dict1) * len(dict2))


def sentence_length(sentence):
    sentence = sentence.replace('\n', '').replace("'", "").replace("`", "")
    return len(sentence)


def upper_case_count(sentence):
    count = 0
    for s in sentence:
        if s.isupper():
            count += 1
    return count


def numerical_count(sentence):
    count = 0
    tokens = sentence.split(' ')
    for t in tokens:
        if t.isnumeric():
            count += 1

    return count


def sentence_pos(sentence_id, sentences):
    return sentence_id / len(sentences)


def tf_id_init(sentences):
    tf = TfidfVectorizer(stop_words='english', use_idf=True)
    tfidf_matrix = tf.fit_transform(sentences)

    return tfidf_matrix


def tf_idf_sum(sentence_id, tfidf_matrix):
    return tfidf_matrix[sentence_id].sum()


def sentence_importance_score(ids, sentences):
    scores = [0] * len(sentences)
    tfidf_matrix = tf_id_init(sentences)
    for sentence_id in ids:
        scores[sentence_id] += sentence_length(sentences[sentence_id])
        scores[sentence_id] += upper_case_count(sentences[sentence_id])
        scores[sentence_id] += numerical_count(sentences[sentence_id])
        # scores[sentence_id] += sentence_pos(sentence_id, sentences)
        scores[sentence_id] += tf_idf_sum(sentence_id, tfidf_matrix)

    return scores
