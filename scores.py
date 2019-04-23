from sklearn.feature_extraction.text import TfidfVectorizer


def sentence_length(sentence):
    print(sentence)
    sentence = sentence.replace('\n', '').replace("'", "").replace("`", "")
    print(sentence)
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


def sentence_importance_score(sentences):
    scores = []
    tfidf_matrix = tf_id_init(sentences)
    for sentence_id in range(len(sentences)):
        scores.append(0)
        scores[sentence_id] += sentence_length(sentences[sentence_id])
        scores[sentence_id] += upper_case_count(sentences[sentence_id])
        scores[sentence_id] += numerical_count(sentences[sentence_id])
        scores[sentence_id] += sentence_pos(sentence_id, sentences)
        scores[sentence_id] += tf_idf_sum(sentence_id, tfidf_matrix)

    return scores
