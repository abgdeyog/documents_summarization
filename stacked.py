import numpy as np
import scores
import heapq
import nltk
from sklearn.metrics.pairwise import cosine_similarity


class StackedDecoder:

    # tokenize text using nltk lib
    def tokenize(self, text):
        return nltk.word_tokenize(text)

    # checks if word is appropriate - not a stop word and isalpha
    def is_apt_word(self, word):
        return word not in self.stop_words and word.isalpha()

    # combines all previous methods together
    def preprocess(self, text):
        tokenized = self.tokenize(text.lower())
        return [w for w in tokenized if self.is_apt_word(w)]

    def __init__(self, doc):

        self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is',
                           'it',
                           'its',
                           'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}

        self.ps = nltk.stem.PorterStemmer()

        glove = open('data/glove.6B.100d.txt', encoding='utf-8')
        self.sentences = nltk.tokenize.sent_tokenize(doc)
        word_embedding_length = 100
        word_vectors = {}
        for line in glove:
            token = line.split()
            word_vectors[token[0]] = [float(token[i]) for i in range(1, word_embedding_length + 1)]
        self.preprocessed_sentences = [self.preprocess(sentence) for sentence in self.sentences]
        glove.close()
        self.importance = scores.sentence_importance_score(list(range(len(self.sentences))), self.sentences)
        self.sentence_vectors = []
        self.tokenized_sentences = [self.tokenize(text.lower()) for text in self.sentences]
        for sentence in self.preprocessed_sentences:
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
                    mean[i] = sum / len(sentence)
                self.sentence_vectors.append(mean)

    def similarity(self, sentence_id_1, sentence_id_2):
        return cosine_similarity(np.array(self.sentence_vectors[sentence_id_1]).reshape(1, -1),
                                 np.array(self.sentence_vectors[sentence_id_2]).reshape(1, -1))[0][0]

    def solution_importance(self, sol):
        return sum([self.importance[i] for i in sol])

    def similarity_to_solution(self, sol, sentence):
        return sum([self.similarity(sentence, sol_sent) for sol_sent in sol]) / len(sol)

    def summary(self, threshold=0.8, maxlength=30):
        stacks = [[] for i in range(maxlength + 2)]  # list of priority queues

        for i in range(0, len(self.tokenized_sentences)):
            index = maxlength
            if len(self.tokenized_sentences[i]) <= maxlength:
                index = len(self.tokenized_sentences[i])
            heapq.heappush(stacks[index], (-self.importance[i], {i: 1}))

        for i in range(0, maxlength + 1):
            if len(stacks[i]) == 0:
                continue
            stack_clone = stacks[i].copy()
            for score, sol in stack_clone:
                for s in range(len(self.tokenized_sentences)):
                    if s in sol:
                        continue
                    newlen = maxlength + 1

                    if i + len(self.tokenized_sentences[s]) <= maxlength:
                        newlen = i + len(self.tokenized_sentences[s])

                    if self.similarity_to_solution(sol, s) < threshold:
                        newsol = sol.copy()
                        newsol[s] = 1
                        newscore = self.solution_importance(newsol)
                        tmp = len(stacks[newlen])
                        if tmp == 0 or -newscore < stacks[newlen][0][0]:
                            heapq.heappush(stacks[newlen], (-newscore, newsol))
        return ' '.join([self.sentences[i] for i in self.get_best(stacks)])

    def get_best(self, stacks):
        res = []
        best_score = -1
        for stack in stacks:
            if len(stack) == 0:
                continue
            if -stack[0][0] > best_score:
                best_score = -stack[0][0]
                res = stack[0][1]
        return res