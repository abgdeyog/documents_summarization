import numpy as np
import scores
import heapq
import nltk

class StackedDecoder:

    def __init__(self, doc, max_length):
        self.doc = doc
        self.max_length = max_length
        self.stack = [[] for i in range(max_length + 1)]

    def build_dict(self, query):
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

    def cosine_scoring(self, dict1, dict2):
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


    def importance(self, sol):

        res = 0
        for sentence in sol:
            res += scores.sentence_importance(self.doc, sentence)
        return 1 # sum of the importance of sentences


    def similarity(self, sentence1, sentence2):
        return self.cosine_scoring(self.build_dict(sentence1), self.build_dict(sentence2))


    def similarity_to_solution(self, sol, sentence):
        return sum([self.similarity(sentence, sol_sent) for sol_sent in sol])/len(sol)


    def summary(self, doc, threshold=1e-2, maxlength = 30):
        stack = [[] for i in range(1, maxlength + 1)] # list of priority queues
        sentences = nltk.sent_tokenize(doc)
        for i in range(1, maxlength):
            for score, sol in stack[i]:
                for s in sentences:
                    #newlen = maxlength + 1
                    if i + len(s) <= maxlength:
                        newlen = i + len(s)
                        if self.similarity_to_solution(sol, sentences) < threshold or len(sol) == 0:
                            newsol = sol.copy()
                            newsol = heapq.heappush(newsol, sentences)
                        else:
                            continue
                        newscore = self.importance(newsol)
                        heapq.heappush(stack[newlen], (-newscore, newsol))
                        # append to stack[newlen]
        return heapq.heappop([maxlength])

