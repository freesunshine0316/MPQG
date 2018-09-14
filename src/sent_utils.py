import re

class QASentence(object):
    def __init__(self, tokText, annotation=None, ID_num=None, isLower=False, end_sym=None):
        self.tokText = tokText
        self.annotation = annotation
        # it's the answer sequence
        if end_sym != None:
            self.tokText += ' ' + end_sym
        if isLower:
            self.tokText = self.tokText.lower()
        self.words = re.split("\\s+", self.tokText)
        self.startPositions = []
        self.endPositions = []
        self.POSs = []
        self.NERs = []
        if annotation != None:
            positions = re.split("\\s+", annotation['positions'])
            for i in xrange(len(positions)):
                tmps = re.split("-", positions[i])
                self.startPositions.append(int(tmps[1]))
                self.endPositions.append(int(tmps[2]))
            self.POSs = annotation['POSs']
            self.NERs = annotation['NERs']
        self.length = len(self.words)
        self.ID_num = ID_num

        self.index_convered = False
        self.chunk_starts = None

    def get_length(self):
        return self.length

    def get_max_word_len(self):
        max_word_len = 0
        for word in self.words:
            cur_len = len(word)
            if max_word_len < cur_len: max_word_len = cur_len
        return max_word_len

    def get_char_len(self):
        char_lens = []
        for word in self.words:
            cur_len = len(word)
            char_lens.append(cur_len)
        return char_lens

    def convert2index(self, word_vocab, char_vocab, POS_vocab, NER_vocab, max_char_per_word=-1):
        if self.index_convered: return # for each sentence, only conver once

        if word_vocab is not None:
            self.word_idx_seq = word_vocab.to_index_sequence(self.tokText)

        if char_vocab is not None:
            self.char_idx_matrix = char_vocab.to_character_matrix(self.tokText, max_char_per_word=max_char_per_word)

        if POS_vocab is not None:
            self.POS_idx_seq = POS_vocab.to_index_sequence(self.POSs)

        if NER_vocab is not None:
            self.NER_idx_seq = NER_vocab.to_index_sequence(self.NERs)

        self.index_convered = True


if __name__ == "__main__":
    pass
