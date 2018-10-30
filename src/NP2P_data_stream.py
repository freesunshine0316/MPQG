import json
import re
import numpy as np
import random
import padding_utils
from sent_utils import QASentence
import phrase_lattice_utils


def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines


def read_all_GQA_questions(inpath, isLower=True, switch=False):
    with open(inpath) as dataset_file:
        dataset_json = json.load(dataset_file, encoding='utf-8')
        dataset = dataset_json['data']
    all_questions = []
    max_answer_len = 0
    end_sym_q = '</s>' if switch is True else None
    end_sym_a = '</s>' if switch is False else None
    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            context_annotations = paragraph['annotations']
            passageSent = QASentence(context, context_annotations, ID_num=None, isLower=isLower)
            for question in paragraph['qas']:
                question_text = question['question']
                question_id = question['id']
                if not question.has_key('annotations'):
                    continue
                question_annotation = question['annotations']
                questionSent = QASentence(question_text, question_annotation, ID_num=question_id, isLower=isLower, end_sym=end_sym_q)
                answer_text = question['answers'][0]['text']
                answer_annotation = question['answers'][0]['annotations']
                answerSent = QASentence(answer_text, answer_annotation, isLower=isLower, end_sym=end_sym_a)
                if switch:
                    max_answer_len = max(max_answer_len, len(questionSent.tokText.split()))
                    all_questions.append((passageSent, questionSent, answerSent))
                else:
                    max_answer_len = max(max_answer_len, len(answerSent.tokText.split()))
                    all_questions.append((passageSent, answerSent, questionSent))
    return all_questions, max_answer_len


def read_all_GenerationDatasets(inpath, isLower=True):
    with open(inpath) as dataset_file:
        dataset = json.load(dataset_file, encoding='utf-8')
    all_instances = []
    max_answer_len = 0
    for instance in dataset:
        ID_num = None
        if instance.has_key('id'): ID_num = instance['id']

        text1 = instance['annotation1']['toks'] if 'annotation1' in instance else instance['text1']
        if text1 == "": continue
        annotation1 = instance['annotation1'] if 'annotation1' in instance else None
        sent1 = QASentence(text1, annotation1, ID_num=ID_num, isLower=isLower)

        text2 = instance['annotation2']['toks'] if 'annotation2' in instance else instance['text2']
        if text2 == "": continue
        annotation2 = instance['annotation2'] if 'annotation2' in instance else None
        sent2 = QASentence(text2, annotation2, ID_num=ID_num, isLower=isLower, end_sym='</s>')
        max_answer_len = max(max_answer_len, sent2.get_length()) # text2 is the sequence to be generated

        sent3 = None
        if instance.has_key('text3'):
            text3 = instance['annotation3']['toks'] if 'annotation3' in instance else instance['text3']
            annotation3 = instance['annotation3'] if 'annotation3' in instance else None
            sent3 = QASentence(text3, annotation3, ID_num=ID_num, isLower=isLower)
        all_instances.append((sent1, sent2, sent3))
    return all_instances, max_answer_len


def read_generation_datasets_from_fof(fofpath, isLower=True):
    all_paths = read_text_file(fofpath)
    all_instances = []
    max_answer_len = 0
    for cur_path in all_paths:
        print(cur_path)
        (cur_instances, cur_max_answer_len) = read_all_GenerationDatasets(cur_path, isLower=isLower)
        print("cur_max_answer_len: %s" % cur_max_answer_len)
        all_instances.extend(cur_instances)
        if max_answer_len<cur_max_answer_len: max_answer_len = cur_max_answer_len
    return all_instances, max_answer_len

def collect_vocabs(all_instances):
    all_words = set()
    all_POSs = set()
    all_NERs = set()
    for (sent1, sent2, sent3) in all_instances:
        sentences = [sent1, sent2]
        if sent3 is not None: sentences.append(sent3)
        for sentence in sentences:
            all_words.update(re.split("\\s+", sentence.tokText))
            if sentence.POSs != None and sentence.POSs != []:
                all_POSs.update(re.split("\\s+", sentence.POSs))
            if sentence.NERs != None and sentence.NERs != []:
                all_NERs.update(re.split("\\s+", sentence.NERs))
    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_POSs, all_NERs)

class QADataStream(object):
    def __init__(self, all_questions, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None, options=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.options = options
        if batch_size ==-1: batch_size=options.batch_size
        # index tokens and filter the dataset
        instances = []
        for (sent1, sent2, sent3) in all_questions:# sent1 is the long passage or article
            if options.max_passage_len!=-1:
                if sent1.get_length()> options.max_passage_len: continue # remove very long passages
            if sent2.get_length() < 3: continue # filter out very short questions (len<3)
            sent1.convert2index(word_vocab, char_vocab, POS_vocab, NER_vocab, max_char_per_word=options.max_char_per_word)
            #if len(sent1.word_idx_seq) != len(sent1.POS_idx_seq):
            #    print '!!sent1', len(sent1.word_idx_seq), len(sent1.POS_idx_seq)
            sent2.convert2index(word_vocab, char_vocab, POS_vocab, NER_vocab, max_char_per_word=options.max_char_per_word)
            if sent3 is not None:
                sent3.convert2index(word_vocab, char_vocab, POS_vocab, NER_vocab, max_char_per_word=options.max_char_per_word)
                #if len(sent3.word_idx_seq) != len(sent3.POS_idx_seq):
                #    print '!!sent3', len(sent3.word_idx_seq), len(sent3.POS_idx_seq)

            instances.append((sent1, sent2, sent3))

        all_questions = instances
        instances = None

        # sort instances based on length
        if isSort:
            all_questions = sorted(all_questions, key=lambda question: (question[0].get_length(), question[1].get_length()))
        else:
            random.shuffle(all_questions)
            random.shuffle(all_questions)
        self.num_instances = len(all_questions)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_questions = []
            for i in xrange(batch_start, batch_end):
                cur_questions.append(all_questions[i])
            cur_batch = QAQuestionBatch(cur_questions, options, word_vocab=word_vocab, char_vocab=char_vocab,
                    POS_vocab=POS_vocab, NER_vocab=NER_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]

class QAQuestionBatch(object):
    def __init__(self, instances, options, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None):
        self.options = options

        self.instances = instances
        self.batch_size = len(instances)
        self.vocab = word_vocab

        self.passage_words = [instances[i][0].tokText.split() for i in range(self.batch_size)]

        self.has_sent3 = False
        if instances[0][2] is not None: self.has_sent3 = True

        # create length
        self.sent1_length = [] # [batch_size]
        self.sent2_length = [] # [batch_size]
        if self.has_sent3: self.sent3_length = [] # [batch_size]
        for (sent1, sent2, sent3) in instances:
            self.sent1_length.append(sent1.get_length())
            self.sent2_length.append(sent2.get_length())
            if self.has_sent3: self.sent3_length.append(sent3.get_length())
        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        self.sent2_length = np.array(self.sent2_length, dtype=np.int32)
        if self.has_sent3: self.sent3_length = np.array(self.sent3_length, dtype=np.int32)

        # create word representation
        start_id = word_vocab.getIndex('<s>')
        end_id = word_vocab.getIndex('</s>')
        if options.with_word:
            self.sent1_word = [] # [batch_size, sent1_len]
            self.sent2_word = [] # [batch_size, sent2_len]
            self.sent2_input_word = []
            if self.has_sent3: self.sent3_word = [] # [batch_size, sent3_len]
            for (sent1, sent2, sent3) in instances:
                self.sent1_word.append(sent1.word_idx_seq)
                self.sent2_word.append(sent2.word_idx_seq)
                self.sent2_input_word.append([start_id]+sent2.word_idx_seq[:-1])
                if self.has_sent3: self.sent3_word.append(sent3.word_idx_seq)
            self.sent1_word = padding_utils.pad_2d_vals_no_size(self.sent1_word)
            self.sent2_word = padding_utils.pad_2d_vals(self.sent2_word, len(self.sent2_word), options.max_answer_len)
            self.sent2_input_word = padding_utils.pad_2d_vals(self.sent2_input_word, len(self.sent2_input_word), options.max_answer_len)
            if self.has_sent3: self.sent3_word = padding_utils.pad_2d_vals_no_size(self.sent3_word)

            self.in_answer_words = self.sent2_word
            self.gen_input_words = self.sent2_input_word
            self.answer_lengths = self.sent2_length

        if options.with_char:
            self.sent1_char = [] # [batch_size, sent1_len]
            self.sent2_char = [] # [batch_size, sent2_len]
            if self.has_sent3: self.sent3_char = [] # [batch_size, sent3_len]
            for (sent1, sent2, sent3) in instances:
                self.sent1_char.append(sent1.char_idx_seq)
                self.sent2_char.append(sent2.char_idx_seq)
                if self.has_sent3: self.sent3_char.append(sent3.char_idx_seq)
            self.sent1_char = padding_utils.pad_3d_vals_no_size(self.sent1_char)
            self.sent2_char = padding_utils.pad_3d_vals_no_size(self.sent2_char)
            if self.has_sent3: self.sent3_char = padding_utils.pad_3d_vals_no_size(self.sent3_char)

        if options.with_POS:
            self.sent1_POS = [] # [batch_size, sent1_len]
            self.sent2_POS = [] # [batch_size, sent2_len]
            if self.has_sent3: self.sent3_POS = [] # [batch_size, sent3_len]
            for (sent1, sent2, sent3) in instances:
                self.sent1_POS.append(sent1.POS_idx_seq)
                self.sent2_POS.append(sent2.POS_idx_seq)
                if self.has_sent3: self.sent3_POS.append(sent3.POS_idx_seq)
            self.sent1_POS = padding_utils.pad_2d_vals_no_size(self.sent1_POS)
            self.sent2_POS = padding_utils.pad_2d_vals_no_size(self.sent2_POS)
            if self.has_sent3: self.sent3_POS = padding_utils.pad_2d_vals_no_size(self.sent3_POS)

        if options.with_NER:
            self.sent1_NER = [] # [batch_size, sent1_len]
            self.sent2_NER = [] # [batch_size, sent2_len]
            if self.has_sent3: self.sent3_NER = [] # [batch_size, sent3_len]
            for (sent1, sent2, sent3) in instances:
                self.sent1_NER.append(sent1.NER_idx_seq)
                self.sent2_NER.append(sent2.NER_idx_seq)
                if self.has_sent3: self.sent3_NER.append(sent3.NER_idx_seq)
            self.sent1_NER = padding_utils.pad_2d_vals_no_size(self.sent1_NER)
            self.sent2_NER = padding_utils.pad_2d_vals_no_size(self.sent2_NER)
            if self.has_sent3: self.sent3_NER = padding_utils.pad_2d_vals_no_size(self.sent3_NER)

        if options.with_phrase_projection:
            self.build_phrase_vocabs()
            if options.pretrain_with_max_matching and options.with_target_lattice:
                (_, prediction_lengths, generator_input_idx, generator_output_idx) = self.sample_a_partition(max_matching=True)
                self.in_answer_words = generator_output_idx
                self.gen_input_words = generator_input_idx
                self.answer_lengths = prediction_lengths


    def build_phrase_vocabs(self):
        self.phrase_vocabs = []
        word_size = self.vocab.vocab_size + 1

        self.phrase_starts = []
        self.phrase_ends = []
        self.phrase_idx = []
        self.phrase_lengths = []
        self.max_phrase_size = 0
        if self.options.with_target_lattice:
            self.target_lattices = []
        for (sent1, sent2, sent3) in self.instances:
            # collect all phrases
            if self.options.withSyntaxChunk:
                (cur_phrase_starts, cur_phrase_ends, _) = sent1.collect_all_syntax_chunks(self.options.max_chunk_len)
            else:
                (cur_phrase_starts, cur_phrase_ends) = sent1.collect_all_possible_chunks(self.options.max_chunk_len)

            # collect phrase vocab and map phrase into phrase_id
            cur_phrase2id = {}
            cur_phrase_idx = []
            for i in xrange(len(cur_phrase_starts)):
                cur_start = cur_phrase_starts[i]
                cur_end = cur_phrase_ends[i]
                cur_phrase = sent1.getTokChunk(cur_start, cur_end)
                cur_index = None
                if cur_start==cur_end:
                    cur_index = self.vocab.getIndex(cur_phrase)
                elif cur_phrase2id.has_key(cur_phrase):
                    cur_index = cur_phrase2id[cur_phrase]
                else:
                    cur_index = len(cur_phrase2id) + word_size
                    cur_phrase2id[cur_phrase] = cur_index
                cur_phrase_idx.append(cur_index)
            cur_phrase_vocab = phrase_lattice_utils.prefix_tree(cur_phrase2id)
            self.phrase_vocabs.append(cur_phrase_vocab)
            self.phrase_starts.append(cur_phrase_starts)
            self.phrase_ends.append(cur_phrase_ends)
            self.phrase_idx.append(cur_phrase_idx)
            self.phrase_lengths.append(len(cur_phrase_starts))
            cur_phrase_size = len(cur_phrase2id)
            if self.max_phrase_size<cur_phrase_size: self.max_phrase_size = cur_phrase_size

            if self.options.with_target_lattice:
                cur_lattice = phrase_lattice_utils.phrase_lattice(sent2.words, word_vocab=self.vocab, prefix_tree=cur_phrase_vocab)
                self.target_lattices.append(cur_lattice)

        self.phrase_starts = padding_utils.pad_2d_vals_no_size(self.phrase_starts) # [batch_size, phrase_size]
        self.phrase_ends = padding_utils.pad_2d_vals_no_size(self.phrase_ends) # [batch_size, phrase_size]
        self.phrase_idx = padding_utils.pad_2d_vals_no_size(self.phrase_idx) # [batch_size, phrase_size]
        self.phrase_lengths = np.array(self.phrase_lengths, dtype=np.int32) # [batch_size]

    def map_phrase_idx_to_text(self, samples):
        '''
        sample: [batch_size, length] of idx
        '''
        word_size = self.vocab.vocab_size + 1
        all_words = []
        all_word_idx = []
        for i in xrange(len(samples)):
#             cur_passage = self.instances[i][0]
            cur_sample = samples[i]
            if self.options.with_phrase_projection: cur_phrase_vocab = self.phrase_vocabs[i]
            cur_words = []
            cur_word_idx = []
            for idx in cur_sample:
                if idx<word_size:
                    cur_word = self.vocab.getWord(idx)
                elif not cur_phrase_vocab.has_phrase_id(idx): # if an OOV phrase is sampled, reset it to UNK
                    idx = self.vocab.vocab_size
                    cur_word = self.vocab.getWord(idx)
                else:
#                     if not cur_id2phrase.has_key(idx):
#                         print(cur_id2phrase)
#                         print(idx)
#                     cur_word = cur_id2phrase[idx]
                    cur_word = cur_phrase_vocab.get_phrase(idx)
#                     if not self.options.withTextChunk:
#                         items = re.split('-', cur_word)
#                         cur_word = cur_passage.getTokChunk(int(items[0]), int(items[1]))
                    idx = self.vocab.getIndex(re.split("\\s+", cur_word)[-1]) # take the last word of a phrase as the input word for decoding
                cur_words.append(cur_word)
                cur_word_idx.append(idx)
            all_words.append(cur_words)
            all_word_idx.append(cur_word_idx)
        return (all_words, all_word_idx) # [batch_size, length]

    def sample_a_partition(self, max_matching=False):
        word_size = self.vocab.vocab_size + 1
        sentences = []
        prediction_lengths = []
        generator_input_idx = []
        generator_output_idx = []
        for i, cur_lattice in enumerate(self.target_lattices):
            (cur_phrases, cur_phrase_ids) = cur_lattice.sample_a_partition(max_matching=max_matching)
            sentences.append(" ".join(cur_phrases))
            prediction_lengths.append(len(cur_phrases))
            generator_output_idx.append(cur_phrase_ids)
            cur_input_idx = [self.gen_input_words[i][0]]
            for cur_phrase, cur_phrase_id in zip(cur_phrases, cur_phrase_ids):
                if cur_phrase_id<word_size:
                    cur_word_id = cur_phrase_id
                elif not self.phrase_vocabs[i].has_phrase_id(cur_phrase_id): # if an OOV phrase is sampled, reset it to UNK
                    cur_word_id = self.vocab.vocab_size
                else:
                    cur_word_id = self.vocab.getIndex(re.split("\\s+", cur_phrase)[-1]) # take the last word of a phrase as the input word for decoding
                cur_input_idx.append(cur_word_id)
            generator_input_idx.append(cur_input_idx[:-1])

        generator_input_idx = padding_utils.pad_2d_vals(generator_input_idx, len(generator_input_idx), self.options.max_answer_len)
        generator_output_idx = padding_utils.pad_2d_vals(generator_output_idx, len(generator_output_idx), self.options.max_answer_len)
        return (sentences, prediction_lengths, generator_input_idx, generator_output_idx)



if __name__ == "__main__":
    ''' # collect vocab
    inpath = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/data/fof.tok"
    outpath = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/data/vocab.txt"
#     inpath = "/u/zhigwang/zhigwang1/sentence_generation/mscoco/data/fof.tok"
#     outpath = "/u/zhigwang/zhigwang1/sentence_generation/mscoco/data/vocab.txt"
    all_paths = read_text_file(inpath)
    all_instances = []
    for cur_path in all_paths:
        print('Loading instances from ' + cur_path)
        all_instances.extend(read_all_GenerationDatasets(cur_path, isLower=True)[0])
    print('Number of training samples: {}'.format(len(all_instances)))

    (all_words, all_chars, all_POSs, all_NERs) = collect_vocabs(all_instances)
    outfile = open(outpath, 'wt')
    for word in all_words:
        outfile.write(("%s\n" % word).encode('utf-8'))
    outfile.close()
    # '''

    '''
    inpath = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/data/train.json.0.tok"
    read_all_GenerationDatasets(inpath, isLower=True)
    '''

    inpath = "/u/zhigwang/zhigwang1/sentence_generation/cnn-dailymail/data/test.fof"
    (all_instances, max_answer_len)= read_generation_datasets_from_fof(inpath, isLower=True)
    print(max_answer_len)

    print('DONE!')
