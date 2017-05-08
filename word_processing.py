import numpy as np
import pickle
import myconf as cf

word_file = './data/word'
embedding_file = './data/word_embedding'
sentence_file = './data/sentence_list'
	
sentence_length = cf.sentence_length

PAD = 0 
EOS = 1

def get_sentences():
	idx2word, word2idx = get_wordListFromFile()
	
	sentence_list = list()
	decoder_sentence_list = list()
	
	with open(sentence_file, 'rb') as file:
		s_list = pickle.load(file)
		
	for _l in s_list:
		_l = _l[1]
		_l = _l.split(' ')
		
		t = list()
		decoder_t = list()
		
		for w in _l:
			w = w.strip()
			
			if w == '':
				continue
			
			if w in word2idx:
				t.append(word2idx[w])
				decoder_t.append(word2idx[w])
			else:
				t.append(word2idx['UNK'])
				decoder_t.append(word2idx['UNK'])
		
		decoder_t.append(EOS)
		
		if len(t) <= sentence_length:
			padding_size = sentence_length - len(t)
			padding_vector = [PAD for i in range(padding_size)]
			
			t.extend(padding_vector)
			decoder_t.extend(padding_vector)
			
			sentence_list.append(t)
			decoder_sentence_list.append(decoder_t)
			
	return sentence_list, decoder_sentence_list
	
def get_sentences_with_document_id():
	idx2word, word2idx = get_wordListFromFile()
	
	document_list = list()
	sentence_list = list()
	decoder_sentence_list = list()
	
	with open(sentence_file, 'rb') as file:
		s_list = pickle.load(file)
		
	for _l in s_list:
		dc = _l[0]
		_l = _l[1]
		_l = _l.split(' ')
		
		t = list()
		decoder_t = list()
		
		for w in _l:
			w = w.strip()
			
			if w == '':
				continue
			
			if w in word2idx:
				t.append(word2idx[w])
				decoder_t.append(word2idx[w])
			else:
				t.append(word2idx['UNK'])
				decoder_t.append(word2idx['UNK'])
		
		decoder_t.append(EOS)
		
		if len(t) <= sentence_length:
			padding_size = sentence_length - len(t)
			padding_vector = [PAD for i in range(padding_size)]
			
			t.extend(padding_vector)
			decoder_t.extend(padding_vector)
			
			sentence_list.append(t)
			decoder_sentence_list.append(decoder_t)
			document_list.append(dc)
			
	return sentence_list, decoder_sentence_list	, document_list

def get_sentences_with_document_id_and_eos():
	idx2word, word2idx = get_wordListFromFile()
	
	document_list = list()
	sentence_list = list()
	decoder_sentence_list = list()
	
	with open(sentence_file, 'rb') as file:
		s_list = pickle.load(file)
		
	for _l in s_list:
		dc = _l[0]
		_l = _l[1]
		_l = _l.split(' ')
		
		t = list()
		decoder_t = list()
		
		for w in _l:
			w = w.strip()
			
			if w == '':
				continue
			
			if w in word2idx:
				t.append(word2idx[w])
				decoder_t.append(word2idx[w])
			else:
				t.append(word2idx['UNK'])
				decoder_t.append(word2idx['UNK'])
		
		decoder_t.append(EOS)
		
		if len(t) <= sentence_length:
			padding_size = sentence_length - len(t)
			padding_vector = [PAD for i in range(padding_size)]
			
			t.extend(padding_vector)
			t = [EOS] + t
			decoder_t.extend(padding_vector)
			
			
			sentence_list.append(t)
			decoder_sentence_list.append(decoder_t)
			document_list.append(dc)
			
	return sentence_list, decoder_sentence_list	, document_list
	
def get_wordEmbeddings():
	with open(embedding_file, 'rb') as file:
		embedding = pickle.load(file)
		
	return embedding
	

def get_wordListFromFile():
	with open(word_file, 'rb') as file:
		idx2word = pickle.load(file)
			
	word2idx = {w:i for i,w in enumerate(idx2word)}
		
	return idx2word, word2idx

if __name__ == '__main__':
	'''
	sentences = get_sentences()
	print(sentences[0])
	print(sentences[1])
	print(sentences[2])
	print(len(sentences))
	'''
