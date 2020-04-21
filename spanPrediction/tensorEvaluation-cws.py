# -*- coding: utf-8 -*-
import numpy as np
import pickle
import codecs
import os
from collections import Counter
from copy import deepcopy
import re
import argparse
from ner_overall_f1 import evaluate,evaluate_chunk_level,evaluate_each_class,evaluate_each_class_listone
import math
import scipy.stats as statss

import random
import matplotlib
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_word(seq_chars,seq_tags):
	e_idx = 'E'
	s_idx = 'S'
	tag_seqs_sent = []
	word_seqs_sent = []
	tag_seqs = []
	word_seqs = []
	charsss = []
	for seq_char, seq_tag in zip(seq_chars, seq_tags):
		tag_seq = []
		word_seq = []
		start = 0
		for i, (char, tok) in enumerate(zip(seq_char, seq_tag)):
			char = convert_charLen2one(char)
			charsss.append(char)
			# print('tok',tok)

			if tok == e_idx or tok == s_idx:
				# print('start',start)
				# print('i+1',i+1)
				word = seq_char[start: i + 1]
				tag = seq_tag[start: i + 1]
				start = i + 1
				# print('w
				# ord',word)
				# print('tag', tag)
				word_seq.append(''.join(word))
				tag_seq.append(''.join(tag))

		if seq_char[-1] not in [e_idx,s_idx]:
			if start!=len(seq_char):
				word = seq_char[start:]
				tag = seq_tag[start:]
				word_seq.append(''.join(word))
				tag_seq.append(''.join(tag))


		# print('tag_seq',tag_seq)
		# print('word_seq', word_seq)
		tag_seqs_sent.append(tag_seq)
		word_seqs_sent.append(word_seq)
		tag_seqs += tag_seq
		word_seqs += word_seq

	count =0
	for word,tag in zip(word_seqs,tag_seqs):
		for char,t in zip(word,tag):
			# print(count,charsss[count],char,t,word,tag)

			count+=1
			if count == len(charsss):
				break
		if count== len(charsss):
			break
	# print('total char',count)



	return word_seqs,tag_seqs,word_seqs_sent,tag_seqs_sent

def get_chunks(word_seqs):
	chunks = []
	chunk_start = 0
	for word in word_seqs:
		chunk = (word,chunk_start,chunk_start+len(word) )
		# print("chunk: ", chunk)
		chunks.append(chunk)
		chunk_start = chunk_start+len(word)
	return chunks

def convert_charLen2one(char):
	# char =''
	if char == '<NUM>':
		char = 'N'
	elif char == '<ENG>':
		char = 'E'
	if len(char) != 1:
		# print(char)
		# print('the length of char is not equal to 1 ...')
		char = 'U'
	else:
		char = char
	return char

def get_words_num(word_sequences):
	return sum(len(word_seq) for word_seq in word_sequences)

def read_data(corpus_type,fn):
	mode = '#'
	column_no = -1

	src_data = []
	data = []
	label = []
	data_sent =[]
	label_sent =[]

	src_data_sentence = []
	data_sentence = []
	label_sentence = []

	with codecs.open(fn, 'r', 'utf-8') as f:
		lines = f.readlines()
	for line in lines:
		# print('line',line)
		# for k in range(len(lines)):
		#     line = str(line, 'utf-8')
		line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
		if len(line_t) < 3:
			if len(data_sentence) == 0:
				continue
			src_data.append(src_data_sentence)
			data.append(data_sentence)
			label.append(label_sentence)
			src_data_sentence = []
			data_sentence = []
			label_sentence = []
			continue
		src_word =line_t[0]
		word = convert_charLen2one(line_t[1])
		src_data_sentence.append(src_word)
		data_sentence.append(word)
		data_sent.append(word)
		label_sent.append(line_t[2].split('_')[0])
		label_sentence += [line_t[2].split('_')[0]]
	print('Loading from %s: %d samples, %d words.' % (fn, len(data), get_words_num(data)))


	return  data_sent,label_sent,data, label

def read_data_test2(corpus_type,fn_test_result,fn_test):
	words_sent = []
	truetags_sent =[]
	predtags_sent=[]

	words_all = []
	truetags_all = []
	predtags_all = []

	words = []
	truetags = []
	predtags = []

	word1=[]
	truetag1 =[]
	predtag1 =[]
	count =-1
	with codecs.open(fn_test_result, 'r', 'utf-8') as f:
		lines = f.readlines()
	for line in lines:

		line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').strip().split()
		if len(line_t) ==0:
			count += 1
			# if count > 5:
			# 	break
			words_sent.append(word1)
			truetags_sent.append(truetag1)
			predtags_sent.append(predtag1)
			words += word1
			truetags += truetag1
			predtags += predtag1

			word1 = []
			truetag1 = []
			predtag1 = []
		else:
			if len(line_t)==2:
				w1='U'
				t1 =line_t[0]
				t2 =line_t[1]
			else:
				w1 = convert_charLen2one(line_t[0])
				t1 =line_t[1]
				t2 =line_t[2]
			word1.append(w1)
			truetag1.append(t1)
			predtag1.append(t2)
	print('Loading from %s: %d samples, %d words.' % (fn_test_result, len(words), get_words_num(words)))

	for ws,ts,ps in zip(words_sent,truetags_sent,predtags_sent):
		words_all += ws
		truetags_all += ts
		predtags_all += ps



	words_final =[]
	words_sent_final =[]
	count_testwordNotEqual2Original =0
	count=0
	data_test, label_test, data_sent_test, label_sent_test = read_data(corpus_type, fn_test)
	print('!!!!!!!!!!!!!!!!!!!!!!!!! read_test')
	print('data_sent_test',len(data_sent_test) )
	print('words_sent',len(words_sent))
	print('truetags_sent',len(truetags_sent))
	print('predtags_sent',len(predtags_sent))
	for wt_s,ws,ts,ps in zip(data_sent_test,words_sent,truetags_sent,predtags_sent):
		new_ws =[]
		# print(len(wt_s),len(ws),len(ts),len(ps))
		for wt,w,t,p in zip(wt_s,ws,ts,ps):
			count+=1
			if wt !=w:
				w= wt
				count_testwordNotEqual2Original+=1
			words_final.append(w)
			new_ws.append(w)
		words_sent_final.append(new_ws)

	print('************************** new test word reading *******************************')
	print('count',count)
	print('count_testwordNotEqual2Original',count_testwordNotEqual2Original)
	print('len(words_final)',len(words_final))
	print('len(truetags)',len(truetags))
	print('len(predtags)',len(predtags))

	print('len(words_sent_final)',len(words_sent_final))
	print('len(truetags_sent)',len(truetags_sent))
	print('len(predtags_sent)',len(predtags_sent))
	print('************************** end new test word reading *******************************')

	# for i,(ws,ts,ps) in enumerate(zip(words_sent, truetags_sent, predtags_sent)):
	# 	if i<=5:
	# 		print("words: ",words_sent[i])
	# 		print("truetags_sent: ",truetags_sent[i])
	# 		print("predtags_sent: ",predtags_sent[i])


	# return  words_final,truetags,predtags,words_sent_final,truetags_sent,predtags_sent

	# count_lastLabel_SE = 0
	# for i in range(len(predtags_sent)):
	# 	if predtags_sent[i][-1] not in ['E','S']:
	# 		count_lastLabel_SE += 1
	# 		print('predtags_sent[i][-1]',predtags_sent[i][-1])
	# print('*********************************')
	# print('fn_test_result: ', fn_test_result)
	# print('count_lastLabel_SE: ', count_lastLabel_SE)
	# print()

	return words_all, truetags_all, predtags_all, words_sent, truetags_sent, predtags_sent


def span_entity_amb_fre(word_seqs_train_sent,tag_seqs_train_sent, fnwrite_GoldenWordHard):
	if os.path.exists(fnwrite_GoldenWordHard):
		print('load the hard dictionary of cws word in test set...')
		fread = open(fnwrite_GoldenWordHard, 'rb')
		Hard_cwsWord_inTrain, Freq_cwsWord_inTrain_normalize = pickle.load(fread)
		return Hard_cwsWord_inTrain, Freq_cwsWord_inTrain_normalize
	else:
		occur_error_words = []
		Hard_cwsWord_inTrain = dict()
		Freq_cwsWord_inTrain = dict()
		word_seqs_train = []
		tag_seqs_train = []
		for w_s, t_s in zip(word_seqs_train_sent, tag_seqs_train_sent):
			for w, t in zip(w_s, t_s):
				word_seqs_train += w
				tag_seqs_train += t

		word_seqs_train_str = ''.join(word_seqs_train)
		tag_seqs_train_str = ''.join(tag_seqs_train)
		print('len(word_seqs_train)', len(word_seqs_train))
		print('len(tag_seqs_train)', len(tag_seqs_train))
		print('len(word_seqs_train_str)', len(word_seqs_train_str))
		print('len(tag_seqs_train_str)', len(tag_seqs_train_str))

		# word_seqs_train=['螃蟹']
		count_word_seen = 0
		count_word_notseen = 0
		count = 0
		for word_train_sent in word_seqs_train_sent:
			for word_train in word_train_sent:
				count += 1
				print('count', count)
				if word_train in Hard_cwsWord_inTrain:
					continue
				else:
					Hard_cwsWord_inTrain[word_train] = dict()
				word_train1 = deepcopy(word_train)
				# word_train = '(螃蟹'
				if '(' in word_train1 or ')' in word_train1 or '（' in word_train1 or '）' in word_train1:
					word_train1 = word_train1.replace('（', '\（')
					word_train1 = word_train1.replace('）', '\）')
					word_train1 = word_train1.replace('(', '\(')
					word_train1 = word_train1.replace(')', '\)')
				if '*' in word_train1:
					word_train1 = word_train1.replace('*', '\*')
				if '+' in word_train1:
					word_train1 = word_train1.replace('+', '\+')

				# word_seqs_train_str = '我喜欢吃（螃蟹，（螃蟹特别好吃，大家都很喜欢吃螃蟹。'
				# tag_seqs_train_str ='SBESBMESBMEBEBESBESSBEBME'
				if word_train1.strip() == '':
					word_train1 = ','
				cwsword_str_index = []
				try:
					cwsword_str_index = [m.start() for m in re.finditer(word_train1, word_seqs_train_str)]
				except:
					try:
						cwsword_str_index = [m.start() for m in re.finditer(word_train, word_seqs_train_str)]
					except:
						print("Unexpected error:")
						print('word_train1', word_train1)
						occur_error_words.append(word_train1)

				word_len = len(word_train)
				# print('cwsword_str_index:',cwsword_str_index)
				# print('word_train', word_train)
				# print('word_train1:', word_train1)

				if len(cwsword_str_index) > 0:
					count_word_seen += 1
					# print('find word...',word_train1)
					label_list = []
					# convert the string index into list index...
					for str_idx in cwsword_str_index:
						cwsword_idx = len(word_seqs_train_str[0:str_idx])
						label_list_cand = tag_seqs_train_str[cwsword_idx:cwsword_idx + word_len]
						label_cand_str = ''.join(label_list_cand)
						label_list.append(label_cand_str)
					# print('label_list:',label_list)
					Freq_cwsWord_inTrain[word_train] = len(label_list)
					label_norep = list(set(label_list))
					for lab_norep in label_norep:
						hard = float('%.3f' % (float(label_list.count(lab_norep)) / len(label_list)))

						Hard_cwsWord_inTrain[word_train][lab_norep] = hard
				# Hard_cwsWord_inTrain[word_train][lab_norep] = str(label_list.count(lab_norep))+'_'+str(len(label_list))
				else:
					count_word_notseen += 1
			# print('not find word...',word_train1)

		print('string finding in train set, num errors:', len(occur_error_words))
		print('occur_error_word is:', occur_error_words)
		print('count_word_seen:', count_word_seen)
		print('count_word_notseen:', count_word_notseen)
		### for the char frequency normalization...
		sorted_Freq_cwsWord_inTrain = sorted(Freq_cwsWord_inTrain.items(), key=lambda item: item[1], reverse=True)
		# for char,freq in sorted_Freq_cwsWord_inTrain:
		# 	print('char',char)
		# 	print('freq',freq)

		max_word_freq = sorted_Freq_cwsWord_inTrain[1][1]
		print('max_word_freq', max_word_freq)
		Freq_cwsWord_inTrain_normalize = {}
		count_word_bigerThan_maxFreq = 0
		for word, freq in Freq_cwsWord_inTrain.items():
			if freq <= max_word_freq:
				Freq_cwsWord_inTrain_normalize[word] = '%.3f' % (float(freq) / max_word_freq)
			else:
				Freq_cwsWord_inTrain_normalize[word] = '1.0'
				count_word_bigerThan_maxFreq += 1
		print('count_word_bigerThan_maxFreq', count_word_bigerThan_maxFreq)

		fwrite = open(fnwrite_GoldenWordHard, 'wb')
		pickle.dump([Hard_cwsWord_inTrain, Freq_cwsWord_inTrain_normalize], fwrite)
		fwrite.close()

		# in the true tags, the num of no repeated entity is 2529
		# the num of no repeated entity, including the true and predicted, is 2633.
		# print('Hard_cwsWord_inTrain:',Hard_cwsWord_inTrain)
		# print('len, Hard_cwsWord_inTrain',len(Hard_cwsWord_inTrain))

		print('Freq_cwsChar_inTrain_normalize[李]', Freq_cwsWord_inTrain_normalize['李'])
		print('Hard_cwsChar_inTrain[李]', Hard_cwsWord_inTrain['李'])
		# return Hard_cwsWord_inTrain,Freq_cwsWord_inTrain_normalize
		return Hard_cwsWord_inTrain, Freq_cwsWord_inTrain_normalize

def span_token_amb_fre(train_seqchar, train_seqtag, fnwrite_GoldenCharHard):
	if os.path.exists(fnwrite_GoldenCharHard):
		print('load the hard dictionary of entity token in test set...')
		fread = open(fnwrite_GoldenCharHard, 'rb')
		Hard_cwsChar_inTrain, Freq_cwsChar_inTrain_normalize = pickle.load(fread)
		return Hard_cwsChar_inTrain, Freq_cwsChar_inTrain_normalize
	else:
		# build the word2tags dictionary.
		char2tags_inTrain = dict()
		for i in range(len(train_seqchar)):
			tag = train_seqtag[i]
			if train_seqchar[i] in char2tags_inTrain:
				char2tags_inTrain[train_seqchar[i]].append(tag)
			else:
				char2tags_inTrain[train_seqchar[i]] = [tag]
		print('the char list in train set is:', len(char2tags_inTrain))

		Hard_cwsChar_inTrain = dict()
		Freq_cwsChar_inTrain = dict()
		for char, labels_list in char2tags_inTrain.items():
			Hard_cwsChar_inTrain[char] = dict()
			Freq_cwsChar_inTrain[char] = len(labels_list)  # compute the frequency of char in the train set...
			labels_norep = list(set(labels_list))
			for lab_norep in labels_norep:
				hard = float('%.3f' % (float(labels_list.count(lab_norep)) / len(labels_list)))
				Hard_cwsChar_inTrain[char][lab_norep] = hard

		### for the char frequency normalization...
		sorted_Freq_cwsChar_inTrain = sorted(Freq_cwsChar_inTrain.items(), key=lambda item: item[1], reverse=True)
		# for char,freq in sorted_Freq_cwsChar_inTrain:
		# 	print('char',char)
		# 	print('freq',freq)

		max_char_freq = sorted_Freq_cwsChar_inTrain[1][1]
		print('max_char_freq', max_char_freq)
		Freq_cwsChar_inTrain_normalize = {}
		count_char_bigerThan_maxFreq = 0
		for char, freq in Freq_cwsChar_inTrain.items():
			if freq <= max_char_freq:
				Freq_cwsChar_inTrain_normalize[char] = '%.3f' % (float(freq) / max_char_freq)
			else:
				Freq_cwsChar_inTrain_normalize[char] = '1.0'
				count_char_bigerThan_maxFreq += 1
		print('count_char_bigerThan_maxFreq', count_char_bigerThan_maxFreq)

		fwrite = open(fnwrite_GoldenCharHard, 'wb')
		pickle.dump([Hard_cwsChar_inTrain, Freq_cwsChar_inTrain_normalize], fwrite)
		fwrite.close()

		# print('len(Hard_entityToken_inTrain)',len(Hard_entityToken_inTrain))

		# the num of no repeated entity token in test set is 8,548
		# print('Hard_cwsChar_inTrain',Hard_cwsChar_inTrain)

		print('len(Hard_cwsChar_inTrain)', len(Hard_cwsChar_inTrain))
		print('len(Freq_cwsChar_inTrain_normalize)', len(Freq_cwsChar_inTrain_normalize))

		print('Freq_cwsChar_inTrain_normalize[简]', Freq_cwsChar_inTrain_normalize['简'])
		print('Hard_cwsChar_inTrain[简]', Hard_cwsChar_inTrain['简'])
		return Hard_cwsChar_inTrain, Freq_cwsChar_inTrain_normalize


def span_sent_oovDen_length(train_vocab,test_word_sequences_sent,fnwrite_sentLevelDensity):
	if os.path.exists(fnwrite_sentLevelDensity):
		print('load the max_wordLength, oov_density, sent_len...')
		fread = open(fnwrite_sentLevelDensity, 'rb')
		max_wordLength, oov_density, sent_len = pickle.load(fread)
		return max_wordLength, oov_density, sent_len
	else:
		# loading the train_vocab...
		f_train_vocab = open(train_vocab, 'rb')
		word_vocab, emb = pickle.load(f_train_vocab)

		# compute the max length of word.
		word_lengths = []
		sent_str_list = []
		for test_sent in test_word_sequences_sent:
			sent_str_list.append(''.join(test_sent))
			for word in test_sent:
				word_lengths.append(len(word))
		max_wordLength = np.max(word_lengths)
		# print('max_wordLength',max_wordLength)
		# compute the cws char density and oov density ...
		oov_density = []
		sent_len = []
		for sent_char in sent_str_list:
			num_oov = 0
			sent_len.append(len(sent_char))
			for char in sent_char:
				if char not in word_vocab:
					num_oov += 1
			oov_density.append(float(num_oov) / len(sent_char))

		# print('sent_str_list[10]',sent_str_list[10])
		# print('sent_len[10]', sent_len[10])

		fwrite = open(fnwrite_sentLevelDensity, 'wb')
		pickle.dump([max_wordLength, oov_density, sent_len], fwrite)
		fwrite.close()

		return max_wordLength,oov_density, sent_len


def property_block(property_range_dic):
	n_piles = len(property_range_dic)
	low_bounds = []
	up_bounds = []
	p_ranges = []
	total_count_entity = 0
	for p_range, count_entity in property_range_dic.items():
		low_bounds.append(float(p_range.split(',')[0][1:]) )
		up_bounds.append(float(p_range.split(',')[1][:-1]) )
		p_ranges.append(p_range)
		total_count_entity+=count_entity
	print('low_bounds',low_bounds)
	print('up_bounds', up_bounds)
	print('p_ranges',p_ranges)
	print('total_count_entity',total_count_entity)
	return low_bounds, up_bounds,p_ranges



def intervalTransformer(inter_list):
	dict_old2new = {}
	last = 0
	for ind, interval in enumerate(inter_list):
		if ind == 0:
			last = interval[0]
		if len(interval) == 1:
			#new_inter_list.append(interval)
			dict_old2new[interval] = interval
			last = interval[0]
		else:
			#new_inter_list.append((last, interval[1]))
			dict_old2new[interval] = (last, interval[1])
			last = interval[1]
	return dict_old2new







def getAttrValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
						preCompute_ambSpan,
						preCompute_ambToken,
						preCompute_freqSpan,
						preCompute_freqToken,
						preCompute_odensity,
						preCompute_slength):

	dict_entity2amb = {}
	dict_token2amb = {}
	dict_entity2fre = {}
	dict_token2fre = {}
	dict_entity2elen = {}
	dict_entity2slen={}
	dict_entity2eden={}
	dict_entity2oden={}
	dict_entity2tag = {}


	#test_word_sequences, test_trueTag_sequences
	dict_pos2sid = getPos2SentId(test_word_sequences_sent)
	all_chunks = get_chunks(test_word_sequences)
	# print('last two word: ',test_word_sequences[-2])
	# print('total char num: ', len(''.join(test_word_sequences)))
	print("len(all_chunks)",len(all_chunks))
	print("len(test_trueTag_sequences)",len(test_trueTag_sequences))
	for k,span_info in enumerate(all_chunks):
		span_type = test_trueTag_sequences[k]

		idx_start = span_info[1]
		idx_end = span_info[2]
		# span_cnt = ' '.join(test_word_sequences[idx_start:idx_end]).lower()
		span_cnt = span_info[0].lower()
		span_pos = str(idx_start) + "_" + str(idx_end) + "_" + span_type

		span_length = idx_end - idx_start

		span_token_list = list(span_cnt)
		span_token_pos_list = [ str(pos) + "_" + span_type for pos in range(idx_start, idx_end)]


		span_sentid = dict_pos2sid[idx_start] # not implement


		# Attribute 1:  Span-level Ambiguity
		#print("Attribute 1:  Span-level Ambiguity")
		span_amb_value = 0.0
		if span_cnt in preCompute_ambSpan:
			# print('preCompute_ambSpan[span_cnt]: ', preCompute_ambSpan[span_cnt])
			if span_type in preCompute_ambSpan[span_cnt]:
				span_amb_value = preCompute_ambSpan[span_cnt][span_type]
		dict_entity2amb[span_pos] = span_amb_value


		# Attribute 2:  Token-level Ambiguity
		#print("Attribute 2:  Token-level Ambiguity")
		token_amb_value = 0.0
		for i,(token, token_pos)  in enumerate(zip(span_token_list, span_token_pos_list)):
			if token.lower() in preCompute_ambToken:
				# print('preCompute_ambToken[token.lower()]: ',preCompute_ambToken[token.lower()])
				token_type = token_pos.split("_")[1]
				if span_type in preCompute_ambToken[token.lower()]:
					token_amb_value = preCompute_ambToken[token.lower()][token_type[i]]
			dict_token2amb[token_pos] = token_amb_value


		#print("Attribute 3:  Token-level Ambiguity")
		# Attribute 3:  Span-level Frequency
		span_fre_value = 0.0
		if span_cnt in preCompute_freqSpan:
			span_fre_value = preCompute_freqSpan[span_cnt]
		dict_entity2fre[span_pos] = span_fre_value


		# Attribute 4:  Token-level Frequency
		#print("Attribute 4:  Token-level Ambiguity")
		token_fre_value = 0.0
		for token, token_pos  in zip(span_token_list, span_token_pos_list):
			if token.lower() in preCompute_freqToken:
				token_fre_value = preCompute_freqToken[token.lower()]
			dict_token2fre[token_pos] = token_fre_value



		# Attribute 5:  Entity Length
		dict_entity2elen[span_pos] = span_length


		# Attribute 6:  Sentence Length
		dict_entity2slen[span_pos] = float(preCompute_slength[span_sentid])

		# # Attribute 7:  Entity Density
		# dict_entity2eden[span_pos] = preCompute_edensity[span_sentid]

		# Attribute 8:  OOV Density
		dict_entity2oden[span_pos] = float(preCompute_odensity[span_sentid])


		# Attribute 9: Tag
		dict_entity2tag[span_pos] = span_type




	return  dict_entity2amb, dict_token2amb, dict_entity2fre, dict_token2fre, dict_entity2elen, dict_entity2slen, dict_entity2oden, dict_entity2tag







def bucketAttribute_SpecifiedBucketValue(dict_span2attVal, n_buckets, hardcoded_bucket_values):
	################       Bucketing different Attributes

	# 	hardcoded_bucket_values = [set([float(0), float(1)])]
		n_spans = len(dict_span2attVal)
		dict_attVal2span = reverseDict(dict_span2attVal)
		#print(dict_attVal2span)
		#print("---------debug--dict_attVal2span")
		dict_attVal2span = sortDict(dict_attVal2span)
		dict_bucket2span = {}


		#print("n_spans:\t" + str(n_spans))

		for backet_value in hardcoded_bucket_values:
			if backet_value in dict_attVal2span.keys():
				#print("------------work!!!!---------")
				#print(backet_value)
				dict_bucket2span[(backet_value,)] = dict_attVal2span[backet_value]
				n_spans -= len(dict_attVal2span[backet_value])
				n_buckets -= 1
		#print("-------hardcoded_bucket-")
		#print(hardcoded_bucket_values)
		#print("-------dict_attVal2span.items()-")
		#print(dict_attVal2span.keys())
		#exit()

		avg_entity = n_spans * 1.0 / n_buckets
		n_tmp = 0
		entity_list = []
		val_list = []

		#
		#print("-----avg_entity----------")
		#print(avg_entity)


		for attval, entity in dict_attVal2span.items():
			if attval in hardcoded_bucket_values:
				continue

			val_list.append(attval)
			entity_list += entity
			n_tmp += len(entity)

			if n_tmp > avg_entity:
				if len(val_list) >=2:
					key_bucket = (val_list[0], val_list[-1])
					dict_bucket2span[key_bucket] = entity_list
				else:
					dict_bucket2span[(val_list[0],)] = entity_list
				entity_list = []
				n_tmp = 0
				val_list = []
		if n_tmp != 0:
			if len(val_list) >=2:
				key_bucket = (val_list[0], val_list[-1])
				dict_bucket2span[key_bucket] = entity_list
			else:
				dict_bucket2span[(val_list[0],)] = entity_list
		#
		#
		#
		#print("---------debug--dict_bucket2span.keys()")
		#print(dict_bucket2span.keys())

		return dict_bucket2span





def bucketAttribute_DiscreteValue(dict_span2attVal = None, n_entities = 1, n_buckets = 100000000):
	################          Bucketing different Attributes

	# 	hardcoded_bucket_values = [set([float(0), float(1)])]
	n_spans = len(dict_span2attVal)
	dict_bucket2span = {}

	dict_attVal2span = reverseDict_discrete(dict_span2attVal)
	dict_attVal2span = sortDict(dict_attVal2span, flag = "value")


	avg_entity = n_spans * 1.0 / n_buckets
	n_tmp = 0
	entity_list = []
	val_list = []



	n_total = 1
	for attval, entity in dict_attVal2span.items():
		if len(entity) < n_entities or n_total > n_buckets:
			break
		dict_bucket2span[(attval,)] = entity
		n_total += 1

	return dict_bucket2span



	for attval, entity in dict_attVal2span.items():
		if len(entity) < n_entities:
			continue
		dict_bucket2span[(attval,)] = entity

	return dict_bucket2span





def sortDict(dict_obj, flag = "key"):
	sorted_dict_obj  = []
	if flag == "key":
		sorted_dict_obj = sorted(dict_obj.items(), key=lambda item:item[0])
	elif flag == "value":
		#dict_bucket2span_
		sorted_dict_obj = sorted(dict_obj.items(), key=lambda item:len(item[1]), reverse = True)
	return dict(sorted_dict_obj)

def reverseDict(dict_a2b):
	dict_b2a = {}
	for k, v in dict_a2b.items():
		v = float(v)
		if v not in dict_b2a.keys():
			dict_b2a[float(v)] = [k]
		else:
			dict_b2a[float(v)].append(k)


	return dict_b2a

def reverseDict_discrete(dict_a2b):
	dict_b2a = {}
	for k, v in dict_a2b.items():
		if v not in dict_b2a.keys():
			dict_b2a[v] = [k]
		else:
			dict_b2a[v].append(k)


	return dict_b2a


def findKey(dict_obj, x):
	for k, v in dict_obj.items():
			if len(k) == 1:
				if x == k[0]:
					return k
			elif len(k) ==2 and x > k[0] and x <= k[1]:
					return k





def bucketAttribute_SpecifiedBucketInterval(dict_span2attVal, intervals):
	################       Bucketing different Attributes

	#hardcoded_bucket_values = [set([float(0), float(1)])]

	#intervals = [0, (0,0.5], (0.5,0.9], (0.99,1]]

	dict_bucket2span = {}
	n_spans = len(dict_span2attVal)



	if type(list(intervals)[0][0]) == type("string"):  # discrete value, such as entity tags
		dict_attVal2span = reverseDict_discrete(dict_span2attVal)
		dict_attVal2span = sortDict(dict_attVal2span, flag = "value")
		for attval, entity in dict_attVal2span.items():
			attval_tuple = (attval,)
			if attval_tuple in intervals:
				if attval_tuple not in dict_bucket2span.keys():
				    dict_bucket2span[attval_tuple] = entity
				else:
				    dict_bucket2span[attval_tuple] += entity

		for val in intervals:
			if val not in dict_bucket2span.keys():
				dict_bucket2span[val] = []
		# print("dict_bucket2span: ",dict_bucket2span)
	else:
		dict_attVal2span = reverseDict(dict_span2attVal)
		dict_attVal2span = sortDict(dict_attVal2span)
		for v in intervals:
			if len(v) == 1:
				dict_bucket2span[v] = []
			else:
				dict_bucket2span[v] = []


		for attval, entity in dict_attVal2span.items():
					res_key = findKey(dict_bucket2span, attval)
					#print("res-key:\t"+ str(res_key))
					if res_key == None:
						continue
					dict_bucket2span[res_key] += entity

	return dict_bucket2span


def bucketAttribute_SpecifiedBucketInterval_DiscreteValue(dict_span2attVal, bucket_name_list):


	dict_bucket2span = {}

	n_spans = len(dict_span2attVal)
	dict_attVal2span = reverseDict_discrete(dict_span2attVal)
	dict_attVal2span = sortDict(dict_attVal2span)



	for attval, entity in dict_attVal2span.items():
		attval_tuple = (attval,)
		if attval_tuple in bucket_name_list:
					if attval_tuple not in dict_bucket2span.keys():
							dict_bucket2span[attval_tuple] = entity
					else:
							dict_bucket2span[attval_tuple] += entity

	return dict_bucket2span



################       Calculate Bucket-wise F1 Score:

def getBucketF1(dict_bucket2span, dict_bucket2span_pred):

	dict_bucket2f1 = {}
	for bucket_interval, spans_true in dict_bucket2span.items():
		spans_pred = []
		if bucket_interval not in dict_bucket2span_pred.keys():
			#print(bucket_interval)
			raise ValueError("Predict Label Bucketing Errors")
		else:
			spans_pred = dict_bucket2span_pred[bucket_interval]
		f1, p, r = evaluate_wordchunk_level(spans_pred, spans_true)
		dict_bucket2f1[bucket_interval] = [f1, len(spans_true)]
	return sortDict(dict_bucket2f1)



def getPos2SentId(test_word_sequences_sent):
	dict_pos2sid = {}
	pos = 0
	for sid, sent in enumerate(test_word_sequences_sent):
		sent1 = ''.join(sent)
		for i in range(len(sent1)):
			dict_pos2sid[pos] = sid
			pos += 1
	return dict_pos2sid


def new_metric(corpus_type, column_no, mode, pos_column, train_vocab, n_buckets, dict_hardcoded_mode,
			   fn_train, fn_test_results, fn_test,
			   fnwrite_GoldenWordHard,
			   fnwrite_GoldenCharHard,
			   fnwrite_sentLevelDensity,
			   ):
	train_seqchar, train_seqtag, train_seqchar_sent, train_seqtag_sent = read_data(corpus_type, fn_train)
	test_seqchar, test_truetags, test_predtags, test_seqchar_sent, test_truetags_sent, test_predtags_sent = read_data_test2(
		corpus_type, fn_test_results, fn_test)

	print('len(test_truetags)', len(test_truetags))
	print('len(test_predtags)', len(test_predtags))
	print('len(test_seqchar)', len(test_seqchar))
	test_truetags_sent_1 = []
	for i in range(len(test_truetags_sent)):
		test_truetags_sent_1+=test_truetags_sent[i]

	test_predtags_sent_1 = []
	for i in range(len(test_predtags_sent)):
		test_predtags_sent_1 +=test_predtags_sent[i]


	print("len(test_truetags_sent_1): ",len(test_truetags_sent_1))
	print("len(test_predtags_sent_1): ", len(test_predtags_sent_1))

	print('-----------------------get word chunk-----------------------')
	word_seqs_train, tag_seqs_train, word_seqs_train_sent, tag_seqs_train_sent = get_word(train_seqchar_sent,
																						  train_seqtag_sent)
	word_true_test, truetag_test, word_true_test_sent, truetag_test_sent = get_word(test_seqchar_sent,
																					test_truetags_sent)

	word_pred_test, predtag_test, word_pred_test_sent, predtag_test_sent = get_word(test_seqchar_sent,
																					test_predtags_sent)
	print('####################################')
	print(len(''.join(word_true_test)))
	print(len(''.join(word_pred_test)))
	print('##################')

	# for i in range(10):
	# 	print(len(word_true_test_sent[i]), len(word_pred_test_sent[i]))
	# 	print(word_true_test_sent[i])
	# 	print(word_pred_test_sent[i])
	# 	print()


	# # Pre-computing
	# preCompute_ambSpan = span_entity_amb(train_seqchar, train_seqtag, fnwrite_GoldenEntityHard)
	# preCompute_ambToken = span_token_amb(train_seqchar, train_seqtag, fnwrite_TokenHard)
	# preCompute_freqSpan = span_entity_fre(train_seqchar, train_seqtag,fwrite_SpanFreq_inTrain)
	# preCompute_freqToken = span_token_fre(train_seqchar, train_seqtag, fwrite_TokenFreq_inTrain)

	preCompute_ambSpan, preCompute_freqSpan = span_entity_amb_fre(word_seqs_train_sent, tag_seqs_train_sent,
																			fnwrite_GoldenWordHard)
	preCompute_ambToken, preCompute_freqToken = span_token_amb_fre(train_seqchar, train_seqtag,
																			fnwrite_GoldenCharHard)
	# print('finish loading the four pkl-files...')
	max_wordLength, preCompute_odensity, preCompute_slength = span_sent_oovDen_length(train_vocab,test_seqchar_sent,fnwrite_sentLevelDensity)
	print('finish loading the sentence level feature...')
	# preCompute_ambSpan = []
	# preCompute_ambToken = []
	# preCompute_freqSpan = []
	# preCompute_freqToken = []
	################       Calculate attribute values for each test entity:
	#  Dict:  position -> attribute value
	''' For example:

	Id  : 0 1   2       3   4       5   6        7
	Sent: I am going to New York next month
	Tags: O O   O       O  LOC  LOC  O        O

	dict_entity2amb[4_5] = 0.8
	dict_token2amb[4] = 0.4
	dict_token2amb[5] = 0.3
			...
	'''

	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}

	dict_entity2amb, dict_token2amb, \
	dict_entity2fre, dict_token2fre, \
	dict_entity2elen, dict_entity2slen, dict_entity2oden, dict_entity2tag = getAttrValue(word_true_test, truetag_test, word_true_test_sent,
																			preCompute_ambSpan,
																			preCompute_ambToken,
																			preCompute_freqSpan,
																			preCompute_freqToken,
																			preCompute_odensity,
																			preCompute_slength,
																			)
	dict_span2aspectVal["MF-et"] = dict_entity2amb
	dict_span2aspectVal["MF-tt"] = dict_token2amb
	dict_span2aspectVal["F-ent"] = dict_entity2fre
	dict_span2aspectVal["F-tok"] = dict_token2fre
	dict_span2aspectVal["R-eLen"] = dict_entity2elen
	dict_span2aspectVal["R-sLen"] = dict_entity2slen
	# dict_span2aspectVal["R-eDen"] = dict_entity2eden
	dict_span2aspectVal["R-oov"] = dict_entity2oden
	dict_span2aspectVal["R-tag"] = dict_entity2tag



	print('finish getAttrValue-true ...') # cost most time...
	dict_entity2amb_pred, dict_token2amb_pred, \
	dict_entity2fre_pred, dict_token2fre_pred, \
	dict_entity2elen_pred, dict_entity2slen_pred, dict_entity2oden_pred, dict_entity2tag_pred= getAttrValue(word_pred_test, predtag_test, word_true_test_sent,
																			preCompute_ambSpan,
																			preCompute_ambToken,
																			preCompute_freqSpan,
																			preCompute_freqToken,
																			preCompute_odensity,
																			preCompute_slength,
																			)
	dict_span2aspectVal_pred["MF-et"] = dict_entity2amb_pred
	dict_span2aspectVal_pred["MF-tt"] = dict_token2amb_pred
	dict_span2aspectVal_pred["F-ent"] = dict_entity2fre_pred
	dict_span2aspectVal_pred["F-tok"] = dict_token2fre_pred
	dict_span2aspectVal_pred["R-eLen"] = dict_entity2elen_pred
	dict_span2aspectVal_pred["R-sLen"] = dict_entity2slen_pred
	# dict_span2aspectVal_pred["R-eDen"] = dict_entity2eden_pred
	dict_span2aspectVal_pred["R-oov"] = dict_entity2oden_pred
	dict_span2aspectVal_pred["R-tag"] = dict_entity2tag_pred

	# print("dict_span2aspectVal_pred[R-tag]: ", dict_span2aspectVal_pred["R-tag"])

	print('finish getAttrValue-predict ...')  # cost most time...



	################       Bucketing different Attributes:
	#n_buckets = 4
	#hardcoded_bucket_values = [set([float(0), float(1)])]
	# dict_hardcoded_mode = {}
	# dict_hardcoded_mode[1] = [float(0.0), float(1.0)]
	# dict_hardcoded_mode[2] = [float(1.0)]
	# dict_hardcoded_mode[3] = [float(0.0)]
	# dict_hardcoded_mode[4] = []
	# dict_hardcoded_mode[5] = [float(1.0), float(2.0)]



	def __selectBucktingFunc(func_name, func_setting, dict_obj):
		if func_name == "bucketAttribute_SpecifiedBucketInterval":
			return eval(func_name)(dict_obj, eval(func_setting))
		elif func_name == "bucketAttribute_SpecifiedBucketValue":
			if len(func_setting.split("\t"))!=2:
				raise ValueError("selectBucktingFunc Error!")
			n_buckets, specified_bucket_value_list = int(func_setting.split("\t")[0]), eval(func_setting.split("\t")[1])
			return eval(func_name)(dict_obj, n_buckets, specified_bucket_value_list)
		elif func_name == "bucketAttribute_DiscreteValue": # now the discrete value is R-tag..
			if len(func_setting.split("\t"))!=2:
				raise ValueError("selectBucktingFunc Error!")
			tags_list = list(set(dict_obj.values()))
			print("tags_list: ",tags_list)
			print("len(tags_list): ",len(tags_list))
			topK_buckets, min_buckets = int(func_setting.split("\t")[0]), int(func_setting.split("\t")[1])
			print('topK_buckets: ',topK_buckets)
			print('min_buckets: ',min_buckets)
			#return eval(func_name)(dict_obj, min_buckets, topK_buckets)
			return eval(func_name)(dict_obj, topK_buckets, min_buckets)


	dict_bucket2span = {}
	dict_bucket2span_pred = {}
	dict_bucket2f1={}
	aspect_names = []
	for aspect, func in dict_aspect_func.items():
		#print(aspect, dict_span2aspectVal[aspect])
		dict_bucket2span[aspect]      = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal[aspect])
		# print(aspect, dict_bucket2span[aspect])
		#exit()
		dict_bucket2span_pred[aspect] = bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect], dict_bucket2span[aspect].keys())
		# print("dict_bucket2span_pred[aspect]: ", dict_bucket2span_pred[aspect])
		dict_bucket2f1[aspect]  	  = getBucketF1(dict_bucket2span[aspect],  dict_bucket2span_pred[aspect])
		print("dict_bucket2f1[aspect]: ",dict_bucket2f1[aspect])
		aspect_names.append(aspect)
	print("aspect_names: ",aspect_names)



	return dict_bucket2f1,aspect_names



def evaluate_wordchunk_level(pred_chunks,true_chunks):
	# print('pred_chunks[-10:]',pred_chunks[-10:])
	# print('true_chunks[-10:]',true_chunks[-10:])
	print('len(pred_chunks)',len(pred_chunks))
	print('len(true_chunks)', len(true_chunks))
	correct_preds, total_correct, total_preds = 0., 0., 0.
	correct_preds += len(set(true_chunks) & set(pred_chunks))
	total_preds += len(pred_chunks)
	total_correct += len(true_chunks)
	print('correct_preds',correct_preds)

	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
	# acc = np.mean(accs)
	# f1, p, r = f1*100, p*100, r*100
	return f1, p, r

def draw_f1_fig(model_names,metric_values,metric_name,corpus_type):
	print('metric_name',metric_name)
	print('metric_values',metric_values)
	label_list = metric_values[0].keys()
	# print('label_list',label_list)
	num_lists=[]
	for metric_dic in metric_values:
		num_list =[]
		for key,value in metric_dic.items():
			num_list.append(value)
		num_lists.append(num_list)



	plt.figure(figsize=(15, 8))
	width = 0.08
	total_width, n = width * len(model_names), len(model_names)


	# total_width, n = 0.2*len(model_names), len(model_names)
	# width = total_width / n
	x = np.arange(len(num_lists[0])) + 1
	x = x - (total_width - width)


	# x = np.arange(len(num_lists[0])) +1
	print('x',x)
	"""
	绘制条形图
	left:长条形中点横坐标
	height:长条形高度
	width:长条形宽度，默认值0.8
	label:为后面设置legend准备
	"""
	rects = []
	colors = ['#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#71AD47', '#264478', '#9E480D', '#636363', '#997300',
					  '#255E91', '#43682B', '#698ED0', '#F1975A', '#B7B7B7', '#FFCD32', '#8CC168', '#8CC168',
				'#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#71AD47', '#264478', '#9E480D', '#636363', '#997300',
			  '#255E91', '#43682B', '#698ED0', '#F1975A', '#B7B7B7', '#FFCD32', '#8CC168', '#8CC168',
				'#ED7D31', '#A5A5A5','#FFC000', '#5B9BD5', '#71AD47', '#264478', '#9E480D', '#636363', '#997300',
			  '#255E91', '#43682B', '#698ED0', '#F1975A', '#B7B7B7', '#FFCD32', '#8CC168', '#8CC168'
			  ]
	# if metric_name == 'phoSpan_block_acc':
	for j,(num_list,model_name,color) in enumerate(zip(num_lists,model_names,colors)):
		print('metric_name',metric_name)
		print('num_list',num_list)
		print('j',j)
		num_list = [float(y) for y in num_list]
		# rects.append(plt.bar(left=[xx+j*0.4 for xx in x], height=num_list, width=0.4, color=color, label=model_name))
		rects.append(plt.bar(x+j*width, num_list, width=width,color=color, label=model_name))
		# rects.append(plt.bar(np.arange(len(num_list)) +0.4*j, num_list, width=0.4, color=color,label=model_name))



	# for j,(num_list,model_name,color) in enumerate(zip(deta_highss,model_names[1:],colors)):
	# 	print('metric_name',metric_name)
	# 	print('num_list',num_list)
	# 	print('j',j)
	# 	num_list = [float(y) for y in num_list]
	# 	# rects.append(plt.bar(left=[xx+j*0.4 for xx in x], height=num_list, width=0.4, color=color, label=model_name))
	# 	rects.append(plt.bar(x+j*width, num_list, width=width,color=color, label=model_name))
	# 	# rects.append(plt.bar(np.arange(len(num_list)) +0.4*j, num_list, width=0.4, color=color,label=model_name))


	plt.ylim(0, 100)  # y轴取值范围
	if metric_name =='sent_MaskedToken_bertp_accu' or metric_name =='sent_length_block_acc' \
			or metric_name =='entityToken_density_block_acc' or metric_name =='oov_density_block_acc':
		plt.ylim(50, 100)
	plt.ylabel("Accuracy")
	"""
	设置x轴刻度显示值
	参数一：中点坐标
	参数二：显示值
	"""
	plt.xticks(x+0.5*n*width,label_list)
	plt.xlabel("Baskets")
	plt.title(metric_name)
	plt.legend()  # 设置题注
	# # 编辑文本
	# for rects1 in rects:
	# 	for rect in rects1:
	# 		height = rect.get_height()
	# 		plt.text(rect.get_x() + rect.get_width() / 2.0, height + 1, str(height), ha="center", va="bottom")

	plt.show()
	save_path = 'new_metric/draw_f1/' + corpus_type + '_'  +metric_name +'_'+ 'f1_0917_2310.pdf'

	plt.savefig(save_path)
	plt.close()


def get_selectBucket_xyvalue_minmax_4bucket(corpus_type,
											sub_modelname,
											metric_values,
											metric_names,
											model_names,
											seltAlwaysGood_m1,
											seltAlwaysBad_m2,
											dic_tag_idx,
											operation='max'):
	'''
	midx1 and midx2 are model names' index, if the model-name isn't exist, the index equals to 10000.
	'''
	midx1, midx2 = 10000, 10000
	for p, model_name in enumerate(model_names):
		if seltAlwaysGood_m1 in model_name:
			midx1 = p
		if seltAlwaysBad_m2 in model_name:
			midx2 = p
	print('midx1, midx2', midx1, midx2)
	print("seltAlwaysGood_m1: ",seltAlwaysGood_m1)
	print('seltAlwaysBad_m2: ',seltAlwaysBad_m2)
	print('model_names: ',model_names)

	model1 = metric_values[midx1]
	model2 = metric_values[midx2]
	selectedBuckets = []
	print('len(model1)', len(model1))

	select_bucket_min_idx = []
	select_bucket_max_idx = []
	bucket_lengths = []
	return_string_list = []
	return_deta_heatmap_list = []
	for j, (metric1, metric2) in enumerate(zip(model1, model2)):
		print('metric_names: ',metric_names[j])
		print("metric1: ",metric1)
		attr_ranges = list(metric1.keys())
		list11 = list(metric1.values())
		list22 = list(metric2.values())
		list1, list2 = [],[]
		for elem1,elem2 in zip(list11,list22):
			list1.append(elem1[0])
			list2.append(elem2[0])

		list_key = list(metric1.keys())
		detas = []
		select_list = []
		detas_subs = []
		detas_subs_B = []
		self_diag = ''
		print('list1: ',list1)
		print('list2: ', list2)
		# select reversal bucket ...
		for i in range(len(list1)):
			sub = float(list1[i]) - float(list2[i])
			detas_subs.append(sub)
			if float(list2[i]) == 0:
				detas_subs_B.append(0)
			else:
				detas_subs_B.append(sub/float(list2[i]) )
		print()
		print('detas_subs: ',detas_subs_B)
		print()
		return_deta_heatmap_list.append(detas_subs_B)
		bucket_lengths.append(len(detas_subs))
		index = -1

		if operation=='exceed_reversal': # the input two model is the same one!
			min_value = min(detas_subs)
			min_deta_idx = detas_subs.index(min_value)
			max_value = max(detas_subs)
			max_deta_idx = detas_subs.index(max_value)
			if min_value >=0:
				min_value=0.0
				min_deta_idx= 0
			if max_value <=0:
				max_value=0.0
				max_deta_idx=0

			print("attr_ranges: ",attr_ranges)
			print("min_deta_idx: ",min_deta_idx)
			print("max_deta_idx: ",max_deta_idx)

			# begin{aided-diagnosis, return the best & worse index bucket for the 4 layers results}
			worse_bucket_xlabel = return_4bucket_xsticks(min_deta_idx,metric_names[j],dic_tag_idx)
			best_bucket_xlabel = return_4bucket_xsticks(max_deta_idx,metric_names[j],dic_tag_idx)
			# string1 = metric_names[j] + '\t' + worse_bucket_xlabel + ':' + str(
			# 	min_value)+ ':' +return_idx_range(attr_ranges,min_deta_idx) + '\t' + best_bucket_xlabel + ':' + str(max_value)+ ':' +return_idx_range(attr_ranges,max_deta_idx)
			string1 = metric_names[j] + '\t' + worse_bucket_xlabel + ':' + str(
				min_value) + ' ' + best_bucket_xlabel + ':' + str(max_value)
			return_string_list.append(string1)
			# end{aided-diagnosis, return the best & worse index bucket for the 4 layers results}


			select_bucket_min_idx.append(min_deta_idx)
			select_bucket_max_idx.append(max_deta_idx)
		elif operation=='self_diagnose':
			best_buctet = max(list1)
			worse_bucket = min(list1)
			best_index = list1.index(best_buctet)
			worse_index = list1.index(worse_bucket)

			self_diag1 = float(best_buctet) - float(worse_bucket)
			print('worse_bucket:', worse_bucket)
			print('best_buctet:', best_buctet)
			print('self_diag:', self_diag1)

			# begin{self-diagnosis, return the best & worse index bucket for the 4 layers results}
			worse_bucket_xlabel = return_4bucket_xsticks(worse_index,metric_names[j],dic_tag_idx)
			best_bucket_xlabel = return_4bucket_xsticks(best_index,metric_names[j],dic_tag_idx)
			print("attr_ranges[best_index]: ",attr_ranges[best_index])
			# string1 = metric_names[j]+'\t'+ worse_bucket_xlabel+':'+str(worse_bucket)+ ':' +return_idx_range(attr_ranges,worse_index)+'\t'+best_bucket_xlabel+':'+str(best_buctet)+ ':' +return_idx_range(attr_ranges,best_index)+'\t'+ str(self_diag1)
			string1 = metric_names[j]+'\t'+ worse_bucket_xlabel+':'+str(worse_bucket)+' '+best_bucket_xlabel+':'+str(best_buctet)+' '+ str(self_diag1)
			return_string_list.append(string1)
			# end{self-diagnosis, return the best & worse index bucket for the 4 layers results}

			select_bucket_min_idx.append(worse_index)
			select_bucket_max_idx.append(best_index)



	# select_bucket_min_SML = []
	# select_bucket_max_SML = []
	# for min_idx,max_idx in zip(select_bucket_min_idx,select_bucket_max_idx):
	# 	sml_min = return_4bucket_xsticks(min_idx)
	# 	sml_max = return_4bucket_xsticks(max_idx)

	# 	select_bucket_min_SML.append(sml_min)
	# 	select_bucket_max_SML.append(sml_max)

	# min_sml_str= '\',\''.join(select_bucket_min_SML)
	# max_sml_str = '\',\''.join(select_bucket_max_SML)
	# min_sml_str2 = '[\'' + min_sml_str + '\']'
	# max_sml_str2 = '[\'' + max_sml_str + '\']'
	# min_sml_str = sub_modelname+'_min_sml_str_'+ '[\''+ min_sml_str+'\']'
	# max_sml_str = sub_modelname + '_max_sml_str_' + '[\'' + max_sml_str + '\']'

	# print('min_sml_str:', min_sml_str)
	# print('max_sml_str:', max_sml_str)
	min_sml_str2 =''
	max_sml_str2 = ''
	return min_sml_str2,max_sml_str2,return_string_list,return_deta_heatmap_list




def return_4bucket_xsticks(index,metric_name,dic_tag_idx):
	value = index
	return str(value)
	# if metric_name!='R-tag':
	# 	value =''
	# 	if index==10000:
	# 		value='N'
	# 	elif index==0:
	# 		value = 'XS'
	# 	elif index==1:
	# 		value = 'S'
	# 	elif index==2:
	# 		value = 'L'
	# 	elif index==3:
	# 		value = 'XL'
	# 	return value
	# else:
	# 	# value = dic_tag_idx[index]
	# 	value = str(index+1)
	# 	return value


def printDict(dict_obj, info="dict"):
	print("-----------------------------------------------")
	print("the information of #" + info + "#")
	print("Bucket_interval\tF1\tEntity-Number")
	for k,v in dict_obj.items():
		if len(k) == 1:
			print("[" + str(k[0])+",]" + "\t" + str(v[0]) + "\t" + str(v[1]))
		else:
			print("[" + str(k[0])+", " + str(k[1]) +"]" + "\t" + str(v[0]) + "\t" + str(v[1]))


def compute_holistic_f1(corpus_type,fn_test_results,fn_test,iftest=False):
	test_seqchar, test_truetags, test_predtags, test_seqchar_sent, test_truetags_sent, test_predtags_sent = read_data_test2(
		corpus_type, fn_test_results, fn_test)
	y = test_truetags_sent
	y_pred = test_predtags_sent

	y = sum(y,[])
	y_pred = sum(y_pred, [])
	# print('y_pred',y_pred)
	print()
	e_idx = 'E'
	s_idx = 'S'
	cor_num = 0
	yp_wordnum = y_pred.count(e_idx)+y_pred.count(s_idx)
	yt_wordnum = y.count(e_idx)+y.count(s_idx)
	start = 0
	for i in range(len(y)):
		if y[i] == e_idx or y[i] == s_idx:
			flag = True
			for j in range(start, i+1):
				if y[j] != y_pred[j]:
					flag = False
			if flag == True:
				cor_num += 1
			start = i+1

	P = 100*cor_num / float(yp_wordnum) if yp_wordnum > 0 else 0.0
	R = 100*cor_num / float(yt_wordnum) if yt_wordnum > 0 else 0.0
	F = 2 * P * R / (P + R) if yp_wordnum > 0 else 0.0
	

	print('P: ', P)
	print('R: ', R)
	print('F: ', F)

	P = '%.2f'% P
	R = '%.2f'% R
	F = '%.2f'% F

	if iftest:
		return P,R,F
	else:
		return F


def printDict(dict_obj, info="dict"):
	print("-----------------------------------------------")
	print("the information of #" + info + "#")
	print("Bucket_interval\tF1\tEntity-Number")
	for k,v in dict_obj.items():
		if len(k) == 1:
			print("[" + str(k[0])+",]" + "\t" + str(v[0]) + "\t" + str(v[1]))
		else:
			print("[" + str(k[0])+", " + str(k[1]) +"]" + "\t" + str(v[0]) + "\t" + str(v[1]))

def extValue(cont, fr, to):
    return cont.split(fr)[-1].split(to)[0] 


def loadConf(path_conf):
	fin = open(path_conf,"r")
	all_cont = fin.read()
	dict_aspect_func={}
	for block in all_cont.split("# "):
		notation = extValue(block, "notation:\t", "\n").rstrip(" ")
		if notation == "":
			continue
		func_type = extValue(block, "type:\t", "\n").rstrip(" ")
		func_setting = extValue(block, "setting:\t", "\n").rstrip(" ")
		dict_aspect_func[notation] = (func_type, func_setting)
	return dict_aspect_func



def write_breakDown_performance(corpus_type, model_names, stdModels_metrics,metric_names,fwrite_evaluate,fwrite_buckect_value):
	# begin{write the break-down performance in the file...}
	# fwrite_bucket = open(fn_buckect_value, 'w')
	fwrite_buckect_value.write('## '+corpus_type+'\n')
	idx_dic = {0:"XS",1:"S",2:"L",3:"XL"}
	string1 = "# break-down performance\n"
	fwrite_evaluate.write(string1)
	print("string1: ", string1)
	dic_tag_idx={}
	for mn, (model_name, attrs_result) in enumerate(zip(model_names, stdModels_metrics)):
		string2 = model_name
		fwrite_evaluate.write("%s\n"%(string2) )
		fwrite_buckect_value.write('@@'+model_name+'\n')

		for metric_name, metric_result in zip(metric_names, attrs_result):
			string4bkv =metric_name+' '
			if metric_name!='R-tag':
				# print("metric_name: ",metric_name)
				# print("metric_result: ",metric_result)
				string33=''
				string4bkv=''
				idx = 0
				print('metric_result: ',metric_result)
				for range1, f1_score in metric_result.items():
					#xlabel = idx_dic[idx]
					xlabel = str(idx)
					idx+=1
					string33 += xlabel+':'+str(f1_score[0]) +' '
					if len(range1)==1:
						range1_r = '('+format(float(range1[0]), '.3g') +',)'
						string4bkv +=xlabel+':'+range1_r + " "
					else:
						range1_r = '('+format(float(range1[0]), '.3g') +','
						range1_l = format(float(range1[1]), '.3g') +')'
						string4bkv +=xlabel+':'+range1_r+range1_l + " "
					# string4bkv += xlabel+':'+str(range1)+' '
				string4bkv = metric_name +'\t' +string4bkv.rstrip(" ")
				fwrite_buckect_value.write("%s\n" %string4bkv)
				string3 = metric_name +'\t' +string33.rstrip(" ")
				fwrite_evaluate.write("%s\n" % (string3))
			else:
				string33=''
				string4bkv=''
				print('metric_result: ',metric_result)
				xlabel=0
				for range1, f1_score in metric_result.items():
					tag=str(range1).split(',')[0][1:]
					dic_tag_idx[xlabel]=tag

					string33 += str(xlabel)+':'+str(f1_score[0])+' '
					string4bkv += str(xlabel)+':'+str(range1)+' '
					xlabel +=1 
				string4bkv = metric_name +'\t' +string4bkv.rstrip(" ")
				fwrite_buckect_value.write("%s\n" %string4bkv)
				string3 = metric_name +'\t' +string33.rstrip(" ")
				fwrite_evaluate.write("%s\n" % (string3))
		print("dic_tag_idx: ",dic_tag_idx)
		# fwrite_buckect_value.write('\n')
		fwrite_evaluate.write("\n")
	# # end{write the break-down performance in the file...}


	return dic_tag_idx


if __name__ == '__main__':
	# parser = argparse.ArgumentParser(description='Learning tagger using neural networks')
	# parser.add_argument('--data_list', type=str, required=True, nargs='+',
	# 					help="a list to store [[corpus_type, model_name, result_file], ]")
	# parser.add_argument('--model_list', type=str, required=True, nargs='+',
	# 					help="a list to store [[corpus_type, model_name, result_file], ]")
	# parser.add_argument('--resfile_list', type=str,  required=True,nargs='+',
	# 					help="a list to store [[corpus_type, model_name, result_file], ]")
	# args = parser.parse_args()

	parser = argparse.ArgumentParser(description='Learning tagger using neural networks')

	parser.add_argument('--data_list', type=str, required=True, nargs='+',
						help="a list to store [[corpus_type, model_name, result_file], ]")
	parser.add_argument('--model_list', type=str, required=True, nargs='+',
						help="a list to store [[corpus_type, model_name, result_file], ]")
	parser.add_argument('--resfile_list', type=str,  required=True,nargs='+',
						help="a list to store [[corpus_type, model_name, result_file], ]")


	parser.add_argument('--path_data', type=str, required=True, 
						help="path of training and test set")

	parser.add_argument('--path_output_tensorEval', type=str, required=True, 
						help="path of training and test set")
	parser.add_argument('--path_preComputed', type=str, required=True, 
						help="path of training and test set")
	parser.add_argument('--path_aspect_conf', type=str, required=True, 
						help="conf file for evaluation aspect")
	parser.add_argument('--task_type', type=str, required=True, 
						help="the type of the task")
	parser.add_argument('--path_fig', type=str, required=True, 
						help="the type of the task")

	args = parser.parse_args()

	print('args.data_list', args.data_list[0].split(" "))  ######### note the split(" ") for data_list
	print('args.model_list', args.model_list)
	print('args.resfile_list', args.resfile_list)

	if len(args.model_list) * len(args.data_list[0].split(" ")) != len(args.resfile_list):
		raise ValueError('Lengs of the args.model_list, args.data_list, and args.resfile_list must be the same.s ')

	corpus_types = args.data_list[0].split(" ")
	model_names = args.model_list
	fn_results = args.resfile_list
	model_name1 = model_names[0]
	model_name2 = model_names[1]

	path_data = args.path_data
	task_type = args.task_type
	path_output_tensorEval = args.path_output_tensorEval
	path_preComputed = args.path_preComputed
	path_aspect_conf = args.path_aspect_conf

	dict_aspect_func = loadConf(path_aspect_conf)
	metric_names = list(dict_aspect_func.keys())
	print("dict_aspect_func: ",dict_aspect_func)
	print(dict_aspect_func)

	spears = []
	radar_xticks =[]
	stds = []
	corpus_attrs = []
	entity_norepeats = []

	# ------------------------------------------------ General Setting -----------------------------------------
	n_buckets = 4

	dict_hardcoded_mode = {}
	dict_hardcoded_mode[1] = [float(0.0), float(1.0)]
	dict_hardcoded_mode[2] = [float(1.0)]
	dict_hardcoded_mode[3] = [float(0.0)]
	dict_hardcoded_mode[4] = []
	dict_hardcoded_mode[5] = [float(1.0), float(2.0)]

	# ---------------------------------------------------------------- -----------------------------------------


	
	filename = '-'.join(model_names)
	fn_write_buckect_value = 'analysis/'+args.path_fig+'/'+filename+'/bucket.range'
	print('fn_write_buckect_value: ',fn_write_buckect_value)
	fwrite_buckect_value = open(fn_write_buckect_value, 'w')

	dict_bucket_info = {}
	fn_write_buckect_info = 'analysis/'+args.path_fig+'/'+filename+'/bucketInfo.pkl'




	for ind, corpus_type in enumerate(corpus_types):
		fn_evaluate = path_output_tensorEval +'/' +corpus_type + "-" + filename + "-5Lresults.txt"
		fwrite_evaluate = open(fn_evaluate,'w+')

		column_no = -1
		mode = ' '
		pos_column = 0

		fn_train = path_data +task_type + "/data_" + corpus_type +'/train'
		fn_test  = path_data +task_type + "/data_" + corpus_type +'/test'

		# fn_test = 'new_metric/analysis/new_york_example.txt'
		train_vocab 			 = path_preComputed + '/vocab-cws/' + corpus_type + '_vocab.txt'
		fnwrite_GoldenWordHard 	 = path_preComputed + '/cws_metric/' + corpus_type + '_rhoWordHardFreq.pkl'
		fnwrite_GoldenCharHard 	 = path_preComputed + '/cws_metric/' + corpus_type + '_rhoCharHardFreq.pkl'
		fnwrite_sentLevelDensity = path_preComputed + '/cws_metric/' + corpus_type + '_sentLevelDensity.pkl'

		# begin{compute holistic f1}
		fwrite_evaluate.write("# information\n")
		fwrite_evaluate.write("corpus type: %s\n" % corpus_type)
		fwrite_evaluate.write("model_name1: %s\n" % model_name1)
		fwrite_evaluate.write("model_name2: %s\n" % model_name2)

		fn_results_sub = fn_results[ind*2:ind*2+2]

		# # begin{compute holistic f1}
		# fwrite_evaluate.write("# holistic result \n")
		# for fn_result,model_name in zip(fn_results,model_names):
		# 	fn_test_results = data_dir +fn_result
		# 	print('fn_test_results:',fn_test_results)
		# 	# f1 =compute_holistic_f1(fn_test_results)
		# 	f1 = compute_holistic_f1(corpus_type,fn_test_results,fn_test,iftest=False)
		# 	string = "model_name: "+model_name+" result: "+str(f1)
		# 	fwrite_evaluate.write("%s\n" %(string) )
		# fwrite_evaluate.write('\n')

		fwrite_evaluate.write("# holistic result \n")
		for fn_result, model_name in zip(fn_results_sub, model_names):
			fn_test_results = path_preComputed + "/cws_results/" + fn_result
			print('fn_test_results:',fn_test_results)
			f1 = compute_holistic_f1(corpus_type,fn_test_results,fn_test,iftest=False)
			string = model_name + ": "+str(f1)
			fwrite_evaluate.write("%s\n" %(string) )
		fwrite_evaluate.write('\n')

		tokenDen_f1 =[]
		oovDen_f1 =[]
		sentProb_f1 =[]
		sentLen_f1 = []
		entityLen_f1 =[]
		rhoSpan_f1 =[]
		label_f1 =[]
		spanFreq_f1 =[]
		phoToken_acc =[]
		morpho_acc =[]
		pos_acc =[]
		tokenFreq_acc = []


		models_metrics = []
		stdModels_metrics = []
		for fn_result,model_name in zip(fn_results,model_names):
			fn_test_results = path_preComputed + "/cws_results/" + fn_result
			print('fn_test_results',fn_test_results)


			dict_bucket2f1,aspect_names = new_metric(corpus_type,
															column_no,
															mode,
															pos_column,
															train_vocab,
															n_buckets,
															dict_hardcoded_mode,
															fn_train,
															fn_test_results,
															fn_test,
														   fnwrite_GoldenWordHard,
														   fnwrite_GoldenCharHard,
														   fnwrite_sentLevelDensity,
															 )
			print("@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@")
			print("@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@")
			# entity_norepeats.append(entity_norepeat)
			print('model_name',model_name)
			print("####################################################")

			for aspect in dict_aspect_func.keys():
				printDict(dict_bucket2f1[aspect], aspect)

			# Save the information of buckets (interval, F1, n_spans) of each dataset
			if corpus_type not in dict_bucket_info.keys():
				dict_bucket_info[corpus_type] = dict_bucket2f1
			


			std_model_metrics = list(dict_bucket2f1.values())
			stdModels_metrics.append(std_model_metrics)
		dic_tag_idx = write_breakDown_performance(corpus_type,model_names, stdModels_metrics,metric_names,fwrite_evaluate,fwrite_buckect_value)
		
		# --------------------- save information for bucketing dataset ------------------------------
		pickle.dump(dict_bucket_info, open(fn_write_buckect_info, "wb"))

		
		min_sml_strs, max_sml_strs = [], []
		sub_modelnames = []
		return_deta_heatmap_lists =[]
		# begin select bucket ...
		# operation='self_diagnose';  reversal; max; min
		print('corpus_type:', corpus_type)
		print('###########################____select_buckets____#######################')
		print('#####compare CcnnWglove_lstmCrf self diagnose, model is: CcnnWglove_lstmCrf, CcnnWglove_lstmCrf...')
		string1 = "# self-diagnosis \n"
		fwrite_evaluate.write(string1)
		for mn1 in model_names:
			seltAlwaysGood_m1 = mn1
			seltAlwaysBad_m2 = mn1
			sub_modelname = mn1 +'_'+mn1
			min_sml_str, max_sml_str, return_string_list,return_deta_heatmap_list = get_selectBucket_xyvalue_minmax_4bucket(corpus_type,
																							   sub_modelname,
																							   stdModels_metrics,
																							   metric_names,
																							   model_names,
																							   seltAlwaysGood_m1,
																							   seltAlwaysBad_m2,
																							   dic_tag_idx,
																							   operation='self_diagnose')
			min_sml_strs.append(min_sml_str)
			max_sml_strs.append(max_sml_str)
			sub_modelnames.append(sub_modelname)

			# begin{write the CcnnWglove_lstmCrf self-diagnosis into the file...}
			string2 = seltAlwaysGood_m1 +'\n'
			fwrite_evaluate.write(string2)
			for string3 in return_string_list:
				fwrite_evaluate.write(string3+'\n')
			fwrite_evaluate.write('\n')
		# end{write the CcnnWglove_lstmCrf self-diagnosis into the file...}s


		print('aided-diagnosis: compare lstm&cnn, model is: CcnnWglove_lstmCrf, CcnnWglove_cnnCrf...')
		seltAlwaysGood_m1=model_names[0]
		seltAlwaysBad_m2=model_names[1]
		sub_modelname= seltAlwaysGood_m1+'_' +seltAlwaysBad_m2
		min_sml_str, max_sml_str,return_string_list,return_deta_heatmap_list = get_selectBucket_xyvalue_minmax_4bucket(corpus_type, 
																   sub_modelname, 
																   stdModels_metrics,
																   metric_names, 
																   model_names,
																   seltAlwaysGood_m1, 
																   seltAlwaysBad_m2,
																   dic_tag_idx,
																   operation='exceed_reversal')
		min_sml_strs.append(min_sml_str)
		max_sml_strs.append(max_sml_str)
		sub_modelnames.append(sub_modelname)

		# begin{write the CcnnWglove_lstmCrf self-diagnosis into the file...}
		string1 = "# aided-diagnosis line-chart\n"
		fwrite_evaluate.write(string1)
		string2 = seltAlwaysGood_m1+'_'+seltAlwaysBad_m2 + '\n'
		fwrite_evaluate.write(string2)
		for string3 in return_string_list:
			fwrite_evaluate.write(string3 + '\n')
		fwrite_evaluate.write('\n')

		return_deta_heatmap_lists.append(return_deta_heatmap_list)
		# end{write the CcnnWglove_lstmCrf self-diagnosis into the file...}

		# begin{write detas of aided-diagnosis for heatmap}
		stringa = "# aided-diagnosis heatmap\n"
		#string1 = "heatmap	XS S L XL\n"
		string11 = seltAlwaysGood_m1+'_'+seltAlwaysBad_m2 + '\n'
		fwrite_evaluate.write(stringa)
		#fwrite_evaluate.write(string1)
		fwrite_evaluate.write(string11)
		for return_deta_for_heatmap in return_deta_heatmap_lists:
			for metric_name, detas in zip(metric_names, return_deta_for_heatmap):
				string2 =''
				for value in detas:
					string2 +=str(value)+' '
				string3 = metric_name +'\t' +string2 +'\n'
				fwrite_evaluate.write(string3)





