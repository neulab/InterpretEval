# -*- coding: utf-8 -*-
import numpy as np
import pickle
import codecs
import os
from collections import Counter
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


def get_chunk_type(tok):
	"""
	Args:
		tok: id of token, ex 4
		idx_to_tag: dictionary {4: "B-PER", ...}
	Returns:
		tuple: "B", "PER"
	"""
	# tag_name = idx_to_tag[tok]
	tag_class = tok.split('-')[0]
	tag_type = tok.split('-')[-1]
	return tag_class, tag_type

def get_chunks(seq):
	"""
	tags:dic{'per':1,....}
	Args:
		seq: [4, 4, 0, 0, ...] sequence of labels
		tags: dict["O"] = 4
	Returns:
		list of (chunk_type, chunk_start, chunk_end)

	Example:
		seq = [4, 5, 0, 3]
		tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
		result = [("PER", 0, 2), ("LOC", 3, 4)]
	"""
	default = 'O'
	# idx_to_tag = {idx: tag for tag, idx in tags.items()}
	chunks = []
	chunk_type, chunk_start = None, None
	for i, tok in enumerate(seq):
		#End of a chunk 1
		if tok == default and chunk_type is not None:
			# Add a chunk.
			chunk = (chunk_type, chunk_start, i)
			chunks.append(chunk)
			chunk_type, chunk_start = None, None

		# End of a chunk + start of a chunk!
		elif tok != default:
			tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
			if chunk_type is None:
				chunk_type, chunk_start = tok_chunk_type, i
			elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
				chunk = (chunk_type, chunk_start, i)
				chunks.append(chunk)
				chunk_type, chunk_start = tok_chunk_type, i
		else:
			pass
	# end condition
	if chunk_type is not None:
		chunk = (chunk_type, chunk_start, len(seq))
		chunks.append(chunk)

	return chunks


def read_data(corpus_type, fn, column_no=-1,mode=' '):
	print('corpus_type',corpus_type)
	word_sequences = list()
	tag_sequences = list()
	total_word_sequences = list()
	total_tag_sequences = list()
	with codecs.open(fn, 'r', 'utf-8') as f:
		lines = f.readlines()
	curr_words = list()
	curr_tags = list()
	for k in range(len(lines)):
		line = lines[k].strip()
		if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
			if len(curr_words) > 0:
				word_sequences.append(curr_words)
				tag_sequences.append(curr_tags)
				curr_words = list()
				curr_tags = list()
			continue

		strings = line.split(mode)
		word = strings[0].strip()
		tag = strings[column_no].strip()  # be default, we take the last tag
		if corpus_type=='ptb2':
			tag='B-'+tag
		curr_words.append(word)
		curr_tags.append(tag)
		total_word_sequences.append(word)
		total_tag_sequences.append(tag)
		if k == len(lines) - 1:
			word_sequences.append(curr_words)
			tag_sequences.append(curr_tags)
	# if verbose:
	# 	print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
	# return word_sequences, tag_sequences
	return total_word_sequences,total_tag_sequences,word_sequences,tag_sequences




def span_entity_amb(train_word_sequences,tag_sequences_train,fnwrite_GoldenEntityHard):
	if os.path.exists(fnwrite_GoldenEntityHard):
		print('load the hard dictionary of entity span in test set...')
		fread =open(fnwrite_GoldenEntityHard,'rb')
		Hard_entitySpan_inTrain = pickle.load(fread)
		return Hard_entitySpan_inTrain
	else:
		Hard_entitySpan_inTrain = dict()
		chunks_train = set(get_chunks(tag_sequences_train))
		tags = []
		train_count = Counter(tag_sequences_train)
		for tag_train, times in train_count.most_common():
			if len(tag_train) > 1:
				tag = tag_train.split('-')[1].lower()
				if tag not in tags:
					tags.append(tag)
		tags.append('o')
		print('len(tags)', len(tags))

		# build the entity span in test set, and its label count in test set

		count_idx = 0
		print('len(chunks_test)',len(chunks_train))
		word_sequences_train_str = ' '.join(train_word_sequences).lower()
		for true_chunk in chunks_train:
			# print('true_chunk',true_chunk)
			count_idx += 1
			print()
			print('count:',count_idx)
			type = true_chunk[0].lower()
			idx_start = true_chunk[1]
			idx_end = true_chunk[2]

			entity_span = ' '.join(train_word_sequences[idx_start:idx_end]).lower()
			# print('entity_span', entity_span)
			if entity_span in Hard_entitySpan_inTrain:
				continue
			else:
				Hard_entitySpan_inTrain[entity_span] = dict()
				for tag in tags:
					Hard_entitySpan_inTrain[entity_span][tag] = 0.0

			# Determine if the same position in pred list giving a right prediction.
			entity_span_new = ' ' + entity_span + ' '

			# print('entity_span_new', entity_span_new)
			# if '(' in entity_span_new and ')' not in entity_span_new:
			# 	entity_span_new = entity_span_new.replace('(', '')
			if '(' in entity_span_new or ')' in entity_span_new:
				entity_span_new = entity_span_new.replace('(', '')
				entity_span_new = entity_span_new.replace(')', '')
			if '*' in entity_span_new:
				entity_span_new = entity_span_new.replace('*', '')
			if '+' in entity_span_new:
				entity_span_new = entity_span_new.replace('+', '')
			print('entity_span_new',entity_span_new)
			entity_str_index = [m.start() for m in re.finditer(entity_span_new, word_sequences_train_str)]
			# print('entity_str_index',entity_str_index)
			print('count_find_span:',len(entity_str_index))
			if len(entity_str_index) > 0:
				label_list = []
				# convert the string index into list index...
				entity_list_index = []
				for str_idx in entity_str_index:
					entity_idx = len(word_sequences_train_str[0:str_idx].split())
					entity_list_index.append(entity_idx)
				entity_len = len(entity_span.split())

				for idx in entity_list_index:
					label_list_candidate = tag_sequences_train[idx:idx + entity_len]
					for label in label_list_candidate:
						if len(label) > 1:
							label_list.append(label.split('-')[1].lower())
						else:
							label_list.append(label.lower())

				label_norep = list(set(label_list))
				for lab_norep in label_norep:
					hard = float('%.3f' % (float(label_list.count(lab_norep)) / len(label_list)))
					Hard_entitySpan_inTrain[entity_span][lab_norep] = hard

		fwrite = open(fnwrite_GoldenEntityHard, 'wb')
		pickle.dump(Hard_entitySpan_inTrain, fwrite)
		fwrite.close()

		# in the true tags, the num of no repeated entity is 2529
		# the num of no repeated entity, including the true and predicted, is 2633.
		# print('len, Hard_entitySpan_inTrain',len(Hard_entitySpan_inTrain))
		return Hard_entitySpan_inTrain

def span_token_amb(word_sequences_train,tag_sequences_train,fnwrite_TokenHard):
	if os.path.exists(fnwrite_TokenHard):
		print('load the hard dictionary of entity token in test set...')
		fread = open(fnwrite_TokenHard, 'rb')
		Hard_entityToken_inTrain = pickle.load(fread)
		return Hard_entityToken_inTrain
	else:
		tags = []
		train_count = Counter(tag_sequences_train)
		for tag_train, times in train_count.most_common():
			if len(tag_train) > 1:
				tag = tag_train.split('-')[1].lower()
				if tag not in tags:
					tags.append(tag)
		tags.append('o')

		print('len(tags)', len(tags))

		# build the word2tags dictionary.
		word2tags_inTrain = dict()
		for i in range(len(word_sequences_train)):
			if len(tag_sequences_train[i]) > 1:
				tag_new = tag_sequences_train[i].split('-')[1].lower().strip()
			else:
				tag_new = tag_sequences_train[i].lower().strip()

			if word_sequences_train[i].lower() in word2tags_inTrain:
				word2tags_inTrain[word_sequences_train[i].lower()].append(tag_new)
			else:
				word2tags_inTrain[word_sequences_train[i].lower()] = [tag_new]
		print('the word list in train set is:', len(word2tags_inTrain))

		Hard_entityToken_inTrain = dict()
		for token,labels_list in word2tags_inTrain.items():
			Hard_entityToken_inTrain[token] =dict()
			for tag in tags:
				Hard_entityToken_inTrain[token][tag] = 0.0
			labels_norep = list(set(labels_list))
			for lab_norep in labels_norep:
				hard = float('%.3f' % (float(labels_list.count(lab_norep)) / len(labels_list)))
				Hard_entityToken_inTrain[token][lab_norep] = hard



		fwrite =open(fnwrite_TokenHard, 'wb')
		pickle.dump(Hard_entityToken_inTrain,fwrite)
		fwrite.close()


		# print('len(Hard_entityToken_inTrain)',len(Hard_entityToken_inTrain))

		# the num of no repeated entity token in test set is 8,548

		return Hard_entityToken_inTrain


def span_entity_fre(train_word_sequences,tag_sequences_train,fnwrite_GoldenEntityHard):
	if os.path.exists(fnwrite_GoldenEntityHard):
		print('load the hard dictionary of entity span in test set...')
		fread =open(fnwrite_GoldenEntityHard,'rb')
		entitySpan_fami_inTrain = pickle.load(fread)
		return entitySpan_fami_inTrain
	else:
		entitySpan_fami_inTrain = dict()
		chunks_train = set(get_chunks(tag_sequences_train))
		count_idx = 0
		word_sequences_train_str = ' '.join(train_word_sequences).lower()
		for true_chunk in chunks_train:
			count_idx += 1
			type = true_chunk[0].lower()
			idx_start = true_chunk[1]
			idx_end = true_chunk[2]

			entity_span = ' '.join(train_word_sequences[idx_start:idx_end]).lower()
			# print('entity_span', entity_span)
			if entity_span in entitySpan_fami_inTrain:
				continue
			else:
				entitySpan_fami_inTrain[entity_span] = []

			# Determine if the same position in pred list giving a right prediction.
			entity_span_new = ' ' + entity_span + ' '
			if '(' in entity_span_new or ')' in entity_span_new:
				entity_span_new = entity_span_new.replace('(', '')
				entity_span_new = entity_span_new.replace(')', '')
			if '*' in entity_span_new:
				entity_span_new = entity_span_new.replace('*', '')

			entity_str_index = [m.start() for m in re.finditer(entity_span_new, word_sequences_train_str)]

			entitySpan_fami_inTrain[entity_span] = len(entity_str_index)

		sorted_entitySpan_fami_inTrain = sorted(entitySpan_fami_inTrain.items(), key=lambda item: item[1],reverse =True)

		entitySpan_freq_rate = {}
		count_bigerThan_maxFreq = 0
		max_freq = sorted_entitySpan_fami_inTrain[4][1]
		for span, freq in entitySpan_fami_inTrain.items():
			if freq <=max_freq:
				entitySpan_freq_rate[span] = '%.3f' % (float(freq) / max_freq)
			else:
				count_bigerThan_maxFreq+=1
		print('count_bigerThan_maxFreq',count_bigerThan_maxFreq)


		fwrite = open(fnwrite_GoldenEntityHard, 'wb')
		pickle.dump(entitySpan_freq_rate, fwrite)
		fwrite.close()

		return entitySpan_freq_rate

def span_token_fre(word_sequences_train,tag_sequences_train,fnwrite_TokenHard):
	if os.path.exists(fnwrite_TokenHard):
		print('load the hard dictionary of entity token in test set...')
		fread = open(fnwrite_TokenHard, 'rb')
		entityToken_freq_rate = pickle.load(fread)
		return entityToken_freq_rate
	else:
		# build the word2tags dictionary.
		toks = []
		for tok in word_sequences_train:
			toks.append(tok.lower())
		entityToken_freq_rate = dict()
		token_count = Counter(toks)
		max_freq = token_count.most_common(4)[0][1]
		print("max_freq:",max_freq)
		for token,times in token_count.most_common():
			if times<=max_freq:
				entityToken_freq_rate[token] = float(times) / max_freq
			else:
				entityToken_freq_rate[token] = float(1.0)
		#
		# def count_entity_inTrain(word,word_sequences_train):
		# 	count =0
		# 	for wt in word_sequences_train:
		# 		if word.lower() ==wt.lower():
		# 			count+=1
		# 	return count
		#
		# true_chunks = set(get_chunks(tag_sequences_train))
		#
		# token_fami_inTrain = dict()
		# for token in word_sequences_train:
		# 	word = token.lower()
		# 	if word not in token_fami_inTrain:
		# 		count = count_entity_inTrain(word, word_sequences_train)
		# 		token_fami_inTrain[word] = count
		#
		# sorted_token_fami_inTrain = sorted(token_fami_inTrain.items(), key=lambda item: item[1], reverse=True)
		#
		# entityToken_freq_rate = {}
		# count_token_bigerThan_maxFreq = 0
		# max_freq = sorted_token_fami_inTrain[4][1]
		# max_freq = 185
		# print('max_freq',max_freq)
		# for token, freq in token_fami_inTrain.items():
		# 	if freq <= max_freq:
		# 		entityToken_freq_rate[token] = '%.3f' % (float(freq) / max_freq)
		# 	else:
		# 		entityToken_freq_rate[token] = '1.0'
		# 		count_token_bigerThan_maxFreq += 1
		# print('count_token_bigerThan_maxFreq', count_token_bigerThan_maxFreq)
		#
		#
		fwrite =open(fnwrite_TokenHard, 'wb')
		pickle.dump(entityToken_freq_rate,fwrite)
		fwrite.close()


		# print('len(Hard_entityToken_inTrain)',len(Hard_entityToken_inTrain))

		# the num of no repeated entity token in test set is 8,548
		# entityToken_freq_rate = {}
		return entityToken_freq_rate

def span_sent_oovDen_length(train_vocab,test_trueTag_sequences_sent,test_word_sequences_sent):
	# loading the train_vocab...
	f_train_vocab = open(train_vocab, 'rb')
	word_vocab = pickle.load(f_train_vocab)

	# compute the max length of entity.
	entity_lengths =[]
	for test_sent in test_trueTag_sequences_sent:
		true_chunks = set(get_chunks(test_sent))
		for true_chunk in true_chunks:
			idx_start = true_chunk[1]
			idx_end = true_chunk[2]
			entity_lengths.append(idx_end-idx_start)
	max_entityLength = np.max(entity_lengths)

	# compute the entity token density and oov density ...
	entityToken_density = []
	oov_density = []
	sent_Lengths =[]
	for i, test_sent in enumerate(test_trueTag_sequences_sent):
		pred_chunks = set(get_chunks(test_sent))
		num_entityToken = 0
		for pred_chunk in pred_chunks:
			idx_start = pred_chunk[1]
			idx_end = pred_chunk[2]
			num_entityToken += idx_end - idx_start
		# introduce the entity token density in sentence ...
		entityToken_density.append(float(num_entityToken) / len(test_sent) )

		# introduce the oov density in sentence ...
		num_oov =0
		for word in test_word_sequences_sent[i]:
			if word not in word_vocab:
				num_oov +=1
		oov_density.append(float(num_oov) / len(test_sent) )

		# introduce the sentence length in sentence ...
		sent_Lengths.append(len(test_sent))


	return max_entityLength,entityToken_density,oov_density,sent_Lengths

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
		# print(k, v)
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

def getAttrValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
						preCompute_ambToken,
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

	dict_pos2sid = getPos2SentId(test_word_sequences_sent)

	#test_word_sequences, test_trueTag_sequences
	all_chunks = get_chunks(test_trueTag_sequences)
	for span_info in all_chunks:

		span_type = span_info[0].lower()
		idx_start = span_info[1]
		idx_end = span_info[2]
		span_cnt = ' '.join(test_word_sequences[idx_start:idx_end]).lower()
		span_pos = str(idx_start) + "_" + str(idx_end) + "_" + span_type

		span_length = idx_end - idx_start

		span_token_list = test_word_sequences[idx_start:idx_end]
		span_token_pos_list = [ str(pos) + "_" + span_type for pos in range(idx_start, idx_end)]


		span_sentid = dict_pos2sid[idx_start] # not implement

		# # Attribute 1:  Span-level Ambiguity
		# #print("Attribute 1:  Span-level Ambiguity")
		# span_amb_value = 0.0
		# if span_cnt in preCompute_ambSpan:
		# 	span_amb_value = preCompute_ambSpan[span_cnt][span_type]
		# dict_entity2amb[span_pos] = span_amb_value


		# Attribute 2:  Token-level Ambiguity
		#print("Attribute 2:  Token-level Ambiguity")
		token_amb_value = 0.0
		for token, token_pos  in zip(span_token_list, span_token_pos_list):
			if token.lower() in preCompute_ambToken:
				token_amb_value = preCompute_ambToken[token.lower()][span_type]
			dict_token2amb[token_pos] = token_amb_value


		#print("Attribute 3:  Token-level Ambiguity")
		# # Attribute 3:  Span-level Frequency
		# span_fre_value = 0.0
		# if span_cnt in preCompute_freqSpan:
		# 	span_fre_value = preCompute_freqSpan[span_cnt]
		# dict_entity2fre[span_pos] = span_fre_value


		# Attribute 4:  Token-level Frequency
		#print("Attribute 4:  Token-level Ambiguity")
		token_fre_value = 0.0
		for token, token_pos  in zip(span_token_list, span_token_pos_list):
			if token.lower() in preCompute_freqToken:
				token_fre_value = preCompute_freqToken[token.lower()]
			dict_token2fre[token_pos] = token_fre_value



		# # Attribute 5:  Entity Length
		# dict_entity2elen[span_pos] = span_length


		# Attribute 6:  Sentence Length
		dict_entity2slen[span_pos] = float(preCompute_slength[span_sentid])

		# # Attribute 7:  Entity Density
		# dict_entity2eden[span_pos] = preCompute_edensity[span_sentid]

		# Attribute 8:  OOV Density
		dict_entity2oden[span_pos] = float(preCompute_odensity[span_sentid])


		# Attribute 9: Tag
		dict_entity2tag[span_pos] = span_type




	return  dict_token2amb, dict_token2fre, dict_entity2slen, dict_entity2oden, dict_entity2tag



def bucketAttribute_SpecifiedBucketValue(dict_span2attVal, n_buckets, hardcoded_bucket_values):
	################       Bucketing different Attributes

	# 	hardcoded_bucket_values = [set([float(0), float(1)])]
		n_spans = len(dict_span2attVal)
		dict_attVal2span = reverseDict(dict_span2attVal)
		dict_attVal2span = sortDict(dict_attVal2span)
		dict_bucket2span = {}



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
		# [(0,), (0.1, 0.2), (0.3,0.4), (0.5, 0.6)] --> [(0,), (0,0.2), (0.2, 0.4), (0.4, 0.6)]
		dict_old2new = intervalTransformer(dict_bucket2span.keys())
		dict_bucket2span_new = {}
		for inter_list, span_list in dict_bucket2span.items():
			dict_bucket2span_new[dict_old2new[inter_list]] = span_list

		return dict_bucket2span_new


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
	print('attribute \n')
	dict_bucket2f1 = {}
	for bucket_interval, spans_true in dict_bucket2span.items():
		spans_pred = []
		print('bucket_interval: ',bucket_interval)
		if bucket_interval not in dict_bucket2span_pred.keys():
			#print(bucket_interval)
			raise ValueError("Predict Label Bucketing Errors")
		else:
			spans_pred = dict_bucket2span_pred[bucket_interval]
		f1, p, r = evaluate_chunk_level(spans_pred, spans_true)
		dict_bucket2f1[bucket_interval] = [f1, len(spans_true)]
	# print("dict_bucket2f1: ",dict_bucket2f1)
	return sortDict(dict_bucket2f1)


def getPos2SentId(test_word_sequences_sent):
	dict_pos2sid = {}
	pos = 0
	for sid, sent in enumerate(test_word_sequences_sent):
		for i in range(len(sent)):
			dict_pos2sid[pos] = sid
			pos += 1
	return dict_pos2sid


def new_metric(corpus_type, column_no, mode, pos_column, train_vocab, n_buckets, dict_hardcoded_mode,
			   fn_train, fn_test_results, fn_test,
			   fnwrite_GoldenEntityHard,
			   fnwrite_TokenHard,
			   fwrite_TokenFreq_inTrain,
			   fwrite_SpanFreq_inTrain,
			   ):


	word_sequences_train, tag_sequences_train,word_sequences_train_sent,tag_sequences_train_sent = read_data(corpus_type,fn_train,column_no=column_no,mode=mode)
	# test_word_sequences, test_trueTag_sequences,test_word_sequences_sent, test_trueTag_sequences_sent = read_data('connl03', fn_test_results, verbose=True, column_no=-1)
	print('tag_sequences_train',tag_sequences_train[:80])
	print('word_sequences_train',word_sequences_train[:80])

	test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, test_trueTag_sequences_sent = read_data(corpus_type, fn_test,column_no=column_no,mode=mode)


	mean_test_length = np.mean([len(sent) for sent in test_word_sequences_sent])
	print('mean_test_length:',mean_test_length)

	mode = ' '
	if 'crfpp' in fn_test_results:
		mode ='\t'
	print('mode',mode)
	_,test_predTag_sequences,_,test_predTag_sequences_sent = read_data(corpus_type, fn_test_results,column_no=-1,mode=mode)

	print("class_type\tf1")
	for class_type1 in list(set(test_trueTag_sequences)):
		class_type = class_type1.split('-')[1]
		f1, p, r = evaluate_each_class(test_word_sequences_sent, test_predTag_sequences_sent, test_trueTag_sequences_sent, class_type)
		print("%s\t%s"%(class_type,str(f1)) )
	# # Pre-computing
	# preCompute_ambSpan = span_entity_amb(word_sequences_train, tag_sequences_train, fnwrite_GoldenEntityHard)
	preCompute_ambToken = span_token_amb(word_sequences_train, tag_sequences_train, fnwrite_TokenHard)
	# preCompute_freqSpan = span_entity_fre(word_sequences_train, tag_sequences_train,fwrite_SpanFreq_inTrain)
	preCompute_freqToken = span_token_fre(word_sequences_train, tag_sequences_train, fwrite_TokenFreq_inTrain)
	# print('finish loading the four pkl-files...')
	max_entityLength, preCompute_edensity, preCompute_odensity, preCompute_slength = span_sent_oovDen_length(train_vocab,test_trueTag_sequences_sent,test_word_sequences_sent)
	print('finish loading the sentence level feature...')

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

	dict_token2amb, dict_token2fre, dict_entity2slen, \
	dict_entity2oden, dict_entity2tag = getAttrValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
												   preCompute_ambToken,
												   preCompute_freqToken,
												   preCompute_odensity,
												   preCompute_slength,
													)
	# dict_span2aspectVal["MF-et"] = dict_entity2amb
	dict_span2aspectVal["MF-tt"] = dict_token2amb
	# dict_span2aspectVal["F-ent"] = dict_entity2fre
	dict_span2aspectVal["F-tok"] = dict_token2fre
	# dict_span2aspectVal["R-eLen"] = dict_entity2elen
	dict_span2aspectVal["R-sLen"] = dict_entity2slen
	# dict_span2aspectVal["R-eDen"] = dict_entity2eden
	dict_span2aspectVal["R-oov"] = dict_entity2oden
	dict_span2aspectVal["R-tag"] = dict_entity2tag

	print('finish getAttrValue-true ...') # cost most time...
	dict_token2amb_pred, dict_token2fre_pred, dict_entity2slen_pred, \
	dict_entity2oden_pred, dict_entity2tag_pred= getAttrValue(test_word_sequences, test_predTag_sequences, test_word_sequences_sent,
														   preCompute_ambToken,
														   preCompute_freqToken,
														   preCompute_odensity,
														   preCompute_slength
															)
	# dict_span2aspectVal_pred["MF-et"] = dict_entity2amb_pred
	dict_span2aspectVal_pred["MF-tt"] = dict_token2amb_pred
	# dict_span2aspectVal_pred["F-ent"] = dict_entity2fre_pred
	dict_span2aspectVal_pred["F-tok"] = dict_token2fre_pred
	# dict_span2aspectVal_pred["R-eLen"] = dict_entity2elen_pred
	dict_span2aspectVal_pred["R-sLen"] = dict_entity2slen_pred
	# dict_span2aspectVal_pred["R-eDen"] = dict_entity2eden_pred
	dict_span2aspectVal_pred["R-oov"] = dict_entity2oden_pred
	dict_span2aspectVal_pred["R-tag"] = dict_entity2tag_pred

	print('finish getAttrValue-predict ...')  # cost most time...

	# def __selectBucktingFunc(func_name, func_setting, dict_obj):
	# 	if func_name == "bucketAttribute_SpecifiedBucketInterval":
	# 		return eval(func_name)(dict_obj, eval(func_setting))
	# 	elif func_name == "bucketAttribute_SpecifiedBucketValue":
	# 		if len(func_setting.split("\t"))!=2:
	# 			raise ValueError("selectBucktingFunc Error!")
	# 		n_buckets, specified_bucket_value_list = int(func_setting.split("\t")[0]), eval(func_setting.split("\t")[1])
	# 		return eval(func_name)(dict_obj, n_buckets, specified_bucket_value_list)
	# 	elif func_name == "bucketAttribute_DiscreteValue": # now the discrete value is R-tag..
	# 		if len(func_setting.split("\t"))!=2:
	# 			raise ValueError("selectBucktingFunc Error!")
	# 		tags_list = list(set(dict_obj.values()))
	# 		print("tags_list: ",tags_list)
	# 		print("len(tags_list): ",len(tags_list))
	# 		min_buckets, topK_buckets  = int(func_setting.split("\t")[0]), len(tags_list)
	# 		return eval(func_name)(dict_obj, min_buckets, topK_buckets)

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
		dict_bucket2span_pred[aspect] =  bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect], dict_bucket2span[aspect].keys())
		dict_bucket2f1[aspect]  	  = getBucketF1(dict_bucket2span[aspect],  dict_bucket2span_pred[aspect])
		aspect_names.append(aspect)
	print("aspect_names: ",aspect_names)


	return dict_bucket2f1,aspect_names


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


def compute_holistic_f1(fn_result):
	cmd = 'perl %s < %s' % (os.path.join('.', 'conlleval'), fn_result)
	msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
	msg += ''.join(os.popen(cmd).readlines())
	print("result: ",msg)
	f1 = float(msg.split('\n')[3].split(':')[-1].strip())

	return f1


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
		fn_evaluate = path_output_tensorEval + "/" + corpus_type + "-" + filename + "-5Lresults.txt"
		fwrite_evaluate = open(fn_evaluate,'w+')
		column_no = -1
		mode = ' '
		pos_column = 0
		if corpus_type == 'ptb2':
			column_no = -1
			mode = ' '
			pos_column = 1


		fn_train = path_data +task_type + "/" + corpus_type + '/train.txt'
		fn_test = path_data +task_type + "/" + corpus_type + '/test.txt'


		# fn_test = 'new_metric/analysis/new_york_example.txt'
		train_vocab 			 = path_preComputed + '/vocab-pos/' + corpus_type + '_vocab.txt'
		fnwrite_GoldenEntityHard = path_preComputed + '/pos_metric/' + corpus_type + '_rhoSpan_inTrain_fami.pkl'
		fnwrite_TokenHard 		 = path_preComputed + '/pos_metric/' + corpus_type + '_rhoToken_inTrain_fami.pkl'
		fwrite_TokenFreq_inTrain = path_preComputed + '/pos_metric/' + corpus_type + '_Token_inTrain_freq.pkl'
		fwrite_SpanFreq_inTrain  = path_preComputed + '/pos_metric/' + corpus_type + '_Span_inTrain_freq.pkl'

		# begin{compute holistic f1}
		fwrite_evaluate.write("# information\n")
		fwrite_evaluate.write("corpus type: %s\n" % corpus_type)
		fwrite_evaluate.write("model_name1: %s\n" % model_name1)
		fwrite_evaluate.write("model_name2: %s\n" % model_name2)


		fn_results_sub = fn_results[ind*2:ind*2+2]

		# begin{compute holistic f1}
		fwrite_evaluate.write("# holistic result \n")
		for fn_result, model_name in zip(fn_results_sub, model_names):
			fn_test_results = path_preComputed + "/pos_results/" + fn_result
			print('fn_test_results:',fn_test_results)
			f1 =compute_holistic_f1(fn_test_results)
			string = model_name + ": "+str(f1)
			fwrite_evaluate.write("%s\n" %(string) )
		fwrite_evaluate.write('\n')


		models_metrics = []
		stdModels_metrics = []
		for fn_result,model_name in zip(fn_results_sub,model_names):
			fn_test_results = path_preComputed + "/pos_results/" + fn_result
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
												 fnwrite_GoldenEntityHard,
												 fnwrite_TokenHard,
												 fwrite_TokenFreq_inTrain,
												 fwrite_SpanFreq_inTrain,
												 )

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



