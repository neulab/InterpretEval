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
from sklearn.metrics import f1_score
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


def read_data(corpus_type, fn, column_no=-1, delimiter =' '):
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

		strings = line.split(delimiter)
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





#   getAspectValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, dict_precomputed_path)

def getAspectValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
				   test_trueTag_sequences_sent, dict_preComputed_path, dict_aspect_func):


	def getSententialValue(test_trueTag_sequences_sent, test_word_sequences_sent, dict_oov):

		eDen = []
		oDen = []
		sentLen = []

		for i, test_sent in enumerate(test_trueTag_sequences_sent):
			pred_chunks = set(get_chunks(test_sent))

			num_entityToken = 0
			for pred_chunk in pred_chunks:
				idx_start = pred_chunk[1]
				idx_end = pred_chunk[2]
				num_entityToken += idx_end - idx_start

			# introduce the entity token density in sentence ...
			eDen.append(float(num_entityToken) / len(test_sent))

			# introduce the oov density in sentence ...
			num_oov = 0
			if dict_oov is not None:
				for word in test_word_sequences_sent[i]:
					if word not in dict_oov:
						num_oov += 1
			oDen.append(float(num_oov) / len(test_sent))

			# introduce the sentence length in sentence ...
			sentLen.append(len(test_sent))

		return eDen, oDen, sentLen



	dict_preComputed_model = {}
	for aspect, path in dict_preComputed_path.items():
		print("path:\t"+path)
		if os.path.exists(path):
			print('load the hard dictionary of entity span in test set...')
			fread = open(path, 'rb')
			dict_preComputed_model[aspect] = pickle.load(fread)
		else:
			raise ValueError("can not load hard dictionary" + aspect + "\t" + path)




	dict_span2aspectVal = {}
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}

	eDen_list, oDen_list, sentLen_list = [], [], []
	if "oDen" in dict_aspect_func.keys():
		eDen_list, oDen_list, sentLen_list = getSententialValue(test_trueTag_sequences_sent,
																	 test_word_sequences_sent,
																	 dict_preComputed_model.get("oDen", None))


	dict_pos2sid = getPos2SentId(test_word_sequences_sent)
	all_chunks = get_chunks(test_trueTag_sequences)

	for span_info in all_chunks:

		span_type = span_info[0].lower()
		#print(span_type)
		idx_start = span_info[1]
		idx_end = span_info[2]
		span_cnt = ' '.join(test_word_sequences[idx_start:idx_end]).lower()
		span_pos = str(idx_start) + "_" + str(idx_end) + "_" + span_type

		span_length = idx_end - idx_start

		span_token_list = test_word_sequences[idx_start:idx_end]
		span_token_pos_list = [ str(pos) + "_" + span_type for pos in range(idx_start, idx_end)]


		span_sentid = dict_pos2sid[idx_start]


		# Span-level Ambiguity: tamb_span

		aspect = "eCon"
		if aspect in dict_aspect_func.keys():
			preCompute_ambSpan = dict_preComputed_model[aspect]
			span_amb_value = 0.0
			if span_cnt in preCompute_ambSpan:
				if span_type.lower() in preCompute_ambSpan[span_cnt]:
					span_amb_value = preCompute_ambSpan[span_cnt][span_type]
			dict_span2aspectVal[aspect][span_pos] = span_amb_value


		# Token-level Ambiguity: tamb_token
		aspect = "tCon"
		if aspect in dict_aspect_func.keys():
			preCompute_ambToken = dict_preComputed_model[aspect]
			token_amb_value = 0.0
			for token, token_pos  in zip(span_token_list, span_token_pos_list):
				if token.lower() in preCompute_ambToken:
					if span_type.lower() in  preCompute_ambToken[token.lower()]:
						token_amb_value = preCompute_ambToken[token.lower()][span_type]
				dict_span2aspectVal[aspect][token_pos] = token_amb_value


		# Span-level Frequency: fre_span
		aspect = "eFre"
		if aspect in dict_aspect_func.keys():
			preCompute_freqSpan = dict_preComputed_model[aspect]
			span_fre_value = 0.0
			if span_cnt in preCompute_freqSpan:
				span_fre_value = preCompute_freqSpan[span_cnt]
			dict_span2aspectVal[aspect][span_pos] = span_fre_value


		# Token-level Frequency: fre_token
		aspect = "tFre"
		if aspect in dict_aspect_func.keys():
			preCompute_freqToken = dict_preComputed_model[aspect]
			token_fre_value = 0.0
			for token, token_pos  in zip(span_token_list, span_token_pos_list):
				if token.lower() in preCompute_freqToken:
					token_fre_value = preCompute_freqToken[token.lower()]
				dict_span2aspectVal[aspect][token_pos] = token_fre_value



		# Entity Length: eLen
		aspect = "eLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = span_length

		# Sentence Length: sentLen
		aspect = "sLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = float(sentLen_list[span_sentid])

		# Entity Density: eDen
		aspect = "eDen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = float(eDen_list[span_sentid])

		# OOV Density: oDen
		aspect = "oDen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = float(oDen_list[span_sentid])

		# Tag: tag
		aspect = "tag"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = span_type

		#print(dict_span2aspectVal)
	return  dict_span2aspectVal


def read_data_re(path_file):
	sample_list_sent = []
	sample_list_word = []
	sample_list_tag = []
	sample_list_span = []

	fin = open(path_file, "r")
	all_samples = fin.read()

	for sample_id, sample in enumerate(all_samples.split("\n\n")):

		# print(sample)
		# print(sample.split("\n"))
		# print(len(sample.split("\n")))
		# exit()
		if len(sample.split("\n")) != 3:
			continue
		sentence, tag, comment = sample.split("\n")[0], sample.split("\n")[1], sample.split("\n")[2]
		tag = tag.replace("(e2,e1)","")
		sentence = sentence.split("\t")[1]

		# _, sentence, tag, _ = sample.split("\t")
		entity1 = extValue(sentence, "<e1>", "</e1>")
		entity2 = extValue(sentence, "<e2>", "</e2>")

		sentence = sentence.rstrip(".\"").lstrip("\"").replace("<e1>", "").replace("</e1>", "").replace("<e2>",
																										"").replace(
			"</e2>", "")
		ent1_pos = sentence.find(entity1)
		ent2_pos = sentence.find(entity2)

		if ent1_pos == -1 or ent2_pos == -1 or entity1 == "" or entity2 == "":
			continue

		# for each entity: sample_id, entity, tag, sent_start, sent_end

		span_info1 = (sample_id, entity1, tag, ent1_pos, ent1_pos + len(entity1) - 1)
		span_info2 = (sample_id, entity2, tag, ent2_pos, ent2_pos + len(entity2) - 1)

		sample_list_sent.append(sentence)
		sample_list_word.append(sentence.split(" "))
		sample_list_tag.append(tag)
		sample_list_span.append([span_info1, span_info2])
		return sample_list_sent, sample_list_word, sample_list_span, sample_list_tag


def tuple2str(triplet):
    res = ""
    for v in triplet:
        res += str(v) + "_"
    return res.rstrip("_")



def read_data_re_test(path_file, col):
	sample_list_sent = []
	sample_list_word = []
	sample_list_tag = []
	sample_list_span = []

	fin = open(path_file, "r")
	all_samples = fin.read()
	#col = -1

	for sample_id, sample in enumerate(all_samples.split("\n")):

		# if len(sample.split("\t")) != 3:
		# 	continue
		sentence = sample.split("\n")[0]
		tag = sample.split("\t")[col]
		tag = tag.replace("(e2,e1)", "")

		# _, sentence, tag, _ = sample.split("\t")
		entity1 = extValue(sentence, "<e1>", "</e1>")
		entity2 = extValue(sentence, "<e2>", "</e2>")

		sentence = sentence.rstrip(".\"").lstrip("\"").replace("<e1>", "").replace("</e1>", "").replace("<e2>",
																										"").replace(
			"</e2>", "")
		ent1_pos = sentence.find(entity1)
		ent2_pos = sentence.find(entity2)

		if ent1_pos == -1 or ent2_pos == -1 or entity1 == "" or entity2 == "":
			continue

		# for each entity: sample_id, entity, tag, sent_start, sent_end

		span_info1 = (sample_id, entity1, tag, ent1_pos, ent1_pos + len(entity1) - 1)
		span_info2 = (sample_id, entity2, tag, ent2_pos, ent2_pos + len(entity2) - 1)

		sample_list_sent.append(sentence)
		sample_list_word.append(sentence.split(" "))
		sample_list_tag.append(tag)
		sample_list_span.append([span_info1, span_info2])
	return sample_list_sent, sample_list_word, sample_list_span, sample_list_tag










def read_data_atc(path_file, col):
	sample_list_sent = []
	sample_list_word = []
	sample_list_tag = []
	sample_list_span = []

	fin = open(path_file, "r")
	#all_samples = fin.read()
	#col = -1

	for sample_id, sample in enumerate(fin):

		sample = sample.rstrip("\n")

		# if len(sample.split("\t")) != 4:
		# 	continue
		span = sample.split("\t")[0]
		sentence = sample.split("\t")[1]
		tag = sample.split("\t")[col]

		# print("----sample------")
		# print(sample)
		# print(col)
		# print(sample.split("\t")[col])
		# exit()



		span_pos = sentence.find(span)


		if span_pos == -1:
			continue

		# for each entity: sample_id, entity, tag, sent_start, sent_end

		span_info = (sample_id, span, tag, span_pos, span_pos + len(span) - 1)

		sample_list_sent.append(sentence)
		sample_list_word.append(sentence.split(" "))
		sample_list_tag.append(tag)
		sample_list_span.append(span_info)
	return sample_list_sent, sample_list_word, sample_list_span, sample_list_tag




def getAspectValue_atc(sample_list_sent, sample_list_word, sample_list_span, sample_list_tag, dict_preComputed_path, dict_aspect_func):



	dict_preComputed_model = {}
	oDen_list, sentLen_list = [], []

	for aspect, path in dict_preComputed_path.items():
		print("path:\t"+path)
		if os.path.exists(path):
			print('load the hard dictionary of entity span in test set...')

			if aspect == "tCon":
				fread = open(path, 'rb')
				dict_preComputed_model["tCon"], dict_preComputed_model["tFre"] = pickle.load(fread)
			if aspect == "eCon":
				fread = open(path, 'rb')
				dict_preComputed_model["eCon"], dict_preComputed_model["eFre"] = pickle.load(fread)
			if aspect == "oDen":
				fread = open(path, 'rb')
				_, oDen_list, sentLen_list = pickle.load(fread)

		else:
			raise ValueError("can not load hard dictionary" + aspect + "\t" + path)




	dict_span2aspectVal = {}
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}


	#print("-----sample_list_sent-----")
	#print(sample_list_sent)

	sample_id = 0
	for  sent, word_list, spanInfo, tag in zip(sample_list_sent, sample_list_word, sample_list_span, sample_list_tag):

		sent_length = len(word_list)
		sent_pos = tuple2str((sample_id, tag))
		# for each span (entity): sample_id, entity, tag, sent_start, sent_end
		span_length = len(spanInfo[1].split(" "))
		span_pos = tuple2str(spanInfo)

		# Sentence Length: sentLen
		aspect = "sentLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["sentLen"][sent_pos] = float(sent_length)



		# Tag: tag
		aspect = "tag"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["tag"][sent_pos] = tag



		# span-length: eLen
		aspect = "eLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = span_length




		sample_id += 1



	return  dict_span2aspectVal




def new_metric_atc(corpus_type, delimiter, column_info,
			   task2funcName,
			   dict_aspect_func,
			   dict_precomputed_path,
			   fn_train,
			   fn_test_results
			   ):

	#column_true_tag_train, column_true_tag_test, column_pred_tag_test = column_info

	#sample_list_sent, sample_list_word, sample_list_span, sample_list_tag = read_data_atc(fn_train, -1)
	sample_list_sent, sample_list_word, sample_list_span, sample_list_tag = read_data_atc(fn_test_results, -2)
	sample_list_sent, sample_list_word, sample_list_span_pred, sample_list_tag_pred = read_data_atc(fn_test_results, -1)








	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}




	dict_span2aspectVal      = eval(task2funcName)(sample_list_sent, sample_list_word, sample_list_span, sample_list_tag, dict_precomputed_path, dict_aspect_func)
	dict_span2aspectVal_pred = eval(task2funcName)(sample_list_sent, sample_list_word, sample_list_span_pred, sample_list_tag_pred, dict_precomputed_path, dict_aspect_func)





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



	return dict_bucket2f1, aspect_names






def read_data_tc(path_file, col):
	sample_list_sent = []
	sample_list_word = []
	sample_list_tag = []
	sample_list_span = []

	fin = open(path_file, "r")
	#all_samples = fin.read()
	#col = -1

	#for sample_id, sample in enumerate(all_samples.split("\n")):
	for sample_id, sample in enumerate(fin):

		sample = sample.rstrip("\n")
		sentence = sample.split("\t")[0]
		tag = sample.split("\t")[col]






		#span_pos = sentence.find(span)


		# if span_pos == -1:
		# 	continue

		# for each entity: sample_id, entity, tag, sent_start, sent_end

		#span_info = (sample_id, span, tag, span_pos, span_pos + len(span) - 1)

		sample_list_sent.append(sentence)
		sample_list_word.append(sentence.split(" "))
		sample_list_tag.append(tag)
		# sample_list_span.append(span_info)
	#print(sample_list_sent)
	return sample_list_sent, sample_list_word, sample_list_span, sample_list_tag




def getAspectValue_tc(sample_list_sent, sample_list_word, sample_list_span, sample_list_tag, dict_preComputed_path, dict_aspect_func):



	dict_preComputed_model = {}
	oDen_list, sentLen_list = [], []

	for aspect, path in dict_preComputed_path.items():
		print("path:\t"+path)
		if os.path.exists(path):
			print('load the hard dictionary of entity span in test set...')

			if aspect == "tCon":
				fread = open(path, 'rb')
				dict_preComputed_model["tCon"], dict_preComputed_model["tFre"] = pickle.load(fread)
			if aspect == "eCon":
				fread = open(path, 'rb')
				dict_preComputed_model["eCon"], dict_preComputed_model["eFre"] = pickle.load(fread)
			if aspect == "oDen":
				fread = open(path, 'rb')
				_, oDen_list, sentLen_list = pickle.load(fread)

		else:
			raise ValueError("can not load hard dictionary" + aspect + "\t" + path)




	dict_span2aspectVal = {}
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}


	#print("-----sample_list_sent-----")
	#print(sample_list_sent)

	sample_id = 0
	for  sent, word_list, tag in zip(sample_list_sent, sample_list_word, sample_list_tag):

		sent_length = len(word_list)
		sent_pos = tuple2str((sample_id, tag))
		# for each span (entity): sample_id, entity, tag, sent_start, sent_end
		# span_length = len(spanInfo[1].split(" "))
		# span_pos = tuple2str(spanInfo)

		# Sentence Length: sentLen
		aspect = "sentLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["sentLen"][sent_pos] = float(sent_length)



		# Tag: tag
		aspect = "tag"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["tag"][sent_pos] = tag



		# # span-length: eLen
		# aspect = "eLen"
		# if aspect in dict_aspect_func.keys():
		# 	dict_span2aspectVal[aspect][span_pos] = span_length




		sample_id += 1


	#print(dict_span2aspectVal)
	return  dict_span2aspectVal




def new_metric_tc(corpus_type, delimiter, column_info,
			   task2funcName,
			   dict_aspect_func,
			   dict_precomputed_path,
			   fn_train,
			   fn_test_results
			   ):

	#column_true_tag_train, column_true_tag_test, column_pred_tag_test = column_info

	#sample_list_sent, sample_list_word, sample_list_span, sample_list_tag = read_data_atc(fn_train, -1)
	sample_list_sent, sample_list_word, sample_list_span, sample_list_tag           = read_data_tc(fn_test_results, -2)
	sample_list_sent, sample_list_word, sample_list_span_pred, sample_list_tag_pred = read_data_tc(fn_test_results, -1)








	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}







	dict_span2aspectVal      = eval(task2funcName)(sample_list_sent, sample_list_word, sample_list_span, sample_list_tag, dict_precomputed_path, dict_aspect_func)
	dict_span2aspectVal_pred = eval(task2funcName)(sample_list_sent, sample_list_word, sample_list_span_pred, sample_list_tag_pred, dict_precomputed_path, dict_aspect_func)





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



	return dict_bucket2f1, aspect_names



def getAspectValue_re(sample_list_sent, sample_list_word, sample_list_span, sample_list_tag, dict_preComputed_path, dict_aspect_func):



	dict_preComputed_model = {}
	oDen_list, sentLen_list = [], []

	for aspect, path in dict_preComputed_path.items():
		print("path:\t"+path)
		if os.path.exists(path):
			print('load the hard dictionary of entity span in test set...')

			if aspect == "tCon":
				fread = open(path, 'rb')
				dict_preComputed_model["tCon"], dict_preComputed_model["tFre"] = pickle.load(fread)
			if aspect == "eCon":
				fread = open(path, 'rb')
				dict_preComputed_model["eCon"], dict_preComputed_model["eFre"] = pickle.load(fread)
			if aspect == "oDen":
				fread = open(path, 'rb')
				_, oDen_list, sentLen_list = pickle.load(fread)

		else:
			raise ValueError("can not load hard dictionary" + aspect + "\t" + path)




	dict_span2aspectVal = {}
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}


	#print("-----sample_list_sent-----")
	#print(sample_list_sent)

	sample_id = 0
	for  sent, word_list, spanInfo_list, tag in zip(sample_list_sent, sample_list_word, sample_list_span, sample_list_tag):

		sent_length = len(word_list)
		sent_pos = tuple2str((sample_id, tag))
		span1_info, span2_info = spanInfo_list[0], spanInfo_list[1]
		# for each span (entity): sample_id, entity, tag, sent_start, sent_end
		spanDistance = span2_info[3] - span1_info[3]
		span1_length = len(span1_info[1].split(" "))
		span2_length = len(span2_info[1].split(" "))
		span1_pos = tuple2str(span1_info)
		span2_pos = tuple2str(span2_info)


		# Sentence Length: sentLen
		aspect = "sentLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["sentLen"][sent_pos] = float(sent_length)

		# Span Distance: spanDis
		aspect = "spanDis"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["spanDis"][sent_pos] = float(spanDistance)


		# Tag: tag
		aspect = "tag"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["tag"][sent_pos] = tag



		# span-length: eLen
		aspect = "eLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span1_pos] = span1_length
			dict_span2aspectVal[aspect][span2_pos] = span2_length




		sample_id += 1


	#print("-----dict_span2aspectVal-----")
	#print(dict_span2aspectVal)
	#exit()
	return  dict_span2aspectVal




def new_metric_re(corpus_type, delimiter, column_info,
			   task2funcName,
			   dict_aspect_func,
			   dict_precomputed_path,
			   fn_train,
			   fn_test_results
			   ):

	#column_true_tag_train, column_true_tag_test, column_pred_tag_test = column_info

	#sample_list_sent, sample_list_word, sample_list_span, sample_list_tag = read_data_re(fn_train)
	sample_list_sent, sample_list_word, sample_list_span, sample_list_tag = read_data_re_test(fn_test_results, -2)
	sample_list_sent, sample_list_word, sample_list_span_pred, sample_list_tag_pred = read_data_re_test(fn_test_results, -1)








	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}







	dict_span2aspectVal      = eval(task2funcName)(sample_list_sent, sample_list_word, sample_list_span, sample_list_tag, dict_precomputed_path, dict_aspect_func)
	dict_span2aspectVal_pred = eval(task2funcName)(sample_list_sent, sample_list_word, sample_list_span_pred, sample_list_tag_pred, dict_precomputed_path, dict_aspect_func)





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



	return dict_bucket2f1, aspect_names









def read_data_match(path_file, col):

	sample_list_sent1 = []
	sample_list_sent2 = []
	sample_list_word1 = []
	sample_list_word2 = []
	sample_list_tag = []

	fin = open(path_file, "r")
	#all_samples = fin.read()
	#col = -1

	#for sample_id, sample in enumerate(all_samples.split("\n")):
	for sample_id, sample in enumerate(fin):

		sample = sample.rstrip("\n")

		sent1 = sample.split("\t")[0]
		sent2 = sample.split("\t")[1]
		tag = sample.split("\t")[col]




		sample_list_sent1.append(sent1)
		sample_list_sent2.append(sent2)
		sample_list_word1.append(sent1.split(" "))
		sample_list_word2.append(sent2.split(" "))
		sample_list_tag.append(tag)
		# sample_list_span.append(span_info)
	#print(sample_list_sent)
	return sample_list_sent1, sample_list_sent2, sample_list_word1, sample_list_word2, sample_list_tag




def getAspectValue_match(sample_list_sent1, sample_list_sent2, sample_list_word1, sample_list_word2, sample_list_tag, dict_preComputed_path, dict_aspect_func):



	dict_preComputed_model = {}
	oDen_list, sentLen_list = [], []

	for aspect, path in dict_preComputed_path.items():
		print("path:\t"+path)
		if os.path.exists(path):
			print('load the hard dictionary of entity span in test set...')

			if aspect == "tCon":
				fread = open(path, 'rb')
				dict_preComputed_model["tCon"], dict_preComputed_model["tFre"] = pickle.load(fread)
			if aspect == "eCon":
				fread = open(path, 'rb')
				dict_preComputed_model["eCon"], dict_preComputed_model["eFre"] = pickle.load(fread)
			if aspect == "oDen":
				fread = open(path, 'rb')
				_, oDen_list, sentLen_list = pickle.load(fread)

		else:
			raise ValueError("can not load hard dictionary" + aspect + "\t" + path)




	dict_span2aspectVal = {}
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}


	#print("-----sample_list_sent-----")
	#print(sample_list_sent)

	sample_id = 0
	for  sent1, sent2, word_list1, word_list2, tag in zip(sample_list_sent1, sample_list_sent2, sample_list_word1, sample_list_word2, sample_list_tag):




		sent1_length = len(word_list1)
		sent2_length = len(word_list2)
		sent_pos = tuple2str((sample_id, tag))





		# Sentence Length: sentALen
		aspect = "sentALen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["sentALen"][sent_pos] = float(sent1_length)


		# Sentence Length: sentBLen
		aspect = "sentBLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["sentBLen"][sent_pos] = float(sent2_length)


		# The difference of sentence length: senDeltaLen
		aspect = "senDeltaLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["senDeltaLen"][sent_pos] = float(sent1_length-sent2_length)


		# Tag: tag
		aspect = "tag"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["tag"][sent_pos] = tag




		sample_id += 1


	#print(dict_span2aspectVal)
	return  dict_span2aspectVal




def new_metric_match(corpus_type, delimiter, column_info,
			   task2funcName,
			   dict_aspect_func,
			   dict_precomputed_path,
			   fn_train,
			   fn_test_results
			   ):

	#column_true_tag_train, column_true_tag_test, column_pred_tag_test = column_info

	#sample_list_sent, sample_list_word, sample_list_span, sample_list_tag = read_data_atc(fn_train, -1)
	sample_list_sent1, sample_list_sent2, sample_list_word1, sample_list_word2, sample_list_tag      = read_data_match(fn_test_results, -2)
	sample_list_sent1, sample_list_sent2, sample_list_word1, sample_list_word2, sample_list_tag_pred = read_data_match(fn_test_results, -1)








	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}







	dict_span2aspectVal      = eval(task2funcName)(sample_list_sent1, sample_list_sent2, sample_list_word1, sample_list_word2, sample_list_tag, dict_precomputed_path, dict_aspect_func)
	dict_span2aspectVal_pred = eval(task2funcName)(sample_list_sent1, sample_list_sent2, sample_list_word1, sample_list_word2, sample_list_tag_pred, dict_precomputed_path, dict_aspect_func)





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



	return dict_bucket2f1, aspect_names






def getAspectValue_cws(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
				   test_trueTag_sequences_sent, dict_preComputed_path, dict_aspect_func):



	dict_preComputed_model = {}
	oDen_list, sentLen_list = [], []

	for aspect, path in dict_preComputed_path.items():
		print("path:\t"+path)
		if os.path.exists(path):
			print('load the hard dictionary of entity span in test set...')

			if aspect == "tCon":
				fread = open(path, 'rb')
				dict_preComputed_model["tCon"], dict_preComputed_model["tFre"] = pickle.load(fread)
			if aspect == "eCon":
				fread = open(path, 'rb')
				dict_preComputed_model["eCon"], dict_preComputed_model["eFre"] = pickle.load(fread)
			if aspect == "oDen":
				fread = open(path, 'rb')
				_, oDen_list, sentLen_list = pickle.load(fread)

		else:
			raise ValueError("can not load hard dictionary" + aspect + "\t" + path)




	dict_span2aspectVal = {}
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}




	dict_pos2sid = getPos2SentId(test_word_sequences_sent)
	all_chunks = get_chunks(test_trueTag_sequences)

	for span_info in all_chunks:

		span_type = span_info[0].lower()
		#print(span_type)
		idx_start = span_info[1]
		idx_end = span_info[2]
		span_cnt = ' '.join(test_word_sequences[idx_start:idx_end]).lower()
		span_pos = str(idx_start) + "_" + str(idx_end) + "_" + span_type

		span_length = idx_end - idx_start

		span_token_list = test_word_sequences[idx_start:idx_end]
		span_token_pos_list = [ str(pos) + "_" + span_type for pos in range(idx_start, idx_end)]


		span_sentid = dict_pos2sid[idx_start]


		# Span-level Ambiguity: tamb_span

		aspect = "eCon"
		if aspect in dict_aspect_func.keys():
			preCompute_ambSpan = dict_preComputed_model[aspect]
			span_amb_value = 0.0
			if span_cnt in preCompute_ambSpan:
				if span_type.lower() in preCompute_ambSpan[span_cnt]:
					span_amb_value = preCompute_ambSpan[span_cnt][span_type]
			dict_span2aspectVal[aspect][span_pos] = span_amb_value


		# Token-level Ambiguity: tamb_token
		aspect = "tCon"
		if aspect in dict_aspect_func.keys():
			preCompute_ambToken = dict_preComputed_model[aspect]
			token_amb_value = 0.0
			for token, token_pos  in zip(span_token_list, span_token_pos_list):
				if token.lower() in preCompute_ambToken:
					if span_type.lower() in  preCompute_ambToken[token.lower()]:
						token_amb_value = preCompute_ambToken[token.lower()][span_type]
				dict_span2aspectVal[aspect][token_pos] = token_amb_value


		# Span-level Frequency: fre_span
		aspect = "eFre"
		if aspect in dict_aspect_func.keys():
			preCompute_freqSpan = dict_preComputed_model[aspect]
			span_fre_value = 0.0
			if span_cnt in preCompute_freqSpan:
				span_fre_value = preCompute_freqSpan[span_cnt]
			dict_span2aspectVal[aspect][span_pos] = span_fre_value


		# Token-level Frequency: fre_token
		aspect = "tFre"
		if aspect in dict_aspect_func.keys():
			preCompute_freqToken = dict_preComputed_model[aspect]
			token_fre_value = 0.0
			for token, token_pos  in zip(span_token_list, span_token_pos_list):
				if token.lower() in preCompute_freqToken:
					token_fre_value = preCompute_freqToken[token.lower()]
				dict_span2aspectVal[aspect][token_pos] = token_fre_value



		# Entity Length: eLen
		aspect = "eLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["eLen"][span_pos] = span_length



		# Sentence Length: sentLen
		aspect = "sLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["sLen"][span_pos] = float(sentLen_list[span_sentid])


		# Entity Density: eDen
		aspect = "eDen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["eDen"][span_pos] = float(eDen_list[span_sentid])

		# OOV Density: oDen
		aspect = "oDen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["oDen"][span_pos] = float(oDen_list[span_sentid])

		# Tag: tag
		aspect = "tag"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal["tag"][span_pos] = span_type


	return  dict_span2aspectVal






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







def bucketAttribute_DiscreteValue(dict_span2attVal = None, n_buckets = 100000000, n_entities = 1):
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


def new_metric(corpus_type, delimiter, column_info,
			   task2funcName,
			   dict_aspect_func,
			   dict_precomputed_path,
			   fn_train,
			   fn_test_results
			   ):

	column_true_tag_train, column_true_tag_test, column_pred_tag_test = column_info


	word_sequences_train, tag_sequences_train, word_sequences_train_sent, tag_sequences_train_sent \
		= read_data(corpus_type, fn_train, column_no = column_true_tag_train, delimiter = delimiter)

	test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, test_trueTag_sequences_sent \
		= read_data(corpus_type, fn_test_results, column_no = column_true_tag_test, delimiter = delimiter)


	_, test_predTag_sequences, _, test_predTag_sequences_sent = \
		read_data(corpus_type, fn_test_results, column_no= column_pred_tag_test, delimiter = delimiter )



	mean_test_length = np.mean([len(sent) for sent in test_word_sequences_sent])
	print('mean_test_length:',mean_test_length)

	delimiter = ' '
	if 'crfpp' in fn_test_results:
		delimiter ='\t'
	print('delimiter',delimiter)
	print('column_no: ',column_no)






	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}







	dict_span2aspectVal      = eval(task2funcName)(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, test_trueTag_sequences_sent, dict_precomputed_path, dict_aspect_func)
	dict_span2aspectVal_pred = eval(task2funcName)(test_word_sequences, test_predTag_sequences, test_word_sequences_sent, test_trueTag_sequences_sent, dict_precomputed_path, dict_aspect_func)





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



	return dict_bucket2f1, aspect_names





# def draw_f1_fig(model_names,metric_values,metric_name,corpus_type):
# 	print('metric_name',metric_name)
# 	print('metric_values',metric_values)
# 	label_list = metric_values[0].keys()
# 	# print('label_list',label_list)
# 	num_lists=[]
# 	for metric_dic in metric_values:
# 		num_list =[]
# 		for key,value in metric_dic.items():
# 			num_list.append(value)
# 		num_lists.append(num_list)



# 	plt.figure(figsize=(15, 8))
# 	width = 0.08
# 	total_width, n = width * len(model_names), len(model_names)


# 	# total_width, n = 0.2*len(model_names), len(model_names)
# 	# width = total_width / n
# 	x = np.arange(len(num_lists[0])) + 1
# 	x = x - (total_width - width)



# 	print('x',x)
# 	"""
# 	绘制条形图
# 	left:长条形中点横坐标
# 	height:长条形高度
# 	width:长条形宽度，默认值0.8
# 	label:为后面设置legend准备
# 	"""
# 	rects = []
# 	colors = ['#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#71AD47', '#264478', '#9E480D', '#636363', '#997300',
# 					  '#255E91', '#43682B', '#698ED0', '#F1975A', '#B7B7B7', '#FFCD32', '#8CC168', '#8CC168',
# 				'#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#71AD47', '#264478', '#9E480D', '#636363', '#997300',
# 			  '#255E91', '#43682B', '#698ED0', '#F1975A', '#B7B7B7', '#FFCD32', '#8CC168', '#8CC168',
# 				'#ED7D31', '#A5A5A5','#FFC000', '#5B9BD5', '#71AD47', '#264478', '#9E480D', '#636363', '#997300',
# 			  '#255E91', '#43682B', '#698ED0', '#F1975A', '#B7B7B7', '#FFCD32', '#8CC168', '#8CC168'
# 			  ]
# 	# if metric_name == 'phoSpan_block_acc':
# 	for j,(num_list,model_name,color) in enumerate(zip(num_lists,model_names,colors)):
# 		print('metric_name',metric_name)
# 		print('num_list',num_list)
# 		print('j',j)
# 		num_list = [float(y) for y in num_list]
# 		# rects.append(plt.bar(left=[xx+j*0.4 for xx in x], height=num_list, width=0.4, color=color, label=model_name))
# 		rects.append(plt.bar(x+j*width, num_list, width=width,color=color, label=model_name))
# 		# rects.append(plt.bar(np.arange(len(num_list)) +0.4*j, num_list, width=0.4, color=color,label=model_name))



# 	# for j,(num_list,model_name,color) in enumerate(zip(deta_highss,model_names[1:],colors)):
# 	# 	print('metric_name',metric_name)
# 	# 	print('num_list',num_list)
# 	# 	print('j',j)
# 	# 	num_list = [float(y) for y in num_list]
# 	# 	# rects.append(plt.bar(left=[xx+j*0.4 for xx in x], height=num_list, width=0.4, color=color, label=model_name))
# 	# 	rects.append(plt.bar(x+j*width, num_list, width=width,color=color, label=model_name))
# 	# 	# rects.append(plt.bar(np.arange(len(num_list)) +0.4*j, num_list, width=0.4, color=color,label=model_name))


# 	plt.ylim(0, 100)  # y轴取值范围
# 	if metric_name =='sent_MaskedToken_bertp_accu' or metric_name =='sent_length_block_acc' \
# 			or metric_name =='entityToken_density_block_acc' or metric_name =='oov_density_block_acc':
# 		plt.ylim(50, 100)
# 	plt.ylabel("Accuracy")
# 	"""
# 	设置x轴刻度显示值
# 	参数一：中点坐标
# 	参数二：显示值
# 	"""
# 	plt.xticks(x+0.5*n*width,label_list)
# 	plt.xlabel("Baskets")
# 	plt.title(metric_name)
# 	plt.legend()  # 设置题注
# 	# # 编辑文本
# 	# for rects1 in rects:
# 	# 	for rect in rects1:
# 	# 		height = rect.get_height()
# 	# 		plt.text(rect.get_x() + rect.get_width() / 2.0, height + 1, str(height), ha="center", va="bottom")

# 	plt.show()
# 	save_path = 'new_metric/draw_f1/' + corpus_type + '_'  +metric_name +'_'+ 'f1_0917_2310.pdf'

# 	plt.savefig(save_path)
# 	plt.close()

def return_idx_range(attr_ranges,index):
	idx_range =''
	if index== 10000:
		idx_range = str(attr_ranges[0])
	else:
		idx_range = str(attr_ranges[index])

	return idx_range

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



	min_sml_str2 =''
	max_sml_str2 = ''
	return min_sml_str2,max_sml_str2,return_string_list,return_deta_heatmap_list


def return_4bucket_xsticks(index, metric_name,dic_tag_idx):
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
	
def compute_holistic_f1_re(path, delimiter = "\t"):
	fin = open(path, "r")
	true_list = []
	pred_list = []
	for line in fin:
		line = line.rstrip()
		true_list.append(line.split("\t")[-2])
		pred_list.append(line.split("\t")[-1])
	f1 = f1_score(true_list, pred_list, average='micro')
	# print(true_list[0:10])
	# print(pred_list[0:10])
	# print("------f1-----------")
	# print(f1)
	#exit()
	return f1


def compute_holistic_f1(fn_result, delimiter = " "):
	if not os.path.isfile(fn_result):
		raise FileNotFoundError(f'result file not found: {fn_result}')
	if delimiter == " ":
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
		is_preComputed = extValue(block, "is_preComputed:\t", "\n").rstrip(" ")
		dict_aspect_func[notation] = (func_type, func_setting, is_preComputed)
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
			if metric_name!='tag':
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
	parser.add_argument('--delimiter', type=str, required=True, 
      help="the type of the interval")

	


	args = parser.parse_args()

	print('args.data_list', args.data_list[0].split(" "))  ######### note the split(" ") for data_list
	print('args.model_list', args.model_list)
	print('args.resfile_list', args.resfile_list[0].split(" "))

	if len(args.model_list) * len(args.data_list[0].split(" ")) != len(args.resfile_list[0].split(" ")):
		raise ValueError('Lengs of the args.model_list, args.data_list, and args.resfile_list must be the same.s ')

	corpus_types = args.data_list[0].split(" ")
	model_names = args.model_list
	fn_results = args.resfile_list[0].split(" ")
	model_name1 = model_names[0]
	model_name2 = model_names[1]


	path_data = args.path_data
	task_type = args.task_type
	path_output_tensorEval = args.path_output_tensorEval
	path_preComputed = args.path_preComputed
	path_aspect_conf = args.path_aspect_conf

	dict_aspect_func = loadConf(path_aspect_conf)
	metric_names = list(dict_aspect_func.keys())
	print("dict_aspect_func: ", dict_aspect_func)
	print(dict_aspect_func)

	spears = []
	radar_xticks =[]
	stds = []
	corpus_attrs = []
	entity_norepeats = []



	# ---------------------------------------------------------------- -----------------------------------------
	def ensureDir(f):
	    if not os.path.exists(f):
	        os.makedirs(f)

	filename = '-'.join(model_names)
	fn_write_buckect_value = 'analysis/'+args.path_fig+'/'+filename+'/bucket.range'
	print('fn_write_buckect_value: ',fn_write_buckect_value)

	ensureDir(fn_write_buckect_value.replace('/bucket.range',""))
	fwrite_buckect_value = open(fn_write_buckect_value, 'w')

	dict_bucket_info = {}
	fn_write_buckect_info = 'analysis/'+args.path_fig+'/'+filename+'/bucketInfo.pkl'


	for ind, corpus_type in enumerate(corpus_types):
		fn_evaluate = path_output_tensorEval + "/" + corpus_type + "-" + filename + "-5Lresults.txt"

		ensureDir(path_output_tensorEval)
		fwrite_evaluate = open(fn_evaluate,'w+')
		column_no = -1
		# delimiter = ' '
		delimiter = args.delimiter
		pos_column = 0

		fn_train = path_data +task_type + "/" + corpus_type +'/train.txt'
		fn_test  = path_data +task_type + "/" + corpus_type +'/test.txt'


		# get preComputed paths from conf file
		dict_preComputed_path = {}
		for aspect, func in dict_aspect_func.items():
			is_preComputed = func[2].lower()
			if is_preComputed == "yes":
				dict_preComputed_path[aspect] = path_preComputed + "/" + task_type + '/metric/' + corpus_type + '_' + aspect + ".pkl"



		if corpus_type == 'conll03':
			column_no = -2
			delimiter = ' '
			pos_column = 1
		elif corpus_type == 'wnut16':
			column_no = -1
			delimiter = '\t'
			pos_column = 0
		elif corpus_type in ['notebn', 'notebc','notewb','notemz','notenw','notetc']:
			column_no = 3
			delimiter = ' '
			pos_column = 2


		# Task-dependent functions for Aspect-value Caculation
		dict_task2func = {}
		dict_task2func["ner"] = "getAspectValue"
		dict_task2func["chunk"] = "getAspectValue"
		dict_task2func["pos"] = "getAspectValue"
		dict_task2func["cws"] = "getAspectValue_cws"
		dict_task2func["re"] = "getAspectValue_re"
		dict_task2func["atc"] = "getAspectValue_atc"
		dict_task2func["tc"] = "getAspectValue_tc"
		dict_task2func["match"] = "getAspectValue_match"

		dict_task2newMetric = {}
		dict_task2newMetric["ner"] = "new_metric"
		dict_task2newMetric["chunk"] = "new_metric"
		dict_task2newMetric["pos"] = "new_metric"
		dict_task2newMetric["cws"] = "new_metric"
		dict_task2newMetric["re"] = "new_metric_re"
		dict_task2newMetric["atc"] = "new_metric_atc"
		dict_task2newMetric["tc"] = "new_metric_tc"
		dict_task2newMetric["match"] = "new_metric_match"
		# Task-dependent formats for training and test results files.
		delimiter = " "
		column_info = [-1, -2, -1]


		if task_type+"_"+ corpus_type in set(["ner_conll03", "ner_wnut16", "pos_ptb2", "chunk_conll00"]):
			column_info = [-1, -2, -1] # [x, y, z] : column_true_tag_train, column_true_tag_test, column_pred_tag_test
			delimiter = " "
		elif task_type+"_"+ corpus_type in set(['ner_notebn', 'ner_notebc','ner_notewb','ner_notemz','ner_notenw','ner_notetc']):
			column_info = [3, -2, -1]
			delimiter = " "

		print("column_info")
		print(column_info)





		# begin{compute holistic f1}
		fwrite_evaluate.write("# information\n")
		fwrite_evaluate.write("corpus type: %s\n" % corpus_type)
		fwrite_evaluate.write("model_name1: %s\n" % model_name1)
		fwrite_evaluate.write("model_name2: %s\n" % model_name2)


		fn_results_sub = fn_results[ind*2:ind*2+2]


		fwrite_evaluate.write("# holistic result \n")
		for fn_result, model_name in zip(fn_results_sub, model_names):
			#fn_test_results = path_preComputed + "/ner_results/" + fn_result
			fn_test_results = fn_result
			print('fn_test_results:',fn_test_results)
			if task_type in ["re", "tc", "atc", "match"]:
				f1 = compute_holistic_f1_re(fn_test_results, "\t")
			else:
				f1 = compute_holistic_f1(fn_test_results, " ")
			string = model_name + ": "+str(f1)
			fwrite_evaluate.write("%s\n" %(string) )
		fwrite_evaluate.write('\n')


		tokenDen_f1 =[]
		oDen_f1 =[]
		sentProb_f1 =[]
		sentLen_f1 = []
		entityLen_f1 =[]
		rhoSpan_f1 =[]
		label_f1 =[]
		eFreq_f1 =[]
		phoToken_acc =[]
		morpho_acc =[]
		pos_acc =[]
		tokenFreq_acc = []


		models_metrics = []
		stdModels_metrics = []



		for fn_result, model_name in zip(fn_results_sub, model_names):

			fn_test_results = fn_result
			print('fn_test_results', fn_test_results)


			dict_bucket2f1,aspect_names = eval(dict_task2newMetric[task_type])(corpus_type,
															delimiter,
													 		column_info,
													 		dict_task2func[task_type],
															dict_aspect_func,
													        dict_preComputed_path,
															fn_train,
															fn_test_results
															 )
			print("@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@")
			print("@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@")
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


