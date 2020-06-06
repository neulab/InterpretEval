import pickle
import string
import sys



fileName = "model_data_metric_bucket_evals6.pkl"

with open(fileName,"rb") as f:
	tensorDict_list = pickle.load(f)


dict_f1 = tensorDict_list[0]
dict_precision = tensorDict_list[1]
dict_recall = tensorDict_list[2]
dict_correct_prediction = tensorDict_list[3]
dict_total_prediction = tensorDict_list[4]
dict_num_of_existing  =tensorDict_list[5]


## print f1
n_bucket = 0
n_dataet = 0
n_attribute = 0
n_bucket = 0
for model in dict_f1.keys():
	n_model = len(dict_f1.keys())
	for dataset in dict_f1[model].keys():
		n_dataset = len(dict_f1[model].keys())
		for attribute in dict_f1[model][dataset].keys():
			n_attribute = len(dict_f1[model][dataset])
			for bucket in dict_f1[model][dataset][attribute].keys():
				n_bucket = len(dict_f1[model][dataset][attribute])
				#if n_bucket == 6:
				#	print(model, dataset, attribute)
				print(dict_f1[model][dataset][attribute][bucket])


		 

print("n_model:\t", n_model)
print("n_datset:\t", n_dataset)
print("n_attribute:\t", n_attribute)
print("n_bucket:\t", n_bucket)
