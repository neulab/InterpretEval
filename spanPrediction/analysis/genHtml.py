import argparse
import re
import os
import copy
import random


# dict_att2desc

def genRow(class_type, val_list, dict_att2desc):
	template_tooltip = "<a href=\"#\" data-toggle=\"tooltip\" title=\"__placeholder_desc\">__placeholder_att</a>"


	info = " <div class=\'" + class_type +"\'>"
	for val in val_list:
		if val in dict_att2desc.keys():
			val = template_tooltip.replace("__placeholder_desc", dict_att2desc[val]).replace("__placeholder_att",val)
		else:
			val = template_tooltip.replace("__placeholder_desc", "").replace("__placeholder_att",val)
		info += "<div class=\'td\'>" + str(val) + "</div>" + "\n"
	info += "</div>" + "\n"
	return info




def genRow2(class_type, val_list,val_list2):

	info = " <div class=\'" + class_type +"\'>"
	for val1,val2 in zip(val_list,val_list2):
		random_int = str(random.randint(1,100000000))
		str_hidden1 = "<button onclick=\"myFunction("+random_int+")\">Click Me</button>"+'\n'
		str_hidden2 = "<div id=\"" +random_int+"\" style=\"display: none;\">" 
		str_hidden = str_hidden1+str_hidden2
		info += "<div class=\'td\'>" + str(val1) + str_hidden+val2 +"</div>"+'\n'+"</div>"+'\n'
	info += "</div>" + "\n"
	return info





def getHolistic(file_name):
	dict_model1 = {}
	dict_model2 = {}
	fin = open(file_name,"r")
	for line in fin:
		line = line.rstrip("\n")
		data_name, val1, val2 = line.split("\t")
		dict_model1[data_name] = val1
		dict_model2[data_name] = val2
	fin.close()
	return dict_model1, dict_model2

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


parser = argparse.ArgumentParser(description='Learning tagger using neural networks')
parser.add_argument('--n_dataset', default='1', help='the number of dataset')
parser.add_argument('--n_attribute', default='ner-list.xlsx', help='model name is utilized to name the error figure.')
parser.add_argument('--bs', default='ner-list.xlsx', help='model name is utilized to name the error figure.')



parser.add_argument('--data_list', type=str, required=True, nargs='+',
					help="a list to store [[corpus_type, model_name, result_file], ]")

parser.add_argument('--model_list', type=str, required=True, nargs='+',
					help="a list to store [[corpus_type, model_name, result_file], ]")

parser.add_argument('--path_holistic_file', default='path_holistic_file', help='the number of dataset')
#parser.add_argument('--path_template_html', default='path_template_html', help='path_template_html')
parser.add_argument('--path_fig_base', default='path_fig_base', help='path_fig_base')
parser.add_argument('--path_aspect_conf', type=str, required=True, help="conf file for evaluation aspect")
parser.add_argument('--path_bucket_range', type=str, required=True, help="conf file for evaluation aspect")

args = parser.parse_args()

#excel_file = unicode(args.excel)
#concept_file = unicode(args.concept)
#bs_directory = unicode(args.bs)



dict_att2desc = {"MF-et":"span-level label ambiguity", \
                "MF-tt":"token-level label ambiguity", \
                "F-ent":"span-level frequency in training set", \
                "F-tok":"token-level frequency in training set",\
                "R-eLen":"the length of text span", \
                "R-sLen":"the length of sentence",\
                "R-eDen":"span density in a given sentence",\
                "R-oov":"oov density in a given sentence",\
                "R-tag":"span-level label", \
                "Model":"different types of models", \
                "notemz": "dataset: ontonotes-mz",\
                "notebc": "dataset: ontonotes-bc",\
                "notewb": "dataset: ontonotes-wb",\
                "notetc": "dataset: ontonotes-tc",\
                "notenw": "dataset: ontonotes-nw",\
                "conll03": "dataset: CoNLL 2003"
                }




dataset_list = list(args.data_list)
model_list =  list(args.model_list)
path_holistic_file = args.path_holistic_file
path_aspect_conf = args.path_aspect_conf

#path_template_html = args.path_template_html
path_fig_base = args.path_fig_base

path_template_html = "./template/template_visualEval.html"

model1 = model_list[0]
model2 = model_list[1]





dict_model1, dict_model2 = getHolistic(path_holistic_file)
dict_aspect_func = loadConf(path_aspect_conf)

#attribute_list = ["MF-et", "MF-tt", "F-ent", "F-tok", "MF-et", "MF-tt", "F-ent", "F-tok"]
attribute_list = list(dict_aspect_func.keys())


result_list1 = []
result_list2 = []

for data_name in dataset_list:
	result_list1.append(dict_model1[data_name])
	result_list2.append(dict_model2[data_name])



# bucket range value (it will be put into the #show hidden#)
fread_bkr = open(args.path_bucket_range, 'r')
bkr_str = fread_bkr.read()
blocks = bkr_str.split("##")
data_mn_attr_range = {}
for block in blocks[1:]:
	data_name = block.split('\n')[0]
	data_mn_attr_range[data_name.strip()] ={}
	# print("data_name: ",data_name)

	for mn_atts in block.split('@@')[1:]:
		# print("mn_atts: ",mn_atts)
		if mn_atts.strip()!='':
			for i,mn_att in enumerate(mn_atts.split('\n') ):
				if mn_att.strip()!='':
					if i==0:
						model_name = mn_att
						data_mn_attr_range[data_name.strip()][model_name.strip()] ={}
					else:
						attr_m, range1 = mn_att.split('\t')
						data_mn_attr_range[data_name.strip()][model_name.strip()][attr_m]=range1
# print("data_mn_attr_range: ",data_mn_attr_range)




#path_fig_base = ""
breakdown_fig_model1_list = []
breakdown_fig_model2_list = []
breakdown_fig_model1_range_list = []
breakdown_fig_model2_range_list = []
selfdiag_fig_model1_list = []
selfdiag_fig_model2_list = []
aidediag_fig_model12_list = []
aidediag_heatmap_fig_model12_list = []
# Breakdown file


# print('data_mn_attr_range: ',data_mn_attr_range)
for ix, data in enumerate(dataset_list):
	breakdown_fig_each_data_model1 = []
	breakdown_fig_each_data_model2 = []
	breakdown_fig_each_data_model1_range = []
	breakdown_fig_each_data_model2_range = []
	for iy, att in enumerate(attribute_list): 
		# print("attribute_list: ",attribute_list)
		# print(iy, att)
		att_range_model1 = data_mn_attr_range[data][model1][att]
		att_range_model2 = data_mn_attr_range[data][model2][att]
		breakdown_fig_each_data_model1_range.append(att_range_model1)
		breakdown_fig_each_data_model2_range.append(att_range_model1)

		breakdown_fig_each_data_model1.append(data+"-"+model1+"-breakdown-"+att+".png")
		breakdown_fig_each_data_model2.append(data+"-"+model2+"-breakdown-"+att+".png")

	breakdown_fig_model1_range_list.append(breakdown_fig_each_data_model1_range)
	breakdown_fig_model2_range_list.append(breakdown_fig_each_data_model2_range)

	breakdown_fig_model1_list.append(breakdown_fig_each_data_model1)
	breakdown_fig_model2_list.append(breakdown_fig_each_data_model2)

	selfdiag_fig_model1_list.append(data+"-"+model1+"-selfdiag"+".png")
	selfdiag_fig_model2_list.append(data+"-"+model2+"-selfdiag"+".png")


	aidediag_fig_model12_list.append(data+"-"+model1+"_"+ model2+"-aideddiag"+".png")

	aidediag_heatmap_fig_model12_list.append(data+"-"+model1+"_"+ model2+"-heatmap"+".png")







# ------------------------ Holistic Results
__placeholder_holistic_content = "<div class=\'table_self_diagnosis\'>" + "\n"

row_vallist1 = ["Model"] + [dataset for dataset in dataset_list]
row_vallist2 = [model1] + [str(res) for res in result_list1] 
row_vallist3 = [model2] + [str(res) for res in result_list2]

__placeholder_holistic_content     += genRow('tr_title', row_vallist1, dict_att2desc) \
									+ genRow('tr_plain', row_vallist2, dict_att2desc) \
									+ genRow('tr_plain', row_vallist3, dict_att2desc)
__placeholder_holistic_content += "</div>" + "\n"



# ------------------------ Breakdown Performance

__placeholder_breakdown_content = "<div class=\'table_breakdown\'>" + "\n"


__placeholder_breakdown_content += "<li><b>Model Setting:</b> <i>" + model1 + "</i></li>" + "\n"



for ind, dataset in enumerate(dataset_list):


	__placeholder_breakdown_content += "<h5> Dataset: <i> " + dataset + "</i>\n"
	__placeholder_breakdown_content += "<div class=\'table_breakdown\'>" + "\n"

	row_vallist1 = attribute_list
	row_vallist2 = [ "<img src=" + path_fig_base + val + " alt=\"image\" />" for val in breakdown_fig_model1_list[ind]]

	__placeholder_breakdown_content += genRow('tr_title', row_vallist1, dict_att2desc) + genRow2('tr', row_vallist2,breakdown_fig_model1_range_list[ind])
	__placeholder_breakdown_content += "</div>" + "</h5>" +"\n"



__placeholder_breakdown_content += "<li><b>Model Setting:</b> <i>" + model2 + "</i></li>" + "\n"

for ind, dataset in enumerate(dataset_list):

	__placeholder_breakdown_content += "<h5> Dataset: <i> " + dataset + "</i>\n"
	__placeholder_breakdown_content += "<div class=\'table_breakdown\'>" + "\n"

	row_vallist1 = attribute_list
	row_vallist2 = [ "<img src=" + path_fig_base + val + " alt=\"image\" />" for val in breakdown_fig_model2_list[ind]]

	__placeholder_breakdown_content += genRow('tr_title', row_vallist1, dict_att2desc) + genRow2('tr', row_vallist2,breakdown_fig_model2_range_list[ind])
	__placeholder_breakdown_content += "</div>" + "\n"



# ------------------------ Selfdiag Performance

__placeholder_selfdiag_content = "<li><b>Model Setting:</b> <i>" + model1 + "</i></li>" + "\n"
__placeholder_selfdiag_content +=  "<div class=\'table_self_diagnosis\'>" + "\n"

row_vallist1 = dataset_list
row_vallist2 = [ "<img src=" + path_fig_base + val + " alt=\"image\" />" for val in selfdiag_fig_model1_list]

__placeholder_selfdiag_content += genRow('tr_title', row_vallist1, dict_att2desc) + genRow('tr', row_vallist2, dict_att2desc)
__placeholder_selfdiag_content += "</div>" + "\n"  + "<br></br>"




__placeholder_selfdiag_content += "<li><b>Model Setting:</b> <i>" + model2 + "</i></li>" + "\n"
__placeholder_selfdiag_content +=  "<div class=\'table_self_diagnosis\'>" + "\n"
row_vallist1 = dataset_list
row_vallist2 = [ "<img src=" + path_fig_base + val + " alt=\"image\" />" for val in selfdiag_fig_model2_list]

__placeholder_selfdiag_content += genRow('tr_title', row_vallist1, dict_att2desc) + genRow('tr', row_vallist2, dict_att2desc)
__placeholder_selfdiag_content += "</div>" + "\n"


# ------------------------ Aideddiag Performance
__placeholder_aideddiag_content =  "<div class=\'table_aided_diagnosis\'>" + "\n"




row_vallist1 = dataset_list
row_vallist2 = [ "<img src=" + path_fig_base + val + " alt=\"image\" />" for val in aidediag_fig_model12_list]

__placeholder_aideddiag_content += genRow('tr_title', row_vallist1, dict_att2desc) + genRow('tr', row_vallist2, dict_att2desc)
__placeholder_aideddiag_content += "</div>" + "\n"


# ------------------------ Aideddiag Heatmap Performance
__placeholder_aideddiag_heatmap_content =  "<div class=\'table_aided_diagnosis\'>" + "\n"

row_vallist1 = dataset_list
row_vallist2 = [ "<img src=" + path_fig_base + val + " alt=\"image\" />" for val in aidediag_heatmap_fig_model12_list]

__placeholder_aideddiag_heatmap_content += genRow('tr_title', row_vallist1, dict_att2desc) + genRow('tr', row_vallist2, dict_att2desc)
__placeholder_aideddiag_heatmap_content += "</div>" + "\n"








fin = open(path_template_html,"r")
all_cont = fin.read()
all_cont = all_cont.replace("__placeholder_holistic_content", __placeholder_holistic_content)
all_cont = all_cont.replace("__placeholder_breakdown_content", __placeholder_breakdown_content)
all_cont = all_cont.replace("__placeholder_selfdiag_content", __placeholder_selfdiag_content)
all_cont = all_cont.replace("__placeholder_aideddiag_content", __placeholder_aideddiag_content)
all_cont = all_cont.replace("__placeholder_aideddiag_heatmap_content", __placeholder_aideddiag_heatmap_content)
all_cont =  all_cont.replace("__placeholder_aideddiag_A-B", model1 + " - " + model2)
print(all_cont)
