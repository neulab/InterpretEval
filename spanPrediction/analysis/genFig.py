import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import string
import sys
import os
import argparse
import pickle

def extValue(cont, fr, to):
    return cont.split(fr)[-1].split(to)[0] 



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.ax.tick_params(labelsize=25)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False,labelsize=30, pad=15)
    legend = ax.legend()
    legend.remove()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, va="center",
             rotation_mode="anchor")

    #plt.setp(ax.get_yticklabels(), rotation=-90, ha="right", visible=True, rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", va = "center", visible=True, rotation_mode="anchor")
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar



def getTuple(tuple_obj):
    # print('tuple_obj: ',tuple_obj)
    if len(tuple_obj) == 1:
        return str(tuple_obj[0]).replace("_","").replace("0.",".")
    else:
        left, right = str(tuple_obj[0]).replace("0.","."), str(tuple_obj[1]).replace("0.",".")
        # print("")
        return "(" + left + ", " + right + "]"



def genHistTex(dict_breakdown, path_template, path_tex_base, corpus_type, model_type, dict_attr_range, dict_interval2F1number):

    for att_name, val_list in dict_breakdown.items():

        #  path_tex_selfdiag = path_tex_base + corpus_type + "-" + model_type + "-"+ "selfdiag.tex"
        path_tex_breakdown = path_tex_base + corpus_type + "-" + model_type + "-" + "breakdown-"+att_name +".tex"



        xticklabels = ""
        xticklabels_n_spans = ""        
        plot1 = ""
        t_id = 1
        num_list = []
        xticklabel_list = dict_attr_range[att_name].split(" ")
        dict_bucket2pair = dict_interval2F1number[att_name]



        #print("xticklabel_list:---------",xticklabel_list)
        for idx, val_pair in enumerate(val_list.split(" ")):
            xticklabel, val1 = val_pair.split(":")

            xticklabel = getTuple(eval(xticklabel_list[idx].split(":")[1]))

            interval = list(dict_bucket2pair.keys())[idx]
            F1, n_span = dict_bucket2pair[interval]
            xticklabels_n_spans += "\\textcolor{orange}{"+str(n_span)+"}, "
            #print("xticklabels_n_spans:\t-------------------" + xticklabels_n_spans)


            f_val1 = float('%.2f'%(float(val1)*100))
            xticklabels += "\\textcolor{red}{"+xticklabel+"}, "
            plot1 += "\t" + str(t_id) + "\t" + str(f_val1) + "\n"
            t_id  += 1





            num_list.append(f_val1)


        xticklabels = xticklabels.rstrip(", ")
        plot1 = plot1.rstrip("\n")
        xmax = str(t_id)
        ymin = max(min(num_list)-2.0,0)
        ymax = max(num_list)
        #delta = (ymax-ymin)/t_id
        ymin = str(ymin)
        ymax = str(ymax)
        width = t_id * 0.6 + 3
        height = width*0.95


        fin = open(path_template,"r")
        text_template = fin.read()
        text_template = text_template.replace("__placeholder_xticklabels", xticklabels)
        text_template = text_template.replace("__placeholder_span_xticklabels", xticklabels_n_spans)
        text_template = text_template.replace("__placeholder_plot1", plot1)
        text_template = text_template.replace("__placeholder_xmax", xmax)
        text_template = text_template.replace("__placeholder_ymin", ymin)
        text_template = text_template.replace("__placeholder_ymax", ymax)
        text_template = text_template.replace("__placeholder_width", str(width))
        text_template = text_template.replace("__placeholder_height",str(height))
        fin.close()

        fout=open(path_tex_breakdown,"w")
        fout.write(text_template+"\n")
        fout.close()




def genSelfDiagTex(dict_breakdown, path_template, path_tex_base, corpus_type, model_type, dict_attr_range):

    dict_name2id = {"XS":0, "S":1, "L":2, "XL":3}


    xticklabels = ""
    plot1 = ""
    plot2 = ""
    t_id = 1
    num_list_min = []
    num_list_max = []
    for att_name, val_list in dict_breakdown.items():
        xticklabel_list = dict_attr_range[att_name].split(" ")



        min_pair, max_pair, delta = val_list.split(" ")
        min_ind, min_val = min_pair.split(":")
        max_ind, max_val = max_pair.split(":")


        min_bucket_id = int(min_ind)
        #min_bucket_id = dict_name2id[min_ind]
        min_xticklabel = getTuple(eval(xticklabel_list[min_bucket_id].split(":")[1]))

        max_bucket_id = int(max_ind)
        #max_bucket_id = dict_name2id[max_ind]
        max_xticklabel = getTuple(eval(xticklabel_list[max_bucket_id].split(":")[1]))


        f_val1 = float('%.2f'%(float(min_val)*100)) 
        f_val2 = float('%.2f'%(float(delta)*100))
        f_val_max = float('%.2f'%(float(max_val)*100))
        #xticklabels += "\\textcolor{red}{"+ att_name + " \\" +"\\ " + min_xticklabel + " \\" +"\\ " + max_xticklabel +"}, "
        #xticklabels += att_name + " \\" +"\\ " + "\\textcolor{red}{"+ min_xticklabel + "}"  + " \\" +"\\ " + "\\textcolor{blue}{" + max_xticklabel +"}, "
        xticklabels += att_name + " \\" +"\\ " + "\\textcolor{brinkpink}{" + max_xticklabel +"}" + " \\" +"\\ " + "\\textcolor{cyan}{"+ min_xticklabel + "}, "
        plot1 += "\t" + str(t_id) + "\t" + str(f_val1) + "\n"
        plot2 += "\t" + str(t_id) + "\t" + str(f_val2) + "\n"
        t_id  += 1
        num_list_min.append(f_val1)
        num_list_max.append(f_val_max)
        if t_id ==len(dict_breakdown): # delete the R-tag, for the self-diagnosis
            break



    xticklabels = xticklabels.rstrip(", ")
    plot1 = plot1.rstrip("\n")
    plot2 = plot2.rstrip("\n")
    xmax = str(t_id)
    ymin = max(min(num_list_min)-5,0)
    ymax = max(num_list_max)
    delta = (ymax-ymin)/t_id
    ymin = str(ymin)
    ymax = str(ymax)
    # ymin = str(ymin/3.0)
    # ymax = str(ymax+0.5*delta)
    width = t_id * 0.6 + 1.5 + 10
    height = width*0.8



    fin = open(path_template,"r")
    text_template = fin.read()
    text_template = text_template.replace("__placeholder_xticklabels", xticklabels)
    text_template = text_template.replace("__placeholder_plot1", plot1)
    text_template = text_template.replace("__placeholder_plot2", plot2)
    text_template = text_template.replace("__placeholder_xmax", xmax)
    text_template = text_template.replace("__placeholder_ymin", ymin)
    text_template = text_template.replace("__placeholder_ymax", ymax)
    text_template = text_template.replace("__placeholder_width", str(width))
    text_template = text_template.replace("__placeholder_height",str(height))
    fin.close()


    path_tex_selfdiag = path_tex_base + corpus_type + "-" + model_type + "-"+ "selfdiag.tex"
    fout=open(path_tex_selfdiag,"w")
    fout.write(text_template+"\n")
    fout.close()


def genAidedDiagHist(dict_breakdown, path_template, path_tex_base, corpus_type, model_type, dict_attr_range):


    dict_name2id = {"XS":0, "S":1, "L":2, "XL":3, "N":2}
    xticklabels = ""
    plot1 = ""
    plot2 = ""
    t_id = 1
    num_list = []
    for att_name, val_list in dict_breakdown.items():
        xticklabel_list = dict_attr_range[att_name].split(" ")

        min_pair, max_pair = val_list.split(" ")
        min_ind, min_val = min_pair.split(":")
        max_ind, max_val = max_pair.split(":")


        #min_bucket_id = dict_name2id[min_ind]
        min_bucket_id = int(min_ind)
        min_xticklabel = getTuple(eval(xticklabel_list[min_bucket_id].split(":")[1]))

        #max_bucket_id = dict_name2id[max_ind]
        max_bucket_id = int(max_ind)
        max_xticklabel = getTuple(eval(xticklabel_list[max_bucket_id].split(":")[1]))


        
        f_val1 = float('%.2f'%(float(min_val)*100))  
        f_val2 = float('%.2f'%(float(max_val)*100))  
        xticklabels += att_name + " \\" +"\\ " + "\\textcolor{brinkpink}{" + max_xticklabel +"}" + " \\" +"\\ " + "\\textcolor{cyan}{"+ min_xticklabel + "}, "
        plot1 += "\t" + str(t_id) + "\t" + str(f_val1) + "\n"
        plot2 += "\t" + str(t_id) + "\t" + str(f_val2) + "\n"
        t_id  += 1
        num_list.append(f_val1)
        num_list.append(f_val2)
        if t_id ==len(dict_breakdown): # delete the R-tag, for the aided-diagnosis
            break



    xticklabels = xticklabels.rstrip(", ")
    plot1 = plot1.rstrip("\n")
    plot2 = plot2.rstrip("\n")
    xmax = str(t_id)
    ymin = min(num_list)
    ymax = max(num_list)
    delta = (ymax-ymin)/t_id
    ymin = str(ymin+ymin/5.0) # it always a minus value
    ymax = str(ymax+0.5*delta)
    width = t_id * 0.6 + 1.5 + 10
    height = width*0.8



    fin = open(path_template,"r")
    text_template = fin.read()
    text_template = text_template.replace("__placeholder_xticklabels", xticklabels)
    text_template = text_template.replace("__placeholder_plot1", plot1)
    text_template = text_template.replace("__placeholder_plot2", plot2)
    text_template = text_template.replace("__placeholder_xmax", xmax)
    text_template = text_template.replace("__placeholder_ymin", ymin)
    text_template = text_template.replace("__placeholder_ymax", ymax)
    text_template = text_template.replace("__placeholder_width", str(width))
    text_template = text_template.replace("__placeholder_height",str(height))
    fin.close()


    path_tex_selfdiag = path_tex_base + corpus_type + "-"+ model_type + "-" + "aideddiag.tex"
    fout=open(path_tex_selfdiag,"w")
    fout.write(text_template+"\n")
    fout.close()


def padList(list_2d):
    max_bucket = max([len(val_list) for val_list in list_2d])
    n_att = len(list_2d)

    res_mat = np.zeros((n_att,max_bucket))
    for i in range(n_att):
        res_mat[i,:len(list_2d[i])] = list_2d[i]
    return range(max_bucket), res_mat



def genAidedDiagHeatmap(dict_aided_diag_heatmap, path_fig_base, corpus_type, model_type):


    ind = 0
    row_name_list = []
    col_name_list = []
    mat = []
    for att_name, val_list in dict_aided_diag_heatmap.items():
        if att_name == "R-tag":   # remove the R-tag attribute
            continue
        else:
            row_name_list.append(att_name)
            num_list = [float(val) for val in val_list.split(" ")]
            mat.append(num_list)
   
    col_name_list, mat_array = padList(mat)
    #print("mat_array: ",mat_array)

    print("row_name_list: ",row_name_list)    
    print("col_name_list: ",col_name_list)   

    fig, ax = plt.subplots()
    im, cbar = heatmap(mat_array, row_name_list, col_name_list, ax=ax, vmin=-0.03, vmax=0.03,cmap="PiYG")
    #                   cmap="PiYG", cbarlabel="Fine-grained Evaluation")

    '''
    im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                       cmap="Blues", cbarlabel="harvest [t/year]")
    '''
    #texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    #cbar.remove()
    fig.tight_layout()

    path_heatmap_aideddiag = path_fig_base + corpus_type + "-"+ model_type + "-" + "heatmap.png"
    plt.savefig(path_heatmap_aideddiag, bbox_inches = 'tight',
                pad_inches = 0)


def str2dict(str_info):
    dict_breakdown = {}
    for line in str_info.split("\n"):
        info_list = line.strip().split("\t")
        #print(info_list)
        if len(info_list) <=1:
            continue
        att_name = info_list[0]
        val_list = info_list[1]
        dict_breakdown[att_name]=val_list
    return dict_breakdown


def getBuctName(path_bucket_range):
    # bucket range value (it will be put into the #show hidden#)
    fread_bkr = open(path_bucket_range, 'r')
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
    fread_bkr.close()
    return data_mn_attr_range






parser = argparse.ArgumentParser(description='Learning tagger using neural networks')
parser.add_argument('--path_fig', type=str, required=True, 
                        help="the type of the task")
parser.add_argument('--path_bucket_range', type=str, required=True, help="conf file for evaluation aspect")
parser.add_argument('--path_bucketInfo', type=str, required=True, help="conf file for evaluation aspect")


args = parser.parse_args()


path_bucket_range = args.path_bucket_range
data_mn_attr_range = getBuctName(path_bucket_range)


dict_data_bucketInfo = pickle.load(open(args.path_bucketInfo, "rb"))





holistic_res_m1 = ""
dict_breakdown_m1 = {}
dict_self_diag_m1 = {}



holistic_res_m2 = ""
dict_breakdown_m2 = {}
dict_self_diag_m2 = {}

dict_aided_diag_hist_m1_2 = {}
dict_aided_diag_heatmap_m1_2 = {}

corpus_type =""
model_name1 =''
model_name2 =''


holistic_res = []

all_cont =  sys.stdin.read()


for block in all_cont.split("# "):
    if block.find("information") !=-1:
        #print('block:',block)
        corpus_type = extValue(block, "corpus type: ", "\n")
        model_name1 = extValue(block, "model_name1: ", "\n")
        model_name2 = extValue(block, "model_name2: ", "\n")

    if block.find("holistic result") != -1:
        holistic_res_m1 = extValue(block, model_name1+": ", "\n")
        holistic_res_m2 = extValue(block, model_name2+": ", "\n")
        holistic_res = [holistic_res_m1,holistic_res_m2]

    elif block.find("break-down performance") != -1:
        metaInfo_m1 = extValue(block, model_name1+":\n", "\n\n")
        metaInfo_m2 = extValue(block, model_name2+":\n", "\n\n")
        dict_breakdown_m1 = str2dict(metaInfo_m1)
        dict_breakdown_m2 = str2dict(metaInfo_m2)


    elif block.find("self-diagnosis") != -1:
        metaInfo_m1 = extValue(block, model_name1+":\n", "\n\n")
        metaInfo_m2 = extValue(block, model_name2+":\n", "\n\n")
        dict_self_diag_m1 = str2dict(metaInfo_m1)
        dict_self_diag_m2 = str2dict(metaInfo_m2)

    elif block.find("aided-diagnosis line-chart") != -1:
        metaInfo_m1_2 = extValue(block, model_name1+"_"+model_name2+ ":\n", "\n\n")
        dict_aided_diag_hist_m1_2 = str2dict(metaInfo_m1_2)




    elif block.find("aided-diagnosis heatmap") != -1:
        metaInfo_m1_2 = extValue(block, model_name1+"_"+model_name2+ ":\n", "\n\n")
        dict_aided_diag_heatmap_m1_2 = str2dict(metaInfo_m1_2)



def build_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)








## Debug: dict check
print("corpus_type: ",corpus_type)
print("model_name1: ",model_name1)
print("model_name2: ",model_name2)
print("dict_breakdown_m1: ",dict_breakdown_m1)
print("dict_self_diag_m1: ",dict_self_diag_m1)
print("dict_breakdown_m2: ",dict_breakdown_m2)
print("dict_self_diag_m2: ",dict_self_diag_m2)
print("dict_aided_diag_hist_m1_2: ",dict_aided_diag_hist_m1_2)
print("dict_aided_diag_heatmap_m1_2: ",dict_aided_diag_heatmap_m1_2)



path_tex_base = "./"+args.path_fig+"/" + model_name1 + '-' + model_name2 + '/'


fout = open(path_tex_base + "holistic.results","a+")
fout.write(corpus_type + "\t" + holistic_res[0] + "\t" + holistic_res[1] + "\n")
fout.close()







## Setting of Path
path_template_breakdown = "./template/template_stack_one.tex"
path_tex_base_breakdown = path_tex_base
build_dir(path_tex_base_breakdown)

path_template_selfdiag = "./template/template_stack_two.tex"
path_tex_base_selfdiag = path_tex_base
build_dir(path_tex_base_selfdiag)

path_tex_base_aideddiag = path_tex_base
build_dir(path_tex_base_aideddiag)

path_fig_base = path_tex_base
build_dir(path_fig_base)




## Tex or Fig generation
genHistTex(dict_breakdown_m1, path_template_breakdown, path_tex_base_breakdown, corpus_type, model_name1, data_mn_attr_range[corpus_type][model_name1], dict_data_bucketInfo[corpus_type])
genHistTex(dict_breakdown_m2, path_template_breakdown, path_tex_base_breakdown, corpus_type, model_name2, data_mn_attr_range[corpus_type][model_name2], dict_data_bucketInfo[corpus_type])

genSelfDiagTex(dict_self_diag_m1, path_template_selfdiag, path_tex_base_selfdiag, corpus_type, model_name1, data_mn_attr_range[corpus_type][model_name1])
genSelfDiagTex(dict_self_diag_m2, path_template_selfdiag, path_tex_base_selfdiag, corpus_type, model_name2, data_mn_attr_range[corpus_type][model_name1])



genAidedDiagHist(dict_aided_diag_hist_m1_2, path_template_selfdiag, path_tex_base_aideddiag, corpus_type, model_name1 + "_" + model_name2, data_mn_attr_range[corpus_type][model_name1])
genAidedDiagHeatmap(dict_aided_diag_heatmap_m1_2, path_fig_base, corpus_type, model_name1 + "_" + model_name2)






