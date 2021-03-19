# Task type
task_type="ner"
base_dir="./data/"


# path of conf file
path_aspect_conf=$base_dir$task_type"/conf."$task_type"-attributes"


# Part1: Dataset Name:  you should put your training and test set in the directory, for example, ./data/conll03/ and name them as train.txt, test.txt, like this:
# ./data/ner/conll03/data/train.txt
# ./data/ner/conll03/data/txt.txt
datasets[0]="conll03"



# Part2: Model Name： Two model should be input. If only one, just copy it twice. e.g., model1 = "lstm"  model2 = "lstm"
model1="Flair"
model2="ELMo"

# Part3: Path of result files
# ./data/ner/conll03/results/connl03_CflairWglove_lstmCrf_9303.txt
# ./data/ner/conll03/results/connl03_CelmoWglove_lstmCrf_9222.txt
resfiles[0]=$base_dir$task_type"/conll03/results/connl03_CflairWglove_lstmCrf_9303.txt"  # output of model1;  Format:  token true_label pred_label  (delimiter = " ")
resfiles[1]=$base_dir$task_type"/conll03/results/connl03_CelmoWglove_lstmCrf_9222.txt"   # output of model2   Format:  token true_label pred_label  (delimiter = " ")





path_preComputed=$base_dir$task_type"/preComputed"
path_fig=$task_type"-fig"
path_output_tensorEval="output_tensorEval/"$task_type/$model1"-"$model2


# delimiter=get_value_from_frontend()

#delimiter="s" # suggested

rm -fr $path_output_tensorEval/*
echo "${datasets[*]}"
python3 tensorEvaluation-ner.py \
	--path_data $base_dir \
	--task_type $task_type  \
	--path_fig $path_fig \
	--data_list "${datasets[*]}"\
	--model_list $model1  $model2 \
	--path_preComputed $path_preComputed \
	--path_aspect_conf $path_aspect_conf \
	--resfile_list "${resfiles[*]}" \
	--path_output_tensorEval $path_output_tensorEval \
#	--delimiter $delimiter


		       

cd analysis
if [ -f "./$path_fig/$model1"-"$model2/*.results" ]; then
  rm ./$path_fig/$model1"-"$model2/*.results
fi
if [ -f "./$path_fig/$model1"-"$model2/*.tex" ]; then
  rm ./$path_fig/$model1"-"$model2/*.tex
fi


for i in `ls ../$path_output_tensorEval`
do
	cat ../$path_output_tensorEval/$i | python3 genFig.py --path_fig $path_fig/$model1"-"$model2 --path_bucket_range ./$path_fig/$model1"-"$model2/bucket.range \
		--path_bucketInfo ./$path_fig/$model1"-"$model2/bucketInfo.pkl
done


#-----------------------------------------------------

#run pdflatex .tex
cd $path_fig
cd $model1"-"$model2
find=".tex"
replace=""

for i in `ls *.tex`
do
	file=${i//$find/$replace}
	echo $file
	pdflatex $file.tex > log.latex
	pdftoppm -png $file.pdf > $file.png
done

# -----------------------------------------------------

echo "begin to generate html ..."

rm -fr *.aux *.log *.fls *.fdb_latexmk *.gz
cd ../../
# cd analysis
echo "####################"
python3 genHtml.py 	--data_list ${datasets[*]} \
			--model_list $model1  $model2 \
			--path_fig_base ./$path_fig/$model1"-"$model2/ \
			--path_holistic_file ./$path_fig/$model1"-"$model2/holistic.results \
			--path_aspect_conf ../$path_aspect_conf \
			--path_bucket_range ./$path_fig/$model1"-"$model2/bucket.range \
			> tEval-$task_type.html


#sz tEval-$task_type.html
#tar zcvf $path_fig.tar.gz $path_fig
#sz $path_fig.tar.gz
