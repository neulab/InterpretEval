# Task type
task_type="ner"

# path of training data
path_data="./data/"

# path of conf file
path_aspect_conf="conf.ner-attributes"


# Part1: Dataset Name
datasets[0]="conll03"
datasets[1]="wnut16"

# Part2: Model Name
model1="Flair"
model2="ELMo"
# Part3: Path of result files
resfiles[0]="./preComputed/ner/result/connl03_CflairWglove_lstmCrf_9303.txt"
resfiles[1]="./preComputed/ner/result/connl03_CelmoWglove_lstmCrf_9222.txt"
resfiles[2]="./preComputed/ner/result/wnut16_CflairWglove_lstmCrf_27707443_4596.txt"
resfiles[3]="./preComputed/ner/result/wnut16_CelmoWglove_lstmCrf_29275447_4533.txt"


path_preComputed="./preComputed"
path_fig=$task_type"-fig"
path_output_tensorEval="output_tensorEval/"$task_type/$model1"-"$model2


# delimiter=get_value_from_frontend()

delimiter="s"

rm -fr $path_output_tensorEval/*
echo "${datasets[*]}"
python3 tensorEvaluation-ner.py \
	--path_data $path_data \
	--task_type $task_type  \
	--path_fig $path_fig \
	--data_list "${datasets[*]}"\
	--model_list $model1  $model2 \
	--path_preComputed $path_preComputed \
	--path_aspect_conf $path_aspect_conf \
	--resfile_list "${resfiles[*]}" \
	--path_output_tensorEval $path_output_tensorEval \
	--delimiter $delimiter 


		       

cd analysis
rm ./$path_fig/$model1"-"$model2/*.results
rm ./$path_fig/$model1"-"$model2/*.tex
for i in `ls ../$path_output_tensorEval`
do
	cat ../$path_output_tensorEval/$i | python3 genFig.py --path_fig $path_fig/$model1"-"$model2 --path_bucket_range ./$path_fig/$model1"-"$model2/bucket.range \
		--path_bucketInfo ./$path_fig/$model1"-"$model2/bucketInfo.pkl
done


# -----------------------------------------------------

# run pdflatex .tex
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


sz tEval-$task_type.html
tar zcvf $path_fig.tar.gz $path_fig
sz $path_fig.tar.gz
