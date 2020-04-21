path_data="./data/"
path_preComputed="./preComputed"

task_type="pos"
path_fig="pos-fig"
path_output_tensorEval="output_tensorEval/"$task_type
#path_preComputed="preComputed"
path_aspect_conf="conf.pos-aspects"
model1="sent_long"
model2="sent_1"
datasets[0]="ptb2"

# -----------------------------------------------------
#<<COMMENT
rm -fr $path_output_tensorEval/*

echo "${datasets[*]}"
python3 tensorEvaluation-pos.py \
	--path_data $path_data \
	--task_type $task_type  \
	--path_fig $path_fig \
	--data_list "${datasets[*]}"\
	--model_list $model1  $model2 \
	--path_preComputed $path_preComputed \
	--path_aspect_conf $path_aspect_conf \
	--resfile_list ptb2_CbertWnon_snonMlp_sent10_99311656_9766.txt ptb2_CbertWnon_snonMlp_sent1_16663963_9762.txt \
	--path_output_tensorEval $path_output_tensorEval

		       

cd analysis
rm ./$path_fig/$model1"-"$model2/*.results
for i in `ls ../$path_output_tensorEval`
do
	cat ../$path_output_tensorEval/$i | python3 genFig.py --path_fig $path_fig --path_bucket_range ./$path_fig/$model1"-"$model2/bucket.range --path_bucketInfo ./$path_fig/$model1"-"$model2/bucketInfo.pkl
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
#COMMENT

# # -----------------------------------------------------



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
			> tEval-pos.html


sz tEval-pos.html
tar zcvf $path_fig.tar.gz $path_fig
sz $path_fig.tar.gz
