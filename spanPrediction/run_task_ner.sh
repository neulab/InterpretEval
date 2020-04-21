path_data="./data/"
path_preComputed="./preComputed"

task_type="ner"
path_fig="ner-fig"
path_output_tensorEval="output_tensorEval/"$task_type
path_aspect_conf="conf.ner-aspects"
model1="CNN"
model2="LSTM"
datasets[0]="notemz"
datasets[1]="conll03"
# datasets[0]="conll03"
# datasets[1]="wnut16"
# datasets[2]="notebn"
# datasets[3]="notemz"
# -----------------------------------------------------
#<<COMMENT
# rm -fr $path_output_tensorEval/*

echo "${datasets[*]}"
python3 tensorEvaluation-ner.py \
	--path_data $path_data \
	--task_type $task_type  \
	--path_fig $path_fig \
	--data_list "${datasets[*]}"\
	--model_list $model1  $model2 \
	--path_preComputed $path_preComputed \
	--path_aspect_conf $path_aspect_conf \
	--resfile_list notemz_CcnnWglove_lstmCrf_37176065_8539.txt notemz_CelmoWnone_lstmCrf_49178345_8632.txt \
					conll03_CbertWnon_snonMlp_sent10_09906966_long2short_9108.txt conll03_CbertWnon_snonMlp_sent1_40900668_9077.txt \
	--path_output_tensorEval $path_output_tensorEval 
	#> log.tensorEvaluation
	

#COMMENT
# # python3 tensorEvaluation.py \
# # 	--path_data $path_data \
# # 	--task_type $task_type  \
# # 	--data_list "${datasets[*]}"\
# # 	--model_list $model1  $model2 \
# # 	--path_preComputed $path_preComputed \
# # 	--path_aspect_conf $path_aspect_conf \
# # 	--resfile_list conll03_CbertWnon_snonMlp_sent10_09906966_long2short_9108.txt conll03_CbertWnon_snonMlp_sent1_40900668_9077.txt \
# # 	               wnut16_CelmoWglove_lstmCrf_29275447_4533.txt  wnut16_CelmoWnone_lstmCrf_10716009_4456.txt \
# # 		       notebn_CcnnWglove_cnnCrf_75792533_8642.txt  notebn_CcnnWglove_lstmCrf_31653759_8678.txt \
# # 		       notemz_CcnnWglove_lstmCrf_37176065_8539.txt notemz_CelmoWnone_lstmCrf_49178345_8632.txt \
# # 	--path_output_tensorEval $path_output_tensorEval
		       

cd analysis
rm ./$path_fig/$model1"-"$model2/*.results
rm ./$path_fig/$model1"-"$model2/*.tex
for i in `ls ../$path_output_tensorEval`
do
	cat ../$path_output_tensorEval/$i | python3 genFig.py --path_fig $path_fig --path_bucket_range ./$path_fig/$model1"-"$model2/bucket.range \
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

# # -----------------------------------------------------

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
			> tEval-ner.html


sz tEval-ner.html
tar zcvf $path_fig.tar.gz $path_fig
sz $path_fig.tar.gz
