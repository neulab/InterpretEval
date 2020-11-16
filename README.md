<img src="fig/nlp.png" width="800">

### by Pengfei Liu, Jinlan Fu, Graham Neubig and other contributors.


This project is a by-product of these works:

1) [Interpretable Multi-dataset Evaluation for Named Entity Recognition](https://www.aclweb.org/anthology/2020.emnlp-main.489.pdf)

2) [RethinkCWS: Is Chinese Word Segmentation a Solved Task?](https://www.aclweb.org/anthology/2020.emnlp-main.457.pdf)




## Motivated Questions

<img src="fig/ner.gif" width="550">

* #### Performance of many NLP tasks has reached a plateau. What works, and what's next?
* #### <strong>Is XX a solved task? What's left?</strong>
* #### A good evaluation metric can not only rank different systems but also tell their relative advantages (strengths and weaknesses) of them.



## Interpretable Evaluation Methodology

### Attribute definition

### Attribute value calculation

### Bucketing 

### Breakdown




## Application


### ExplainaBoard: Next Generation of LeaderBoard


### System Diagnosis
* Self-diagnosis
* Aided-diagnosis


### Dataset Bias Analysis


### Structural Bias Analysis 









## Interpreting Your Results?


### Method 1: Upload your files to  the ``ExplainaBoard`` website



### Method 2: Run it Locally
Give the Named Entity Recognition task as an example. Run the shell: `./run_task_ner.sh`.

The shell scripts include the following three aspects:

- `tensorEvaluation-ner.py` -> Calculate the dependent results of fine-grained analysis.

- `genFig.py` -> Drawing figures to show the results of the fine-grained analysis.

- `genHtml.py` -> Put the figures drawing in the previous step into the web page.

After running the above command, a web page named `tEval-ner.html` will be generated for displaying the analysis and diagnosis results of the models. 

The running process of the Chinese Word Segmentation task is similar.

```
   Notably, so far, our system only supports limited tasks and datasets, we're extending them currently!
```

Here are some generated results of preliminary evaluation systems: Named Entity Recognition (NER), Chinese Word Segmentation (CWS) and Part-of-Speech (POS).
* [NER](http://pfliu.com/tensorEvaluation/tEval-ner.html)
* [CWS](http://pfliu.com/tensorEvaluation/tEval-cws.html)
* [POS](http://pfliu.com/tensorEvaluation/tEval-pos.html)
* [Chunk](http://pfliu.com/tensorEvaluation/tEval-chunk.html)






