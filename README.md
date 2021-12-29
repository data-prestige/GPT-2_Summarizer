# Introduction
GPT-2 Fine-Tuning for Summarization on the ARTeLab dataset. The dataset is firstly unpacked into tokenized .json files which are passed in the training section.

### Built With

The following technologies, frameworks and libraries have been used:

* [Python](https://www.python.org/)
* [Git](https://git-scm.com/)

We strongly suggest to create a virtual env (i.e. 'GPT-2_Summarizer') providing the python version otherwise it will not install previous libraries:

```bash
conda create -n GPT-2_Summarizer python=3.8.9 
conda activate GPT-2_Summarizer python=3.8.9
```

If you want to run it manually you need to have python 3.8.9 (or later versions) configured on your machine. 

1. Install all the libraries using the requirements.txt files that can be found in the main repository

```bash
pip install -r requirements.txt
```


2. Dataset Creation

The dataset can be created by passing a .csv file to the "dataset_creation.py" script which expects 2 columns, text and summary respectively.

```bash
python dataset_creation.py --path_csv "./path_to_csv" --path_directory "./path_to_directory" --model "model_used_for_tokenization" 
``` 
The script will create tokenized .json files that can be fed to the "train.py" script.





'''
CONTINUARE DA QUI
A
A
A
A
A
A
A
A
A

A

'''








In order to run the system on a GPU follow the Google Colab Notebook provided. 


```bash
python main.py -ep "end_path" -p "path" -mp "model_path" -pdf "pdf_path" -u 'utent row number' -up "user_path" -c "category"
``` 
where:
* -ep folder path that will contain generated files
* -p folder path containing category folders
* -emb switch to specify whether to compute embedding of all documents
* -mp folder path that will contain model files, it can coincide with -ep path
* -pdf path to pdf file for recommendation
* -u user row number in database
* -up path containing user data
* -c category name

i.e.

```bash
python main.py -ep "C:\Desktop\recommendation_data" -p "C:\Desktop\Datasets" -emb "yes" -pdf "C:\Desktop\cool_file.pdf" -u 3000 -up "C:\Desktop\user_data.csv" -c "Fancy category"
``` 
3. Run the System






# Possible next steps

1. Configure a docker registry in order to publish docker images 
2. Code refactoring to remove unused code and polish it
3. Use Cython for Doc2Vec embedding to allow parallelization and faster performance 