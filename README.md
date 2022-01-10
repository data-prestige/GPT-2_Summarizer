# Introduction
GPT-2 Fine-Tuning for Summarization on the ARTeLab dataset. The dataset is firstly unpacked into tokenized .json files which are passed to the training section.

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

3. Training

In order to run the training process on a GPU follow the Google Colab Notebook provided as

bash_train_GPT2.ipynb

Remember to change the Runtime to GPU !


4. Loading and using the model

Once the training has stopped and the best model has been saved you can load it and use it for summarization by running the script below on your terminal.

```bash
python loading_saved_model.py --text "Sarebbe stato molto facile per l'uomo estrarre la freccia dalla carne del malcapitato, eppure questo si rivelò complicato e fatale. La freccia aveva infatti penetrato troppo a fondo nella gamba e aveva provocato una terribile emorragia." --saved_model "./model_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.bin"
```

















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