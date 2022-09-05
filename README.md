# Replicating notebook environments

To recreate the exact environment(s) used for these notebooks:

    conda env create -f environment.yml

You must then activate the environment:

    conda activate book

# Creating the environments manually

Note that the above environments were tested on Linux. They do not work on Macs. On Macs, please create them from scratch using the following instructions:

    conda create --name book
    conda activate book
    conda install pytorch torchvision torchaudio torchtext cpuonly -c pytorch
    conda install jupyter pandas matplotlib scikit-learn gensim nltk
    pip install conllu
    conda install -c huggingface transformers
    conda install -c huggingface -c conda-forge datasets

## Data

- [glove embeddings](https://nlp.stanford.edu/data/glove.6B.zip) trained on wikipedia 2014 and gigaword 5, from the [official website](https://nlp.stanford.edu/projects/glove/)
- [spanish glove embeddings](http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz) found on [github](https://github.com/dccuchile/spanish-word-embeddings)

Fetch the above two files using wget. Then place them in the notebooks/ folder and uncompress them.

