# cs690-multimodal-integration

This repository contains the codebase and documentation for the final project of Group 1 in the course CS690, offered in 2024-25-I. Completed by -

- Aniruddh Pramod (210142) - Dept. of Mathematics and Statistics
- Advaith Kannan (210072) - Dept. of Biological Sciences and Bioengineering
- Debarpita Dash (220328) - Dept. of Biological Sciences and Bioengineering

## Environment Information

The models were trained and inferenced on a Kaggle T4 x2 GPU Environment whose specifications are -

- 30 GB CPU RAM
- 15 x2 GB GPU VRAM
- Python v3.10.14

It is advisable to use a separate python environment for each method. You can reproduce the python environment for scMODAL using the `requirements.txt` file. Alternately if running on Kaggle/Colab, please run the following command - `!pip install anndata scanpy numba umap-learn torch annoy gdown scikit-misc`. Similarly for MaxFuse use the command `!pip install anndata scanpy maxfuse`. These commands are specified at the beginning of each notebook as well. Additionally, the suggested method for running these notebooks is by importing them into Kaggle and not on Colab.

## Dataset Information

All datasets are publicly available. For convenience, we have included code to download the datasets at the beginning of each notebook which will download the dataset for you. The details of the datasets used are -

1. CiteSeq PBMC - The CITE-seq healthy human PBMC dataset with 228 antibody markers was obtained from [Hao et al](https://arxiv.org/abs/1610.03454). for benchmarking.
2. TeaSeq - The TEA-seq neutrophil-depleted human PBMC dataset (GSM4949911) was obtained from [from Swanson et al](https://pubmed.ncbi.nlm.nih.gov/33835024/)

## Code

Kaggle/Colab links for every notebook are provided at the top of every notebook. A brief description of each file is outlined below -

1. cca_util.ipynb - Helper code that uses CCA to generate a correspondence file
2. scMODAL_PBMC - Trains and runs scMODAL on the citeSeq PBMC dataset using the protein-gene conversion matrix
3. scMODAL_CCA_PBM - Trains and runs scMODAL on the citeSeq PBMC dataset using CCA
4. scMODAL_teaseq - Trains and runs scMODAL on trimodal data
5. MaxFuse_PBMC - Trains and runs MaxFuse on citeSeq PBMC data
