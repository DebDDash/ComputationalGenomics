# cs690-Spatial-LDA

Submission made by Debarpita Dash(220328)

## Environment Information

The models were trained and inferenced on a Google Colab environment with the following specifications -

- Standard CPU (no GPU or TPU acceleration)
- Approximately 12 GB RAM
- Python v3.10.14

You can reproduce the python environment using the `requirements.txt` file. Alternately if running on Kaggle/Colab, please run the following command - `!pip install anndata scanpy numpy pandas tqdm scikit-learn matplotlib scipy annoy`. This command is specified at the beginning of each notebook as well. 

## Code

A brief description of each file is outlined below -

1. osmFish.ipynb - This notebook contains SpatialLDA performed on osmFish data.
2. Debarpita_merfish_spatialLDA.ipynb -This notebook contains SpatialLDA performed on MERFISH  data.
3. Dataset1_MERFISH.h5ad - MERFISH dataset provided for this task.
4. Dataset2_OSMFISH.h5ad - OSMFISH dataset provided for this task.
5. merfish_cluster_8.csv - Final submission files for the task on MERFISH data.
6. osmFish_cluster_11.csv - Final submission files for the task on osmFish  data.
