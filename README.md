# COMP0087 Statistical Natural Language Processing Group Project

## Project Summary
The goal is to identify groupings of publicly traded US companies based on their textual business description data, using unsupervised document-modelling approaches.
Using groupings of stocks found by Latent Dirichlet Allocation, K-Means and the Neural Variational Document Model (NVDM), we investigate the extent to which they encode intuitive and economic meaning.

## Project Structure

### Folders

`KMeans/`, `LDA/` and `nvdm/` contain code (e.g. notebooks) for initial experiments that demonstrate how each model is used. `nvdm/models/nvdm.py` contains the NVDM model definition
in PyTorch; the other 2 models are from scikit-learn.

`Edgar/` contains notebooks of initial experiments for accessing and classifying EDGAR data.
`SP500` is our S&P500 business description (BD) dataset containing text files.
Each file is a BD of a specific company.

### Root Files:

- **`NLP Project Demo.ipynb`: the demo notebook. It is expected to be uploaded and run from Google Colab.**
- `All_Models_Selecting_K.ipynb`: aggregates our training and evaluation code for all three models.
- `Final_Models.ipynb`: trains all three models with the best `K` and saves them.
- `LDA_Companies.ipynb`: data exploration and visualization experiments for LDA.
- `Visualisations.ipynb`: basic data visualizations such as word clouds and topic-word plots. 
- `S AND P.xlsx`:  spreadsheet containing additional information about the S&P500 data (Security name, GICS sector etc.) extracted from Wikipedia.
