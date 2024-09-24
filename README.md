# Design as Grammar: Pattern Filling with Graph ML

Repository Overview:

This repository contains the code and dataset used in the Design as Grammar project, focusing on pattern filling using Graph Machine Learning (Graph ML) techniques. The project involves generating patterns, converting them into graphs, embedding features, and training a custom GraphSAGE hybrid model for self-supervised learning.

# Table of Contents

    Project Description
    Folder Structure
    Requirements
    Setup Instructions
    Using the Notebooks
        Dataset Creation
        Model Training
    Results and Visualization
    Project Overview and Presentation
    Contributors
    License

# Project Description

In this project, we explore pattern filling on a grid using Graph Machine Learning. The dataset is generated using random LEGO brick placements, and the patterns are converted into DGL (Deep Graph Library) graphs. We employ a custom GraphSAGE hybrid model to predict brick placements through self-supervised learning. The key objectives of this project are to create complex pattern-filling strategies using Graph ML, embedding features for bricks and edges, and training models on the resulting graphs.
Folder Structure

    BrickLayerX_2000/: Contains the dataset of 2000 generated patterns, DGL graphs, and features in .pkl format.
    DesignGrammar_DatasetCreation_BricklayerX.ipynb: Jupyter notebook for generating patterns and creating the dataset.
    DesignGrammar_SelfSupervised_Hybrid_Model_Training.ipynb: Jupyter notebook for embedding features and training the Graph ML model.

# Requirements

To ensure compatibility, the following specific versions of libraries were used:

    PyTorch 1.13.1 (CPU version):

    bash

pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu

DGL 1.1.3 (CPU version, compatible with PyTorch 1.13.1):

bash

    pip install dgl==1.1.3

    Additional Libraries:
        NetworkX
        Matplotlib
        Pandas
        Numpy

You can install these additional libraries with:

bash

pip install networkx matplotlib pandas numpy

# Using the Notebooks
# Dataset Creation Notebook

    Purpose: The DesignGrammar_DatasetCreation_BricklayerX.ipynb notebook generates random brick patterns, converts them into graphs, and extracts features for use in training.
    Steps in the Notebook:
        Initialize the grid and random insertion points.
        Generate patterns and fill the grid with LEGO bricks.
        Convert the patterns to DGL graphs and extract node and edge features.
        Save the dataset and features as .pkl files.

# Model Training Notebook

    Purpose: The DesignGrammar_SelfSupervised_Hybrid_Model_Training.ipynb notebook loads the dataset, embeds features into the graphs, and trains the custom GraphSAGE hybrid model on the data.
    Steps in the Notebook:
        Load the DGL graphs and feature files.
        Embed node and edge features into the graphs.
        Define and train the GraphSAGE model.
        Visualize the training and validation results.
        Save the trained model and its results.

# Results and Visualization

    After training the model, the results are visualized by plotting the training/validation loss and accuracy.
    The model predictions are applied to random test patterns, and the results are displayed to show the pattern-filling efficiency of the trained GraphSAGE model.

# Project Overview and Blogpost

This repository contains the full project details for Design as Grammar. For a detailed explanation of the research, methodology, and results, you can review the our blogpost here.

In this presentation, we cover:

    Problem Statement: The challenge of modular determination using Graph ML.
    Methodology: How we structured and labeled datasets, converted patterns to graphs, and applied GraphSAGE for modular predictions.
    Dataset: The process of creating a dataset of 2000 patterns using the Legolizer algorithm.
    Model Training: How we applied self-supervised learning for node-level and graph-level predictions.
    Results and Applications: Visualization of the model’s predictions and its applications to modular design in architecture.

# Contributors

This project was developed as part of the MaCAD 23/24 final thesis project at IAAC Barcelona. The key contributors to this project are:

Mohammad Abid (mohammadabid.sada@gmail.com)
Saif Mahfouz (saif.al-islam.mahfouz@students.iaac.net)

Supervision: David Andres Leon (david.andres.leon@iaac.net)

# License

This project is licensed through IAAC Barcelona and is not open for public distribution. For any inquiries or permission requests, please contact IAAC or the project contributors.
