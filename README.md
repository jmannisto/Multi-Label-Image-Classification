Multilabel Image Classification
==============================

Project Report: [Multilabel Image Classification Report](https://github.com/jmannisto/Multi-Label-Image-Classification/blob/main/docs/Hovland_Mannisto_Deep_Learning_Project.pdf) 

Final Model Predictions:[Final_Predictions.tsv](https://github.com/jmannisto/Multi-Label-Image-Classification/blob/main/docs/final_predictions.tsv)

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs                <- A default Sphinx project; see sphinx-doc.org for details
    |   |
    |   ├── Hovland_Mannisto_Deep_Learning_Project.pdf <- project report
    |   └── final_predictions.tsv <- final predictions of model in tsv format
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jm-initial-data-exploration`.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └──src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── data_extract.py <- script to convert data from text files and images to dataframe and images
        |   └── data_load.py <- script for fetching data for models (dataloader, dataset class etc.)
        │
        └── models         <- Scripts to train models and then use trained models to make
            │                 predictions
            ├── train_model.py <- general script for training and evaluating model
            └── predict_labels.py <- script for predicting labels for 
    
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
