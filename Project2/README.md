### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

You'll need to install ### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Following packages need to be installed:
1. nltk 3.3.0
2. numpy 1.15.2
3. pandas 0.23.4
4. scikit-learn 0.20.0
5. sqlalchemy 1.2.12

## Project Motivation<a name="motivation"></a>

The objective is to analyze data from Figure Eight to build a model for an API that classifies disaster messages.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

This is the frontpage:
![Alt text](https://github.com/janierkkilae/Disaster-Response-Pipelines/blob/master/Screenshot1.PNG?raw=true "Screenshot1")

By inputting a word, you can check its category:
![Alt text](https://github.com/janierkkilae/Disaster-Response-Pipelines/blob/master/Screenshot2.PNG?raw=true "Screenshot2")

## Files

- ETL Pipeline Preparation.ipynb: Description for workspace/data/process_data.py
- ML Pipeline Preparation.ipynb: Description for workspace/model/train_classifier.py
- workspace/data/process_data.py: A data cleaning pipeline that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database
- workspace/model/train_classifier.py: A machine learning pipeline that:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file
- workspace/app/run.py: A Flask Webapp which will display classification results of a new message in several categories.

## Acknowledgements

I wish to thank [Figure Eight](https://www.figure-eight.com/) for the dataset, and thank [Udacity](https://www.udacity.com/) for advice and review. In addition, I'd like to thank Matteo Bonanomi https://github.com/matteobonanomi for providing me with a good source of reference for Flask Web App creation.
