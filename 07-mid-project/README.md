# Heart Failure Prediction

This project analysed Heart Failure Clinical Records dataset (heart_failure_clinical_records_dataset.csv, from UC Irvine Machine Learning Repository), built multiple models to predict the death events of patients with previous heart-failures, and finally deployed the best model locally and on the AWS cloud. 

The dataset contains the medical records of 299 patients with previous heart failures. Each patient profile has 13 clinical features regarding clinical, body, and lifestyle information. The goal is to build a model that predicts death events of patients, i.e. whether the patient died before the end of the follow-up period.  

`notebook.ipynb` contains scripts about data preparation, EDA, feature importance analysis, model selection and parameter tuning. For this binary classification problem, logistic regression, random forest, and xgboost have been used. `train.py` contains the scripts about the final model training using a random forest classifier. This produced the pickled model object `model.bin`.`pipenv` was used to manage dependencies (Pipfile, Pipfile.lock). The model was served locally via flask, gunicorn and Docker (see `predict.py`, `predict-test.py` and `Dockerfile`). And finally deployed on the cloud using AWS Elastic Beanstalk (`predict-test-eb.py`). Deployment and testing screenshots can be found in `eb screenshot` and `local screenshot` folders.

