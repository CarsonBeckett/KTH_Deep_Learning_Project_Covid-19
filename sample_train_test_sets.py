import pandas as pd
import shutil
import os
from pathlib import Path
import numpy as np


def sample_train_test_sets():
	metadata = "./covid-chestxray-dataset/metadata.csv"
	imageDir = "./covid-chestxray-dataset/images"  # Directory of images

	metadata_csv = pd.read_csv(metadata)

	covid = metadata_csv.loc[(metadata_csv['finding'] == "COVID-19") & ((metadata_csv['view'] == "AP") | (metadata_csv['view'] == "PA"))]
	non_covid = metadata_csv.loc[(metadata_csv['finding'] != "COVID-19") & ((metadata_csv['view'] == "AP") | (metadata_csv['view'] == "PA"))]

	train_dir = "./covid_dataset/train/"
	test_dir = "./covid_dataset/test/"

	covid_sample = covid.sample((covid.shape[0]), random_state=19923, replace=False)
	covid_train = covid_sample.iloc[0:(covid.shape[0] // 2), :]
	covid_test = covid_sample.iloc[(covid.shape[0] // 2):(covid.shape[0]), :]

	non_covid_sample = non_covid.sample((non_covid.shape[0]), random_state=19923, replace=False)
	non_covid_train = non_covid_sample.iloc[0:(non_covid.shape[0] // 2), :]
	non_covid_test = non_covid_sample.iloc[(non_covid.shape[0] // 2):(non_covid.shape[0]), :]

	# loop over the rows of the data frames
	for (i, row) in covid_train.iterrows():
		filename = row["filename"].split(os.path.sep)[-1]
		filePath = os.path.sep.join([imageDir, filename])
		outputPath = os.path.sep.join([train_dir, "COVID"])
		path = Path(outputPath)
		path.mkdir(parents=True, exist_ok=True)
		shutil.copy2(filePath, outputPath)
	for (i, row) in covid_test.iterrows():
		filename = row["filename"].split(os.path.sep)[-1]
		filePath = os.path.sep.join([imageDir, filename])
		outputPath = os.path.sep.join([test_dir, "COVID"])
		path = Path(outputPath)
		path.mkdir(parents=True, exist_ok=True)
		shutil.copy2(filePath, outputPath)
	for (i, row) in non_covid_train.iterrows():
		filename = row["filename"].split(os.path.sep)[-1]
		filePath = os.path.sep.join([imageDir, filename])
		outputPath = os.path.sep.join([train_dir, "NON-COVID"])
		path = Path(outputPath)
		path.mkdir(parents=True, exist_ok=True)
		shutil.copy2(filePath, outputPath)
	for (i, row) in non_covid_test.iterrows():
		filename = row["filename"].split(os.path.sep)[-1]
		filePath = os.path.sep.join([imageDir, filename])
		outputPath = os.path.sep.join([test_dir, "NON-COVID"])
		path = Path(outputPath)
		path.mkdir(parents=True, exist_ok=True)
		shutil.copy2(filePath, outputPath)


sample_train_test_sets()