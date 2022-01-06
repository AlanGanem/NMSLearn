# [`nbdev`](https://nbdev.fast.ai/) Data Science template
> This template merges [cookiecutter](https://drivendata.github.io/cookiecutter-data-science/)'s data science project template with [nbdev](https://nbdev.fast.ai/) software development tool. some other features to the folde structures are also included, based on this developer's experience with data science projects.


# README.md Template below:

```python
#import parts of your code in order to show examples
import sys
sys.path.append('..')

from src.ds__load import Class
from src.ds__preprocess import func
```

```python
# You can include code snipets on your index (README.md) simply like thiis:
cls_func = Class(func)
cls_func.apply(1,5)
```




    6



```python
# another example
from functools import reduce
prod_function = lambda x: reduce(lambda x1,x2 : x1*x2, x)
cls_prod = Class(prod_function)
cls_prod.apply([1,2,3,4])
```




    24



# < Analytics Projetct Name >

> The goal of the project is to ...

## Checklist

Mark which tasks have been performed

- [ ] **Summary:** you have included a description, usage, output, accuracy and metadata of your model.
- [ ] **Pre-processing:** you have applied pre-processing to your data and this function is reproducible to new datasets.
- [ ] **Feature selection:** you have performed feature selection while modeling.
- [ ] **Modeling dataset creation:** you have well-defined and reproducible code to generate a modeling dataset that reproduces the behavior of the target dataset.This pipeline is also applicable to generate the deploy dataset.
- [ ] **Model selection:** you have chosen a suitable model according to the project specification.
- [ ] **Model validation:** you have validated your model according to the project specification.
- [ ] **Model optimization:** you have defined functions to optimize hyper-parameters and they are reproducible.
- [ ] **Peer-review:** your code and results have been verified by your colleagues and pre-approved by them.
- [ ] **Acceptance:** this model report has been accepted by the Data Science Manager. State name and date.

---

## Summary

The model is designed to ... (state a simple sentence here to indicate what your model does)

## Usage

Describe step-by-step what should be done to run the algorithm, as in the example below:

1.	Download the database from `<path_to_file>` and place it on an acessible folder in your machine
2.	Clone this repository to your machine
3.	Update the `path` variable in main, to the path chosen on step 1. 
3.	Make sure Python 3.6 is installed on your machine
4.	Install all libraries on `requirements.txt` using the command:

`pip install -r requirements.txt`

5.	Run the command `python src/main.py` 
6.	Check the results on the `output` directory

---

## Output

Clarify what are the files generated as output, with their names, paths and a brief description of the data structure (in case it is a CSV for example)

Example:

The model outputs a list called `<name>.csv` onto `<name>` and contains the following variables:

- var1
- var2
- var3
- var4
- var5
- var6

In the future, the Score variable (from the Credit risk algorithm) shall also be automatically merged onto the output file.

---

## [Metadata](docs/project_metadata.json)

Here you should go the project metadata dictionary (written in JSON), as the file describes

---

## Performance Metrics

The project will be followed by a dashboard, available as a source code on this repo.
The model is considered to be drifting when there are no visible distinction between clusters and/or when the KPI's are below the goal settled for the year.

---

## Pre-processing

Here should be stated in a step-by-step way what was done in the pre-processing stage of the project.

Example:

1.	Excluded readings with `<categories>` from `<rows>`: bad lines.
2.	Limited `<name>` to 4, which represents Bank payroll.
3.	Excludes `<name>` readings with more than 30 days in advance, in case they represent more 1%. 
4.	Stacked all invoices by customer with the mean of each variable (although in some cases other aggregation functions are used instead).

---

## Feature selection

Here should be explained how the features used for the model were selected, if any feature selection methodology was implemented and so on.

Example:

The feature selection methodology used was the Forward Selection. Also, some variables used on the Credit risk model were also considered, up until when the model reached its optimal performance. PCA and RFE did not present better results so far.

**Features used**

Here should be placed the features used to run the model entirely.

Example:

The features used to train both the clustering and classifier can be found under `models/config`, with the static variable `COLS_TO_TRAIN`.

---

## Modeling

Here goes a better explanation on what algorithm was chosen and how does the complete model work.

Example:

To identify the clusters, the K-Means algorithm was chosen. A Random Forest Classifier from scikit-learn was then trained to predict the clusters for new customers and also customers with updated variable values. This pipeline has shown a good performance, running under 5 minutes and with satisfactory results.

---

## Model selection

Here should be stated a clarification on how it was decided to choose the running algorithm, and why not other ones. If any future implementations are envisioned, a brief statement should be made as well.

Example:

Although K-Medians normally performs better with outliers, K-Means still has shown more consistent results and therefore was decided to be used. K-Modes and K-Prototypes were also tested, for including categorical variables, but also had not either reached a good Silhouette Score or divided the data in a meaningful way. Spectral Clustering presented good results for a sample dataset, but has shown poor performance for the complete dataset, and still has to be studied for the next version. DBSCAN was considered, but since it neglects the outliers, it was then decided to be discarded. 

For the classification algorithm, Random Forest outperformed the Decision Tree and the XGBoost classifiers.

---

## Model validation

A step-by-step explanation on what was done to choose the running model. The metrics measured to select them are also desired to be placed in the steps.

Example:

1.	The cluster numbers were chosen based on the Silhouette Score of 0,85.
2.	The dataset was sampled to ensure the clustering consistency.
3.	The clustered data was used as input for the Random Forest Classifier
4.	The data was split into train and test (80/20).
5.	The classifier reached the Accuracy Score of 0,96.

---

## Model optimization

Here should be explained if any kind of optimization of the model was made.

Example:

The hyper-parameters `n_estimators` and `max_depth` were manually optimized for the classifier, although there is still room for improvement, through GridSearch, Genetic Algorithms, etc. 

---

## Drifting and Retraining

Here should be explained what are the necessary steps done in order to retrain the model, whenever some kind of drifting or underperformance is noticed.

Example:

Whenever drifting is noticed, lines 40, 41, 47, 56, 60 and 61 need to be uncommented on `main.py` before running the model. Once it is done, they need to be commented again. In future versions, this process is ought to be automated.

---

## Foreseen improvements

If any future improvements are identified during any of the steps, they should be pointed out in this section.

Example:

- New features to be engineered in the next versions can potentially enhance clustering or recommendations. For example, it is still to be tested whether there is a type of customer that only pays back on a specific time of the month or week, and therefore they won't show in the recommendations list when they don't pay. 

- Treating the (many) outliers with some other measures, could potentially enhance predictions.

- Automatically gather data and run the model is something to be worked on for future versions
