# Task 3. Multiclass classification and multiple classification/regression

Task explanation taken from [here](https://github.com/rustam-azimov/ml-course/blob/main/tasks/task03_multiclass_multioutput.md).

Data taken from [this project](https://github.com/Mark-Rozenberg/Crime-And-Climate) by [@Mark-Rozenberg](https://github.com/Mark-Rozenberg).

* **Deadline**: 15.12.2022, 23:59 (?)
* **Basic full score**: 5
* **Max points**: 10

## A task

- [x] Explore [sklearn](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-classification) for **Multiclass classification**, **Multilabel classification** and **Multioutput Regression**. The main components of the library are shown in the figure below.
![multi_org_chart](https://scikit-learn.org/stable/_images/multi_org_chart.png)

- [x] Find data that can solve the **Multiclass classification** problem (classification with more than two classes). As a last resort, convert data intended for another task.
- [x] Perform exploratory analysis (**EDA**), use visualization, draw conclusions that may be useful in further solving the problem.
- [x] If necessary, perform useful data transformations (for example, transform categorical features into quantitative ones), remove unnecessary features, create new ones (**Feature Engineering**).
- [x] Using the **OneVsRest**, **OneVsOne** and **OutputCode** strategies, solve the **Multiclass classification** problem for each of the basic classification algorithms passed (**logistic regression, svm, knn, naive bayes, decision tree**). When training, use **hyperparameter fitting**, **cross-validation** and, if necessary, **data scaling**, to achieve the best prediction quality.
- [x] Measure the training time of each model for each strategy.
- [x] To assess the quality of models, use the **AUC-ROC** metric.
- [x] Compare training time and quality of all models and all strategies. To conclude.
- [x] * (**+3 points**) Repeat all points for the **Multilabel classification** task (classification with multiple target features, for example, binary ones). Try **MultiOutputClassifier** and **ClassifierChain** as strategies.
- [x] * (**+2 points**) Repeat all steps for the **Multioutput Regression** task (multioutput regression, real). Model to try at least one: **Ridge**. Try **MultiOutputRegressor** and **RegressorChain** as strategies. Use **R2** as the metric.
