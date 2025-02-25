# Databricks notebook source
# MAGIC %md
# MAGIC # Treinamento de modelos de aprendizado de máquina em dados tabulares: um exemplo de ponta a ponta (não utiliza Unity Catalog)
# MAGIC
# MAGIC Este tutorial abrange os seguintes passos:
# MAGIC - Visualize os dados utilizando Seaborn e matplotlib
# MAGIC - Execute uma varredura de hiperparâmetros em paralelo para treinar múltiplos modelos
# MAGIC - Explore os resultados da varredura de hiperparâmetros
# MAGIC - Aplique o modelo registrado a outro conjunto de dados utilizando uma UDF do Spark
# MAGIC
# MAGIC Neste exemplo, você constrói um modelo para prever a qualidade do vinho português "Vinho Verde" com base nas propriedades físico-químicas do vinho.
# MAGIC
# MAGIC O exemplo utiliza um conjunto de dados do UCI Machine Learning Repository, apresentado em [*Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009].
# MAGIC
# MAGIC ## Requisitos
# MAGIC Este notebook requer um cluster executando Databricks Runtime 15.4 LTS ML ou superior.
# MAGIC
# MAGIC Se o seu workspace estiver habilitado para o Unity Catalog, não utilize este notebook. Uma versão para workspaces habilitados para o Unity Catalog está disponível: ([AWS](https://docs.databricks.com/mlflow/end-to-end-example.html) | [Azure](https://docs.microsoft.com/azure/databricks/mlflow/end-to-end-example) | [GCP](https://docs.gcp.databricks.com/mlflow/end-to-end-example.html)).

# COMMAND ----------

# MAGIC %md ## Read the data
# MAGIC Read the white wine quality and red wine quality CSV datasets and merge them into a single DataFrame.

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ## Visualize data
# MAGIC
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md Plot a histogram of the dependent variable, quality.

# COMMAND ----------

import seaborn as sns
sns.displot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md Looks like quality scores are normally distributed between 3 and 9. 
# MAGIC
# MAGIC Define a wine as high quality if it has quality >= 7.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md Box plots are useful for identifying correlations between features and a binary label. Create box plots for each feature to compare high-quality and low-quality wines. Significant differences in the box plots indicate good predictors of quality.

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md In the above box plots, a few variables stand out as good univariate predictors of quality. 
# MAGIC
# MAGIC - In the alcohol box plot, the median alcohol content of high quality wines is greater than even the 75th quantile of low quality wines. High alcohol content is correlated with quality.
# MAGIC - In the density box plot, low quality wines have a greater density than high quality wines. Density is inversely correlated with quality.

# COMMAND ----------

# MAGIC %md ## Preprocess data
# MAGIC Before training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the dataset to train a baseline model
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = data.drop(["quality"], axis=1)
y = data.quality

# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

# MAGIC %md ## Train a baseline model
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC
# MAGIC Build a simple classifier using scikit-learn and use MLflow to keep track of the model's accuracy, and to save the model for later use.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflow contains utilities to create a conda environment used to serve models.
  # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

# MAGIC %md Review the learned feature importances output by the model. As illustrated by the previous boxplots, alcohol and density are important in predicting quality.

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md You logged the Area Under the ROC Curve (AUC) to MLflow. Click the Experiment icon <img src="https://docs.databricks.com/_static/images/icons/experiment.png"/> in the right sidebar to display the Experiment Runs sidebar. 
# MAGIC
# MAGIC The model achieved an AUC of 0.854.
# MAGIC
# MAGIC A random classifier would have an AUC of 0.5, and higher AUC values are better. For more information, see [Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

# COMMAND ----------

# MAGIC %md #### Register the model in MLflow Model Registry
# MAGIC
# MAGIC By registering this model in Model Registry, you can easily reference the model from anywhere within Databricks.
# MAGIC
# MAGIC The following section shows how to do this programmatically.

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# COMMAND ----------

# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists with the name "wine_quality". Try using a different name.
model_name = "wine_quality"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# MAGIC %md You should now see the model in the Models page. To display the Models page, click **Models** in the left sidebar. 
# MAGIC
# MAGIC Next, transition this model to production and load it into this notebook from Model Registry.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md The Models page now shows the model version in stage "Production".
# MAGIC
# MAGIC You can now refer to the model using the path "models:/wine_quality/production".

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ##Experiment with a new model
# MAGIC
# MAGIC The random forest model performed well even without hyperparameter tuning.
# MAGIC
# MAGIC Use the xgboost library to train a more accurate model. Run a hyperparameter sweep to train multiple models in parallel, using Hyperopt and SparkTrials. As before, MLflow tracks the performance of each parameter configuration.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}

def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    validation = xgb.DMatrix(data=X_val, label=y_val)
    # Pass in the validation set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
    # is no longer improving.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(validation, "validation")], early_stopping_rounds=50)
    validation_predictions = booster.predict(validation)
    auc_score = roc_auc_score(y_val, validation_predictions)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=96,
    trials=spark_trials,
  )

# COMMAND ----------

# MAGIC %md  
# MAGIC #### Use MLflow to view the results
# MAGIC Open up the Experiment Runs sidebar to see the MLflow runs. Click on Date next to the down arrow to display a menu, and select 'auc' to display the runs sorted by the auc metric. The highest auc value is 0.90.
# MAGIC
# MAGIC MLflow tracks the parameters and performance metrics of each run. Click the External Link icon <img src="https://docs.databricks.com/_static/images/icons/external-link.png"/> at the top of the Experiment Runs sidebar to navigate to the MLflow Runs Table. 
# MAGIC
# MAGIC For details about how to use the MLflow runs table to understand how the effect of individual hyperparameters on run metrics, see the  documentation ([AWS](https://docs.databricks.com/mlflow/runs.html#compare-runs) | [Azure](https://docs.microsoft.com/azure/databricks//mlflow/runs#--compare-runs) | [GCP](https://docs.gcp.databricks.com/mlflow/runs.html#compare-runs)). 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md #### Update the production `wine_quality` model in MLflow Model Registry
# MAGIC Earlier, you saved the baseline model to Model Registry with the name `wine_quality`. Now you can update `wine_quality` to a more accurate model from the hyperparameter sweep.
# MAGIC
# MAGIC Because you used MLflow to log the model produced by each hyperparameter configuration, you can use MLflow to identify the best performing run and save the model from that run to the Model Registry.

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# MAGIC %md Click **Models** in the left sidebar to see that the `wine_quality` model now has two versions. 
# MAGIC
# MAGIC Promote the new version to production.

# COMMAND ----------

# Archive the old model version
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)

# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

# MAGIC %md Clients that call load_model now receive the new model.

# COMMAND ----------

# This code is the same as the last block of "Building a Baseline Model". No change is required for clients to get the new model!
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md The new version achieved a better score (AUC = 0.90) on the test set.

# COMMAND ----------

# MAGIC %md ##Batch inference
# MAGIC
# MAGIC There are many scenarios where you might want to evaluate a model on a corpus of new data. For example, you may have a fresh batch of data, or may need to compare the performance of two models on the same corpus of data.
# MAGIC
# MAGIC Evaluate the model on data stored in a Delta table, using Spark to run the computation in parallel.

# COMMAND ----------

# To simulate a new corpus of data, save the existing X_train data to a Delta table. 
# In the real world, this would be a new batch of data.
spark_df = spark.createDataFrame(X_train)
# Replace <username> with your username before running this cell.
table_path = "dbfs:/<username>/delta/wine_data"
# Delete the contents of this path in case this cell has already been run
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

# MAGIC %md Load the model into a Spark UDF, so it can be applied to the Delta table.

# COMMAND ----------

import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# Read the "new data" from Delta
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct

# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# Each row now has an associated prediction. Note that the xgboost function does not output probabilities by default, so the predictions are not limited to the range [0, 1].
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve the model
# MAGIC To productionize the model for low latency predictions, use Databricks Model Serving to deploy the model to an endpoint. Databricks Model Serving is available in AWS and Azure workspaces. See the documentation for Databricks Model Serving: [AWS](https://docs.databricks.com/machine-learning/model-serving/index.html) | [Azure](https://docs.microsoft.com/azure/databricks/machine-learning/model-serving/index). 
# MAGIC
# MAGIC For information about model serving on Google Cloud Platform, see [Legacy MLflow Model Serving](https://docs.gcp.databricks.com/en/archive/legacy-model-serving/model-serving.html).
