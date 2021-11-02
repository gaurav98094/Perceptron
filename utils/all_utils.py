import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use("fivethirtyeight")

import joblib


def prepare_data(df):
  """It is used to seperate the dependent and independent features

  Args:
      df (pd DataFrame): its a pandas dataset

  Returns:
      tuple: it returns tuples of dependent and independent variables
  """

  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y

def save_model(model,filename):
  """It saves the trained model

  Args:
      model (python object): trained model to
      file_name (str): give the name with which u want to save
  """
  model_dir='MODELS'
  os.makedirs(model_dir,exist_ok=True)
  filePath = os.path.join(model_dir,filename)
  joblib.dump(model,filePath)


def save_plot(df, file_name, model):
  """It is used to save the plot

  Args:
      df (pd DataFrame): Datset only 2D
      file_name (str): give the path to save the plot
      model (python object): modelname for which plot is to be created
  """
  def _create_base_plot(df):
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # as a array
    x1 = X[:, 0] 
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    print(xx1)
    print(xx1.ravel())
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()



  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "IMAGES"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)


