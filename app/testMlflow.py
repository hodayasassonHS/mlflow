
import tkinter as tk
from tkinter import filedialog, Label, Button, Radiobutton, StringVar
from PIL import Image, ImageTk
import io
import os
import shutil
import matplotlib.pyplot as plt
import mlflow

from src.model_definition import Net
from src.data_preprocessing import load_data, compute_histograms, compute_average_histograms
from src.prediction import load_model, get_prediction_with_mlflow
from src.utils import calculate_average_entropy_and_histogram, get_distance



# experiments = mlflow.list_experiments()
# for exp in experiments:
#     print(exp.name)

# runs = mlflow.search_runs(experiment_ids=[263856623132297209])
# print(runs)
# for run in runs:
#     print(run.info.run_id, run.data.metrics)
print(mlflow.__version__)
import mlflow

print(mlflow.__version__)
experiments = mlflow.list_experiments()
print(experiments)

