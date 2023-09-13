
# ===================================
# IMPORTS
# ===================================
import tkinter as tk
from tkinter import filedialog, Label, Button, Radiobutton, StringVar
from PIL import Image, ImageTk
import io
import os
import shutil
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.model_definition import Net
from src.data_preprocessing import load_data, compute_histograms, compute_average_histograms
from src.prediction import load_model, get_prediction_with_mlflow
from src.utils import calculate_average_entropy_and_histogram, get_distance

# ===================================
# GLOBAL VARIABLES
# ===================================
global misclassified_run_id
misclassified_run_id = None
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ===================================
# TKINTER SETUP
# ===================================
root = tk.Tk()
root.title("Image Classifier UI")
root.geometry('600x400')

bg_image = Image.open("app/image.png")
bg_image = bg_image.resize((600, 400), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

train_data, train_labels = load_data()
histograms = compute_histograms(train_data, train_labels)
average_histograms = compute_average_histograms(histograms)
model = load_model()

feedback_var = tk.StringVar(value="good")

# ===================================
# FUNCTIONS
# ===================================
def upload_image():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()

        prediction = get_prediction_with_mlflow(image_data, model, class_names, average_histograms)

        img = Image.open(io.BytesIO(image_data))
        img.thumbnail((150, 150))
        img = ImageTk.PhotoImage(img)

        label_image.config(image=img)
        label_image.image = img

        result_text = f"Prediction: {prediction}"
        label_result.config(text=result_text)

def log_misclassified_image(file_path):
    # Ensure the run is started
    with mlflow.start_run() as run:
        # Directory for misclassified images
        artifact_dir = "misclassified_images"

        # Check if directory exists, if not, create it
        if not os.path.exists(artifact_dir):
            os.makedirs(artifact_dir)

        # Copy the misclassified image to this directory
        import shutil
        shutil.copy(file_path, artifact_dir)
        
        # Log the image in that directory as an artifact in MLflow
        mlflow.log_artifact(os.path.join(artifact_dir, os.path.basename(file_path)))

def plot_metric(metric_values, title):
    plt.figure(figsize=(10,6))
    plt.plot(metric_values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Run Index')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.show()

def fetch_metric_from_all_runs(experiment_name, metric_name):
    # Get the experiment by its name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # Get all runs for the experiment
    runs_info = mlflow.list_run_infos(experiment.experiment_id)
    metric_values = []

    # Loop over each run and fetch the metric value
    for run_info in runs_info:
        data = mlflow.get_run(run_info.run_id).data
        if metric_name in data.metrics:
            metric_values.append(data.metrics[metric_name])
    
    return metric_values



def process_feedback():
    if feedback_var.get() == "bad":
        log_misclassified_image(file_path)




def fetch_metric_from_all_runs(experiment_name, metric_name):
    # Get the experiment by its name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # Get all runs for the experiment using search_runs
    runs = mlflow.search_runs([experiment.experiment_id], filter_string='')
    metric_values = []

    # Construct the metric column name based on your structure
    metric_column_name = f"metrics.{metric_name}"
    
    # Loop over each run and fetch the metric value if it exists
    for _, run in runs.iterrows():
        # Check if the metric value exists and isn't NaN
        if metric_column_name in run and not pd.isna(run[metric_column_name]):
            metric_values.append(run[metric_column_name])

    return metric_values

def initialize_default_image():
    icon_image = Image.open("app/image_icon.png")
    icon_image = icon_image.resize((150, 150), Image.LANCZOS)
    icon_photo = ImageTk.PhotoImage(icon_image)

    label_image.config(image=icon_photo)
    label_image.image = icon_photo


    


# ===================================
# WIDGET CREATION AND LAYOUT
# ===================================
# # Replace "YourExperimentName" with the name of your experiment
experiment_name = "model_monitoring_cifar"

histogram_distances = fetch_metric_from_all_runs(experiment_name, "Histogram_distance")
plot_metric(histogram_distances, "Histogram Distance across Runs")


entropies = fetch_metric_from_all_runs(experiment_name, "Entropy")
plot_metric(entropies, "Entropy across Runs")

confidences = fetch_metric_from_all_runs(experiment_name, "Confidence")
plot_metric(confidences, "Confidence across Runs")



bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

btn_upload = Button(root, text="Upload Image", command=upload_image, bg="#2e7d32", fg="#ffffff", font=("Arial", 12))
btn_upload.pack(pady=20)

label_image = Label(root, bg="white")
label_image.pack(pady=20)
initialize_default_image()  # Initialize default icon image

label_result = Label(root, text="", font=('Arial', 10), fg="#ffffff", bg="#4a4a4a")
label_result.pack(pady=20)

radio_good = Radiobutton(root, text="Good", variable=feedback_var, value="good", bg="#4a4a4a", fg="#ffffff")
radio_bad = Radiobutton(root, text="Not Good", variable=feedback_var, value="bad", bg="#4a4a4a", fg="#ffffff")
radio_good.pack(pady=1)
radio_bad.pack(pady=1)

btn_submit_feedback = Button(root, text="Submit Feedback", command=process_feedback, bg="#2e7d32", fg="#ffffff", font=("Arial", 12))
btn_submit_feedback.pack(pady=20)

# ===================================
# MAINLOOP
# ===================================
root.mainloop()
