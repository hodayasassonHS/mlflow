# # src/prediction.py

import torch
from PIL import Image
import torchvision.transforms as transforms
import mlflow
import io
import cv2
import numpy as np
from scipy.spatial import distance
from src.model_definition import Net
import torch.nn.functional as F
from src.data_preprocessing import load_data, compute_histograms, compute_average_histograms
from src.utils import calculate_average_entropy_and_histogram, get_distance
import mlflow  # Import the mlflow library for experiment tracking

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']







def get_prediction_with_mlflow(image_data, model, class_names, average_histograms):
    mlflow.set_experiment("model_monitoring_cifar")
    with mlflow.start_run():
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_path = "uploaded_image.jpg"
        pil_image.save(image_path)
        prediction, confidence = predict_image(image_path, model, class_names)
        print(f'prediction: {prediction}, confidence: {confidence}')
        
        if prediction is not None:
            try:
                img = preprocess_image(pil_image)
                entropy, histograms = calculate_average_entropy_and_histogram(img)
                hist_dist = get_distance(prediction, histograms, average_histograms)

                metrics = {
                    "Entropy": entropy,
                    "Histogram_distance": hist_dist,
                    "Confidence": confidence
                }
                mlflow.log_metrics(metrics)
                mlflow.set_tag("class", prediction)
                return prediction
            except Exception as e:
                print(e)
                raise e
        else:
            raise ValueError("Prediction error")



# Load the model function
def load_model(model_path='models/cifar10_model.pth'):
    net = Net()
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()
    return net

# Prediction function
def predict_image(image_path, model, class_names):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
    _, predicted_idx = torch.max(output, 1)
    predicted_class = class_names[predicted_idx.item()]
    confidence = probs[0][predicted_idx.item()].item()
    return predicted_class, confidence

# Additional functions
def preprocess_image(pil_image):
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(pil_image)
    return img_tensor.numpy().transpose(1, 2, 0)

def calculate_average_entropy_and_histogram(img):
    entropy = -np.sum(img * np.log2(img + np.finfo(float).eps))
    histograms = {}
    for j, color in enumerate(['b', 'g', 'r']):
        histograms[color] = cv2.calcHist([img], [j], None, [10], [0, 256])
    return entropy, histograms


