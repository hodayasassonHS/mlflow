
# src/utils.py

import cv2
import numpy as np
from scipy.spatial import distance


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def calculate_average_entropy_and_histogram(img):
    entropy = -np.sum(img * np.log2(img + np.finfo(float).eps))
    histograms = {}
    for j, color in enumerate(['b', 'g', 'r']):
        histograms[color] = cv2.calcHist([img], [j], None, [10], [0, 256])
    return entropy, histograms

def get_distance(prediction, histograms, average_histograms):
    predicted_class_idx = class_names.index(prediction)
    avg_histogram = average_histograms[predicted_class_idx]
    dist = 0
    for color in ['r', 'g', 'b']:
        dist += distance.euclidean(histograms[color].flatten(), avg_histogram[color].flatten())
    return dist
