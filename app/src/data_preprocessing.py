# src/data_preprocessing.py

import numpy as np
import cv2
import pickle

def load_data(file_path='data/cifar10_combined.pkl'):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = np.concatenate(dataset['train_data'], axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(dataset['train_labels'])

    return train_data, train_labels

def compute_histograms(train_data, train_labels):
    histograms = {i: {'r': [], 'g': [], 'b': []} for i in range(10)}
    
    for i in range(len(train_data)):
        label = train_labels[i]
        img = train_data[i]
        for j, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([img], [j], None, [10], [0, 256])
            histograms[label][color].append(hist)

    return histograms

def compute_average_histograms(histograms):
    average_histograms = {i: {'r': None, 'g': None, 'b': None} for i in range(10)}
    for label, color_hists in histograms.items():
        for color in ['r', 'g', 'b']:
            average_histograms[label][color] = np.mean(color_hists[color], axis=0)
    return average_histograms
