# Deep human orientation estimation

Human orientation is a crucial information for a mobile robot that has to follow a person moving freely. This paper introduces a deep neural network architecture that uses 2D point clouds from a LiDAR mounted at knee height on a mobile robot platform named SUMMIT-XL to predict human orientation. The proposed architecture integrates Principal Component Analysis (PCA) to reduce input data dimensionality and enhance neural network generalization. It combines a Convolutional Neural Network (CNN) for feature extraction and a Recurrent Neural Network (RNN) with a time windower to capture implicit gait dynamics within a time-series point cloud. Evaluation of this architecture was conducted on a dataset collected over six hours from five volunteers, comprising point cloud data and ground truth of human orientation during complex walking patterns. The general model, trained on a combination of multi-person data, achieved a Mean Absolute Error (MAE) of less than 13 degrees, while the customized model, trained on individual data, achieved an MAE of less than 7 degrees and showed no delay.

# Elements
Datasets after processing by PCA and the code example, including reading the data, building general and customized models, training, and results. Please get in touch with us by email if you need the original point cloud data set. gao@isir.upmc.fr

# Dependencies
TensorFlow Version: 2.13.1, 
tensorflow_addons, 
numpy, 
plotly
