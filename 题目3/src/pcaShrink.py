import cv2
import numpy as np
import math
from segmentation import *
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
img = cv2.imread("imgs/319.png")
