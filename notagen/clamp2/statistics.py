gt_feature_folder = ''
output_feature_folder = ''

import os
import json
import random
import re
import numpy as np
from config import *

def load_npy_files(folder_path_list):
    """
    Load all .npy files from a specified folder and return a list of numpy arrays.
    """
    npy_list = []
    for file_path in folder_path_list:
        if file_path.endswith('.npy'):
            # file_path = os.path.join(folder_path, file_name)
            np_array = np.load(file_path)[0]
            npy_list.append(np_array)
    return npy_list

def average_npy(npy_list):
    """
    Compute the average of a list of numpy arrays.
    """
    return np.mean(npy_list, axis=0)

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two numpy arrays.
    """
    dot_product = np.dot(vec1, vec2)
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cosine_sim



def test_generated_results_similarity():

    gt_feature_paths = []
    for gt_feature_file in os.listdir(gt_feature_folder):
        gt_feature_paths.append(os.path.join(gt_feature_folder, gt_feature_file))
    gt_features = load_npy_files(gt_feature_paths)
    gt_avg_feature = average_npy(gt_features)

    clamp2score_list = []
    for output_feature_file in os.listdir(output_feature_folder):
        output_feature_path = os.path.join(output_feature_folder, output_feature_file)
        output_feature = np.load(output_feature_path)[0]
        clamp2score = cosine_similarity(gt_avg_feature, output_feature)
        clamp2score_list.append(clamp2score)
    avg_clampscore = sum(clamp2score_list) / len(clamp2score_list)

    print('average clamp 2 score:', avg_clampscore)
    



if __name__ == '__main__':

    test_generated_results_similarity()

