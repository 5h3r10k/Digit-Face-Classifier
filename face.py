# Imports


import numpy as np
import os
import constants as c
from NaiveBayes import NaiveBayesClassifier
from Perceptron import PerceptronClassifier
import time
import matplotlib.pyplot as plt




# Constants


ON = [c.PIXEL,c.EDGE]

TRAINING_IMAGES_PATH = c.CLEAN_FACE_TRAINING_IMAGES_DIR
TRAINING_LABELS_PATH = c.CLEAN_FACE_TRAINING_LABELS_FILE
TEST_IMAGES_PATH = c.CLEAN_FACE_TEST_IMAGES_DIR
TEST_LABELS_PATH = c.CLEAN_FACE_TEST_LABELS_FILE

DIGIT_SIZE = c.FACE_SIZE
DIGIT_WIDTH = c.FACE_WIDTH

PLOTS_DIR = c.PLOTS_DIR




# Functions


def load_ascii_image(file_path):
    '''
    Load an ASCII image from a file and convert it to a binary feature vector.
    
    Parameters:
    - file_path (str): Path to the ASCII image file.
    - size (int): Number of rows (height) in the ASCII image.
    - width (int): Number of columns (width) in the ASCII image.
    
    Returns:
    - np.array: A 1D binary feature vector representing the image.
    '''
    with open(file_path, 'r') as file:
        ascii_image = file.readlines()

    return np.concatenate([
        grid_count_on_feature(ascii_image, (7, 6))
    ])

def simple_on_feature(image):
    '''
    Returns a simple matrix where all on pixels are 1 and all off pixels are 0.
    '''
    size = len(image)
    width = len(image[0].rstrip('\n'))
    output = np.zeros(size * width, dtype=int)
    for i in range(size):
        for j in range(width):
            if image[i][j] in ON:
                output[i* width + j] = 1
    return output

def grid_count_on_feature(image, grid_dim):
    '''
    Returns a feature vector that counts the number of on pixels in each grid cell.
    The image is divided into grid cells of size grid_length x grid_width.
    '''
    size = len(image)
    width = len(image[0].rstrip('\n'))
    if size % grid_dim[0] != 0 or width % grid_dim[1] != 0:
        raise ValueError('Grid dimensions do not divide the image dimensions evenly.')
    num_rows = size // grid_dim[0]
    num_cols = width // grid_dim[1]
    num_cells = num_rows * num_cols
    output = np.zeros(num_cells, dtype=int)
    for i in range(num_rows):
        for j in range(num_cols):
            count = 0
            for k in range(grid_dim[0]):
                for l in range(grid_dim[1]):
                    if image[i*grid_dim[0] + k][j*grid_dim[1] + l] in ON:
                        count += 1
            output[i*num_cols + j] = count
    return output

def load_labels(file_path):
    '''
    Load labels from a text file.
    '''
    with open(file_path, 'r') as file:
        return np.array([int(line.strip()) for line in file])

def load_dataset(img_dir, labels_file):
    '''
    Load a dataset from images and labels.
    '''
    images = []
    for file in sorted(os.listdir(img_dir)):
        images.append(load_ascii_image(img_dir + file))
    return np.array(images), load_labels(labels_file)






# Load the data
train_data, train_labels = load_dataset(TRAINING_IMAGES_PATH, TRAINING_LABELS_PATH)
test_data, test_labels = load_dataset(TEST_IMAGES_PATH, TEST_LABELS_PATH)

face_nb = NaiveBayesClassifier()
face_pcp = PerceptronClassifier()

percentages = list(range(10, 101, 10))

accuracies_bayes = []
standard_deviations_bayes = []
times_bayes = []

for i in percentages:
    # get i% of the training data
    train_data_temp = train_data[:(int(i/100*len(train_data)))]
    train_labels_temp = train_labels[:(int(i/100*len(train_labels)))]
    start = time.time()
    face_nb.fit(train_data_temp, train_labels_temp)
    end = time.time()
    accuracy = face_nb.accuracy(test_data, test_labels)
    accuracies_bayes.append(accuracy)
    standard_deviations_bayes.append(face_nb.standard_deviation(test_data, test_labels))
    times_bayes.append(end-start)
    print(f'Training Data Used: {i:.0f}%')
    print(f'Accuracy: {100*accuracy:.2f}%')
    print(f'Standard Deviation: {face_nb.standard_deviation(test_data, test_labels):.2f}')
    print(f'Training Time: {end-start:.9f} seconds')

accuracies_pcp = []
standard_deviations_pcp = []
times_pcp = []

for i in percentages:
    # get i% of the training data
    train_data_temp = train_data[:(int(i/100*len(train_data)))]
    train_labels_temp = train_labels[:(int(i/100*len(train_labels)))]
    start = time.time()
    face_pcp.fit(train_data_temp, train_labels_temp)
    end = time.time()
    accuracy = face_pcp.accuracy(test_data, test_labels)
    accuracies_pcp.append(accuracy)
    standard_deviations_pcp.append(face_pcp.standard_deviation(test_data, test_labels))
    times_pcp.append(end-start)
    print(f'Training Data Used: {i:.0f}%')
    print(f'Accuracy: {100*accuracy:.2f}%')
    print(f'Standard Deviation: {face_pcp.standard_deviation(test_data, test_labels):.2f}')
    print(f'Training Time: {end-start:.9f} seconds')


# graphs

# Bayes

# percentages vs accuracy
plt.plot(percentages, accuracies_bayes)
plt.title('Percentage of Training Data vs Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy')
plt.savefig(f'{PLOTS_DIR}face_bayes_percentages_vs_accuracy.png')

# percentages vs standard deviation
plt.clf()
plt.plot(percentages, standard_deviations_bayes)
plt.title('Percentage of Training Data vs Standard Deviation')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Standard Deviation')
plt.savefig(f'{PLOTS_DIR}face_bayes_percentages_vs_standard_deviation.png')

# percentages vs time
plt.clf()
plt.plot(percentages, times_bayes)
plt.title('Percentage of Training Data vs Training Time')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Training Time (s)')
plt.savefig(f'{PLOTS_DIR}face_bayes_percentages_vs_time.png')


# Perceptron

# percentages vs accuracy
plt.clf()
plt.plot(percentages, accuracies_pcp)
plt.title('Percentage of Training Data vs Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy')
plt.savefig(f'{PLOTS_DIR}face_pcp_percentages_vs_accuracy.png')

# percentages vs standard deviation
plt.clf()
plt.plot(percentages, standard_deviations_pcp)
plt.title('Percentage of Training Data vs Standard Deviation')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Standard Deviation')
plt.savefig(f'{PLOTS_DIR}face_pcp_percentages_vs_standard_deviation.png')

# percentages vs time
plt.clf()
plt.plot(percentages, times_pcp)
plt.title('Percentage of Training Data vs Training Time')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Training Time (s)')
plt.savefig(f'{PLOTS_DIR}face_pcp_percentages_vs_time.png')