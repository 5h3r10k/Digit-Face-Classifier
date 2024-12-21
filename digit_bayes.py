# Imports


import numpy as np
import os
import constants as c
from NaiveBayes import NaiveBayesClassifier




# Constants


ON = [c.PIXEL,c.EDGE]

TRAINING_IMAGES_PATH = c.CLEAN_DIGIT_TRAINING_IMAGES_DIR
TRAINING_LABELS_PATH = c.CLEAN_DIGIT_TRAINING_LABELS_FILE
TEST_IMAGES_PATH = c.CLEAN_DIGIT_TEST_IMAGES_DIR
TEST_LABELS_PATH = c.CLEAN_DIGIT_TEST_LABELS_FILE

DIGIT_SIZE = c.DIGIT_SIZE
DIGIT_WIDTH = c.DIGIT_WIDTH




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
        simple_grid_on_feature(ascii_image, (2, 2)),
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




# Naive Bayes Classifier

# Load the data
train_data, train_labels = load_dataset(TRAINING_IMAGES_PATH, TRAINING_LABELS_PATH)
test_data, test_labels = load_dataset(TEST_IMAGES_PATH, TEST_LABELS_PATH)

# Print the shapes of the data
print(train_data.shape)
print(train_labels.shape)

for i in range(10):
    print(train_labels[i])
    print(train_data[i])

# Initialize and train the classifier
digit_nb = NaiveBayesClassifier()
digit_nb.fit(train_data, train_labels)
predictions = digit_nb.predict_batch(test_data)

# Calculate the accuracy
accuracy = np.mean(predictions == test_labels)
print(f'Accuracy: {100*accuracy:.2f}%')