'''
Data importer for the project
- Will read and split the composite ascii files into separate ascii images
- Will copy the label files
'''

# Imports

import os
import shutil
import constants as c




# Constants

DIGIT_SIZE = c.DIGIT_SIZE
FACE_SIZE = c.FACE_SIZE




# Src data directories

DATA_DIR = c.DATA_DIR

SRC_DIGIT_TRAINING_IMAGES_FILE = c.SRC_DIGIT_TRAINING_IMAGES_FILE
SRC_DIGIT_TEST_IMAGES_FILE = c.SRC_DIGIT_TEST_IMAGES_FILE
SRC_DIGIT_VALIDATION_IMAGES_FILE = c.SRC_DIGIT_VALIDATION_IMAGES_FILE
SRC_DIGIT_TRAINING_LABELS_FILE = c.SRC_DIGIT_TRAINING_LABELS_FILE
SRC_DIGIT_TEST_LABELS_FILE = c.SRC_DIGIT_TEST_LABELS_FILE
SRC_DIGIT_VALIDATION_LABELS_FILE = c.SRC_DIGIT_VALIDATION_LABELS_FILE

SRC_FACE_TRAINING_IMAGES_FILE = c.SRC_FACE_TRAINING_IMAGES_FILE
SRC_FACE_TEST_IMAGES_FILE = c.SRC_FACE_TEST_IMAGES_FILE
SRC_FACE_VALIDATION_IMAGES_FILE = c.SRC_FACE_VALIDATION_IMAGES_FILE
SRC_FACE_TRAINING_LABELS_FILE = c.SRC_FACE_TRAINING_LABELS_FILE
SRC_FACE_TEST_LABELS_FILE = c.SRC_FACE_TEST_LABELS_FILE
SRC_FACE_VALIDATION_LABELS_FILE = c.SRC_FACE_VALIDATION_LABELS_FILE

# Cleaned data directories

CLEAN_DATA_DIR = c.CLEAN_DATA_DIR

CLEAN_DIGIT_TRAINING_IMAGES_DIR = c.CLEAN_DIGIT_TRAINING_IMAGES_DIR
CLEAN_DIGIT_TEST_IMAGES_DIR = c.CLEAN_DIGIT_TEST_IMAGES_DIR
CLEAN_DIGIT_VALIDATION_IMAGES_DIR = c.CLEAN_DIGIT_VALIDATION_IMAGES_DIR
CLEAN_DIGIT_TRAINING_LABELS_FILE = c.CLEAN_DIGIT_TRAINING_LABELS_FILE
CLEAN_DIGIT_TEST_LABELS_FILE = c.CLEAN_DIGIT_TEST_LABELS_FILE
CLEAN_DIGIT_VALIDATION_LABELS_FILE = c.CLEAN_DIGIT_VALIDATION_LABELS_FILE

CLEAN_FACE_TRAINING_IMAGES_DIR = c.CLEAN_FACE_TRAINING_IMAGES_DIR
CLEAN_FACE_TEST_IMAGES_DIR = c.CLEAN_FACE_TEST_IMAGES_DIR
CLEAN_FACE_VALIDATION_IMAGES_DIR = c.CLEAN_FACE_VALIDATION_IMAGES_DIR
CLEAN_FACE_TRAINING_LABELS_FILE = c.CLEAN_FACE_TRAINING_LABELS_FILE
CLEAN_FACE_TEST_LABELS_FILE = c.CLEAN_FACE_TEST_LABELS_FILE
CLEAN_FACE_VALIDATION_LABELS_FILE = c.CLEAN_FACE_VALIDATION_LABELS_FILE




# Data import functions

def generateDirectories():
    '''
    Generates the clean data directories. If they already exist, they will be removed and recreated.
    '''
    # remove existing clean data directories
    if os.path.exists(CLEAN_DATA_DIR):
        shutil.rmtree(CLEAN_DATA_DIR)
    print('Removed existing clean data directories')
    
    # generate new clean data directories
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

    os.makedirs(CLEAN_DIGIT_TRAINING_IMAGES_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIGIT_TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIGIT_VALIDATION_IMAGES_DIR, exist_ok=True)
    
    os.makedirs(CLEAN_FACE_TRAINING_IMAGES_DIR, exist_ok=True)
    os.makedirs(CLEAN_FACE_TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(CLEAN_FACE_VALIDATION_IMAGES_DIR, exist_ok=True)

    print('Generated new clean data directories')

def importDataLabels(src_file, clean_file):
    '''
    Imports the data labels
    '''
    shutil.copyfile(src_file, clean_file)
    print('Imported data labels from ' + src_file + ' to ' + clean_file)

def importDataImages(src_file, clean_dir, size):
    '''
    Imports the ASCII data images from a composite ASCII file to separate ASCII files
    Each image has 'size' lines
    The images are saved in the clean directory
    '''
    line_count = 0
    image_count = 0
    with open(src_file, 'r') as src:
        for line in src:
            if line_count % size == 0:
                image_count += 1
                image_file = clean_dir + str(image_count) + '.txt'
                with open(image_file, 'w') as image:
                    image.write(line)
            else:
                with open(image_file, 'a') as image:
                    image.write(line)
            line_count += 1
    print(f'Imported {image_count} images from {src_file} to {clean_dir}')

def importDigitData():
    '''
    Imports the digit data
    '''
    importDataLabels(SRC_DIGIT_TRAINING_LABELS_FILE, CLEAN_DIGIT_TRAINING_LABELS_FILE)
    importDataLabels(SRC_DIGIT_TEST_LABELS_FILE, CLEAN_DIGIT_TEST_LABELS_FILE)
    importDataLabels(SRC_DIGIT_VALIDATION_LABELS_FILE, CLEAN_DIGIT_VALIDATION_LABELS_FILE)
    importDataImages(SRC_DIGIT_TRAINING_IMAGES_FILE, CLEAN_DIGIT_TRAINING_IMAGES_DIR, DIGIT_SIZE)
    importDataImages(SRC_DIGIT_TEST_IMAGES_FILE, CLEAN_DIGIT_TEST_IMAGES_DIR, DIGIT_SIZE)
    importDataImages(SRC_DIGIT_VALIDATION_IMAGES_FILE, CLEAN_DIGIT_VALIDATION_IMAGES_DIR, DIGIT_SIZE)

def importFaceData():
    '''
    Imports the face data
    '''
    importDataLabels(SRC_FACE_TRAINING_LABELS_FILE, CLEAN_FACE_TRAINING_LABELS_FILE)
    importDataLabels(SRC_FACE_TEST_LABELS_FILE, CLEAN_FACE_TEST_LABELS_FILE)
    importDataLabels(SRC_FACE_VALIDATION_LABELS_FILE, CLEAN_FACE_VALIDATION_LABELS_FILE)
    importDataImages(SRC_FACE_TRAINING_IMAGES_FILE, CLEAN_FACE_TRAINING_IMAGES_DIR, FACE_SIZE)
    importDataImages(SRC_FACE_TEST_IMAGES_FILE, CLEAN_FACE_TEST_IMAGES_DIR, FACE_SIZE)
    importDataImages(SRC_FACE_VALIDATION_IMAGES_FILE, CLEAN_FACE_VALIDATION_IMAGES_DIR, FACE_SIZE)




# Run the data import

def runImport():
    '''
    Imports all the data
    '''
    generateDirectories()
    importDigitData()
    importFaceData()