import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        """
        Initialize the Naive Bayes classifier.
        """
        self.classes = None
        self.priors = {}
        self.likelihoods = {}

    def fit(self, training_data, training_labels):
        """
        Train the Naive Bayes classifier.
        Parameters:
        - training_data (np.array): Training data.
        - training_labels (np.array): Training labels.
        """

        # Get the unique classes (labels) in the training data
        self.classes = np.unique(training_labels)

        # Calculate the prior probability of each class
        for c in self.classes:
            self.priors[c] = np.sum(training_labels == c) / len(training_labels)
        
        # Calculate the likelihood of each feature given the class
        for c in self.classes:
            # Get the training data that belong to class c
            c_data = training_data[training_labels == c]

            # Calculate the likelihood of each feature given the class with Laplace smoothing
            # features are binary
            feature_counts = np.sum(c_data, axis=0)
            self.likelihoods[c] = (feature_counts + 1) / (np.sum(c_data) + 2)

    def predict(self, data):
        """
        Predict the class labels for the input data (single vector)
        Parameters:
        - data (np.array): Input data.
        Returns:
        - np.array: Predicted class labels.
        """
        
        # Initialize posterior probabilities for each class
        posteriors = {}

        for cls in self.classes:
            # start with the prior
            posterior = self.priors[cls]

            # get feature probabilities
            feature_probs = self.likelihoods[cls]

            # multiply by the likelihood of each feature given the class
            for i, feature in enumerate(data):
                if feature == 1:
                    posterior *= feature_probs[i]
                else:
                    posterior *= (1 - feature_probs[i])
            
            posteriors[cls] = posterior

        # return the class with the highest posterior probability
        return max(posteriors, key=posteriors.get)
    
    def predict_batch(self, batch_data):
        """
        Predict the class labels for the input data (batch)
        Parameters:
        - batch_data (np.array): Input data of multiple samples.
        Returns:
        - np.array: Predicted class labels.
        """

        # Predict the class labels for each data sample in the batch
        return [self.predict(data) for data in batch_data]
    
    def accuracy(self, test_data, test_labels):
        """
        Calculate the accuracy of the classifier.
        Parameters:
        - test_data (np.array): Test data.
        - test_labels (np.array): Test labels.
        Returns:
        - float: Accuracy of the classifier.
        """

        # Predict the class labels for the test data
        predictions = self.predict_batch(test_data)

        # Calculate the accuracy
        return np.mean(predictions == test_labels)
    
    def standard_deviation(self, test_data, test_labels):
        """
        Calculate the standard deviation of the classifier.
        Parameters:
        - test_data (np.array): Test data.
        - test_labels (np.array): Test labels.
        Returns:
        - float: Standard deviation of the classifier.
        """

        # Predict the class labels for the test data
        predictions = self.predict_batch(test_data)

        # Calculate the standard deviation
        return np.std(predictions == test_labels)