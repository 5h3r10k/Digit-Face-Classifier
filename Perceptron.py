import numpy as np

class PerceptronClassifier:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        """
        Initialize the Perceptron classifier.
        
        Parameters:
        - learning_rate (float): The step size for weight updates.
        - n_epochs (int): Number of times to iterate over the training dataset.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def fit(self, training_data, training_labels):
        """
        Train the Perceptron classifier.
        
        Parameters:
        - training_data (np.array): Training data of shape (n_samples, n_features).
        - training_labels (np.array): Training labels of shape (n_samples,).
          Labels should be binary: 0 and 1.
        """
        n_samples, n_features = training_data.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels from {0,1} to {-1,1} for the perceptron algorithm
        y = np.where(training_labels == 0, -1, 1)
        
        for epoch in range(self.n_epochs):
            for idx in range(n_samples):
                linear_output = np.dot(training_data[idx], self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)
                
                # Perceptron update rule
                if y[idx] * y_predicted <= 0:
                    self.weights += self.learning_rate * y[idx] * training_data[idx]
                    self.bias += self.learning_rate * y[idx]
    
    def predict(self, data):
        """
        Predict the class label for a single input sample.
        
        Parameters:
        - data (np.array): Input data sample of shape (n_features,).
        
        Returns:
        - int: Predicted class label (0 or 1).
        """
        linear_output = np.dot(data, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return 1 if y_predicted == 1 else 0
    
    def predict_batch(self, batch_data):
        """
        Predict the class labels for multiple input samples.
        
        Parameters:
        - batch_data (np.array): Input data of shape (n_samples, n_features).
        
        Returns:
        - np.array: Predicted class labels of shape (n_samples,).
        """
        linear_output = np.dot(batch_data, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return np.where(y_predicted == 1, 1, 0)
    
    def accuracy(self, test_data, test_labels):
        """
        Calculate the accuracy of the classifier.
        
        Parameters:
        - test_data (np.array): Test data of shape (n_samples, n_features).
        - test_labels (np.array): True labels of shape (n_samples,).
        
        Returns:
        - float: Accuracy of the classifier.
        """
        predictions = self.predict_batch(test_data)
        return np.mean(predictions == test_labels)
    
    def standard_deviation(self, test_data, test_labels):
        """
        Calculate the standard deviation of the classifier's predictions.
        
        Parameters:
        - test_data (np.array): Test data of shape (n_samples, n_features).
        - test_labels (np.array): True labels of shape (n_samples,).
        
        Returns:
        - float: Standard deviation of the prediction accuracy.
        """
        predictions = self.predict_batch(test_data)
        accuracy_array = predictions == test_labels
        return np.std(accuracy_array)
    
    def _activation_function(self, x):
        """
        Activation function that returns 1 if input >= 0, else -1.
        
        Parameters:
        - x (float): Input value.
        
        Returns:
        - int: 1 or -1.
        """
        return np.where(x >= 0, 1, -1)
