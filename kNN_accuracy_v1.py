import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, f1_score

# Read the data from the CSV file
df = pd.read_csv('C:/VU - Machine Learning/MLProject66/food_cleaned_v1.csv', delimiter=';')

# Extract the features and labels from the DataFrame
X = df.iloc[:, 1:].values

# Create a label encoder and fit/transform the 'y' array
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df.iloc[:, 0].values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the range of k values to consider
k_values = range(1, 51)

# Initialize an array to store the accuracy, precision, and F1-score scores for each k value
accuracy_scores = np.zeros(len(k_values))
precision_scores = np.zeros(len(k_values))
f1_scores = np.zeros(len(k_values))

# Loop over each k value and compute its accuracy score
for i, k in enumerate(k_values):
    # Initialize an array to store the predicted labels for the test data
    y_pred = np.zeros(y_test.shape[0])

    # Loop through each test sample and predict its label
    for j in range(X_test.shape[0]):
        # Compute the Euclidean distance between the test sample and each training sample
        #dists = np.sqrt(np.sum((X_train - X_test[j])**2, axis=1))
        
        # Compute the Manhattan distance between the test sample and each training sample
        dists = np.sum(np.abs(X_train - X_test[j]), axis=1)
        
        # Comput the Chebyshev distance between the test sample and each training sample
        #dists = np.max(np.abs(X_train - X_test[j]), axis=1)
        
        # Compute the Jaccard distance between the test sample and each training sample
        #intersection = np.logical_and(X_train, np.tile(X_test[j], (X_train.shape[0], 1)))
        #union = np.logical_or(X_train, np.tile(X_test[j], (X_train.shape[0], 1)))
        #dists = 1 - np.sum(intersection, axis=1) / np.sum(union, axis=1)

        # Find the indices of the k closest training samples
        idx = np.argsort(dists)[:k]

        # Get the corresponding labels for the k closest training samples
        closest_y = y_train[idx]

        # Assign the predicted label for the test sample as the mode (most common value) of the corresponding labels
        y_pred[j] = np.argmax(np.bincount(closest_y))

    accuracy_scores[i] = np.mean(y_pred == y_test)
    f1_scores[i] = f1_score(y_test, y_pred, average='macro')
    print(f"k={k}: accuracy={accuracy_scores[i]:.4f}, F1-score={f1_scores[i]:.4f}")

# Find the k value with the highest accuracy score
best_k = k_values[np.argmax(accuracy_scores)]
print(f"\nBest k value: {best_k}, accuracy={accuracy_scores.max():.4f}, F1-score={f1_scores[np.argmax(accuracy_scores)]:.4f}")
