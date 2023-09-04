import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV file
data = pd.read_csv('Cleaned(csv).csv', delimiter=';')

# Split data into features and labels
X = data.iloc[:, 2:].values
y = data.iloc[:, 0].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create kNN classifier with k=?
knn = KNeighborsClassifier(n_neighbors=50)

# Train the classifier on the training set
knn.fit(X_train, y_train)

# Test the classifier on the testing set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
