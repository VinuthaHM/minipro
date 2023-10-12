import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from faker import Faker
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

fake = Faker()

# Define the number of data points
num_entries = 1000

# Generate random data
data = []
for _ in range(num_entries):
    date = fake.date_between(start_date='-1y', end_date='today')
    location = fake.city()
    waste_type = fake.random_element(elements=('Plastic', 'Paper', 'Glass', 'Organic'))

    # Generate correlated variables
    population_density = np.random.uniform(500, 5000)  # Simulate population density
    temperature = np.random.normal(25, 5)  # Simulate temperature

    # Correlated waste quantity based on population density and temperature
    quantity_mean = population_density * 0.01 + temperature * 0.5
    quantity = np.random.normal(quantity_mean, 20)
    quantity = max(0, quantity)  # Ensure quantity is non-negative

    recycling_status = 'Recycled' if np.random.random() < 0.6 else 'Not Recycled'

    data.append([date, location, waste_type, quantity, recycling_status])

# Create a DataFrame
columns = ['Date', 'Location', 'WasteType', 'Quantity', 'RecyclingStatus']
df = pd.DataFrame(data, columns=columns)

# Save the data to a CSV file
df.to_csv('dummy_waste_data.csv', index=False)

#```````````````````````````````````````````````````````````````````````

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

fake = Faker()

# Define the number of data points
num_entries = 1000

# Generate random data
data = []
for _ in range(num_entries):
    date = fake.date_between(start_date='-1y', end_date='today')
    location = fake.city()
    waste_type = fake.random_element(elements=('Plastic', 'Paper', 'Glass', 'Organic'))

    # Generate correlated variables
    population_density = np.random.uniform(100, 5000)  # Simulate population density
    temperature = np.random.normal(25, 10)  # Simulate temperature

    # Adjusted waste quantity based on population density and temperature
    quantity_mean = population_density * 0.005 + temperature * 0.5
    quantity = np.random.normal(quantity_mean, 15)
    quantity = max(0, quantity)  # Ensure quantity is non-negative

    # Introduce a waste type factor
    if waste_type == 'Plastic':
        quantity *= np.random.uniform(1.5, 2.5)
    elif waste_type == 'Paper':
        quantity *= np.random.uniform(0.8, 1.2)

    recycling_status = 'Recycled' if np.random.random() < 0.7 else 'Not Recycled'

    data.append([date, location, waste_type, quantity, recycling_status])

# Create a DataFrame
columns = ['Date', 'Location', 'WasteType', 'Quantity', 'RecyclingStatus']
df = pd.DataFrame(data, columns=columns)

# Save the data to a CSV file
df.to_csv('adjusted_dummy_waste_data.csv', index=False)

#-----------------------------------------------------------
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

fake = Faker()

# Define the number of data points
num_entries = 1000

# Generate random data
data = []
for _ in range(num_entries):
    date = fake.date_between(start_date='-1y', end_date='today')
    location = fake.city()
    waste_type = fake.random_element(elements=('Plastic', 'Paper', 'Glass', 'Organic'))

    # Generate correlated variables with noise
    population_density = np.random.uniform(100, 5000) + np.random.normal(0, 300)
    temperature = np.random.normal(25, 10) + np.random.normal(0, 5)

    # Adjusted waste quantity based on population density and temperature with noise
    quantity_mean = population_density * 0.005 + temperature * 0.5
    quantity = np.random.normal(quantity_mean, 15) + np.random.normal(0, 5)
    quantity = max(0, quantity)  # Ensure quantity is non-negative

    # Introduce noise to waste type scaling factors
    plastic_factor = np.random.uniform(1.5, 2.5) + np.random.normal(0, 0.2)
    paper_factor = np.random.uniform(0.8, 1.2) + np.random.normal(0, 0.1)
    if waste_type == 'Plastic':
        quantity *= plastic_factor
    elif waste_type == 'Paper':
        quantity *= paper_factor

    # Introduce noise to recycling likelihood
    recycling_status = 'Recycled' if np.random.random() < 0.7 + np.random.normal(0, 0.1) else 'Not Recycled'

    data.append([date, location, waste_type, quantity, recycling_status])

# Create a DataFrame
columns = ['Date', 'Location', 'WasteType', 'Quantity', 'RecyclingStatus']
df = pd.DataFrame(data, columns=columns)

# Save the data to a CSV file
df.to_csv('noisy_dummy_waste_data.csv', index=False)
#-------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('noisy_dummy_waste_data.csv')

# Display basic statistics
print(df.describe())

# Histogram of waste quantities
plt.figure(figsize=(10, 6))
plt.hist(df['Quantity'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Waste Quantities')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

# Box plot of waste quantities by waste type
plt.figure(figsize=(10, 6))
df.boxplot(column='Quantity', by='WasteType', grid=False)
plt.title('Waste Quantities by Waste Type')
plt.ylabel('Quantity')
plt.suptitle('')  # Remove default title
plt.show()

# Count of recycling statuses
recycling_counts = df['RecyclingStatus'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(recycling_counts, labels=recycling_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Recycling Status Distribution')
plt.show()
################## data prepreprocessing########################################################################################################3
# Check for missing values in the dataset step1 handling missing values
print(df.isnull().sum())

# No missing values found, so no further action is needed for this step


#encoding categorical variables
# Perform one-hot encoding for 'WasteType' and 'RecyclingStatus'
df = pd.get_dummies(df, columns=['WasteType', 'RecyclingStatus'])
#Train-Validation-Test Split (Revisited)
# Load the dataset
df = pd.read_csv('noisy_dummy_waste_data.csv')

# Define features (X) and target variable (y)
X = df.drop('RecyclingStatus', axis=1)  # Features
y = df['RecyclingStatus']  # Target variable

# Perform a train-validation-test split
# Split the data into 70% train, 15% validation, and 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Optionally, you can save these datasets to separate CSV files
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
#In this code:

    #We load the dataset as before.
    #We define the features (X) and the target variable (y). You might need to adjust this based on your specific modeling task.
    #We use train_test_split twice to split the data into three sets: training, validation, and test. We specified a 70-15-15 split ratio.
    #You can optionally save these datasets to separate CSV files for later use in model development and evaluation.

#Now, you have your synthetic dataset split into training, validation, and test sets, ready for model development and assessment.


### feature selection#############################################################################################################################################3

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# Load the dataset
df = pd.read_csv('noisy_dummy_waste_data.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract relevant features from the 'Date' column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Drop the original 'Date' column
df.drop('Date', axis=1, inplace=True)
# Convert 'RecyclingStatus' into a binary variable so that u will not encounter any value error
############### in the y_train there will be some non numeric value because of which we encounter value error so we converted this into binary values
df['RecyclingStatusBinary'] = (df['RecyclingStatus'] == 'Recycled').astype(int)
df.drop('RecyclingStatus', axis=1, inplace=True)

# Define features (X) and binary target variable (y)
X = df.drop('RecyclingStatusBinary', axis=1)  # Features
y = df['RecyclingStatusBinary']  # Binary target variable



# Perform one-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['Location', 'WasteType'])

# Perform a train-validation-test split if not already done
# Split the data into 70% train, 15% validation, and 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

################################# Perform feature selection using SelectKBest####################################################################################

k = 5  # Number of top features to select
selector = SelectKBest(score_func=chi2, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

# Getting the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Print the indices of the selected features
print("Indices of selected features:", selected_feature_indices)

# Create new DataFrames with only the selected features
X_train_selected_df = pd.DataFrame(X_train_selected, columns=X.columns[selected_feature_indices])
X_val_selected_df = pd.DataFrame(X_val_selected, columns=X.columns[selected_feature_indices])
X_test_selected_df = pd.DataFrame(X_test_selected, columns=X.columns[selected_feature_indices])
from sklearn.linear_model import LinearRegression#importing linear regression
from sklearn.metrics import mean_squared_error#### this is for evaluating the performance of model
# Perform linear regression
model = LinearRegression()
model.fit(X_train_selected_df, y_train)
y_pred = model.predict(X_val_selected_df)

# Threshold predictions for binary outcome
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)
# Convert binary predictions to string labels so that the dataframe contains string values
y_pred_labels = np.where(y_pred_binary == 1, 'Recyclable', 'Not Recyclable')

# Evaluate performance
mse = mean_squared_error(y_val, y_pred)
accuracy = np.sum(y_val == y_pred_binary) / len(y_val)

# Print predictions
predictions_df = pd.DataFrame({'Actual': y_val, 'Predicted Probability': y_pred})
print(predictions_df.head(10))

# Print performance metrics
print("Mean Squared Error:", mse)
print("Accuracy:", accuracy)
import matplotlib.pyplot as plt  ### this is for comparing the actual values and predicted values so that we can have better idea through visualization
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("actual vs predicted")
pred_y_df=pd.DataFrame({'actual value':y_test,'predicted value':y_pred,'difference':y_test-y_pred})
plt.show()
# Convert binary predictions to string label
from sklearn.metrics import mean_squared_error,accuracy_score
# Predict on the test set
y_test_pred = model.predict(X_test_selected_df)

# Threshold predictions for binary outcome on the test set
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# Convert binary predictions to string labels on the test set
y_test_pred_labels = np.where(y_test_pred_binary == 1, 'Recyclable', 'Not Recyclable')

# Evaluate performance on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred_binary)

# Print predictions on the test set
predictions_test_df = pd.DataFrame({'Actual': y_test, 'Predicted Probability': y_test_pred, 'Predicted Label': y_test_pred_labels})
print(predictions_test_df.head(10))

# Print performance metrics on the test set
print("\nPerformance on Test Set:")
print("Mean Squared Error:", mse_test)
print("Accuracy:", accuracy_test)
print(predictions_test_df)
