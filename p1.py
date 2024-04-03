import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error

# Path
train_path = "dataset/train.csv"
test_path = "dataset/train.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

#print(test_df.head())



# Impute missing values for columns with less than 50% missing
missing_values = train_df.isnull().sum()
missing_percentage = (missing_values / len(train_df)) * 100
num_cols_to_impute = missing_percentage[(missing_percentage > 0) & (missing_percentage < 50)].index.tolist()
num_imputer = SimpleImputer(strategy='median')
train_df[num_cols_to_impute] = num_imputer.fit_transform(train_df[num_cols_to_impute])

# Define a list of selected features based on correlation and data completeness
selected_features = [
    'OutsideHumdity', 'NorthSideHumidity', 'Windspeed', 'Pressure',
    'OutsideTemp', 'LivingRoomHumidity', 'OfficeRoomHumidity',
    'KitchenHumidity', 'BathRoomHumidity', 'BedRoom2Humidity',
    'RandomVariable1', 'RandomVariable2'
]

# Additionally, ensure that 'date' and highly missing columns are not included in the features
features_to_exclude = ['Visibility', 'date'] + num_cols_to_impute
selected_features = [feature for feature in selected_features if feature not in features_to_exclude]

print(f"Selected Features for Modeling: {selected_features}")



# Define features and target variable based on selected features
X = train_df[selected_features]
y = train_df['Visibility']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and validation sets.")


# Initialize the XGBoost regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, random_state=42)

# Train the model on the training set
xgb_model.fit(X_train, y_train)

# Predict visibility on the validation set
y_pred = xgb_model.predict(X_val)

# Evaluate the model using the Mean Squared Log Error (MSLE)
msle = mean_squared_log_error(y_val, y_pred)
score = 100 * max(0, 1 - msle)

print(f'Model Evaluation Score (100*(1-MSLE)): {score}')

