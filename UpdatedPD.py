import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Load your data (replace with your actual file or DataFrame)
df = pd.read_csv("YOUR_FILE.csv")  # or df = your_loaded_dataframe

# 2. Select predictors and response
predictors = [
    'pos_rank', 'stars', 'height_in', 'weight', 'forty', 'vertical',
    'broad_jump_in', 'three_cone', 'shuttle', 'bench', 'Arm_Length_in', 
    'avg_inflated_apy'
]
response = 'contract_apy'

df = df[predictors + [response]].dropna()

# 3. Train-test split
X = df[predictors]
y = df[response]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale features
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 5. Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(train_x_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# 6. Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

# 7. Train the model
history = model.fit(
    train_x_scaled, train_y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 8. Evaluate the model
test_predictions = model.predict(test_x_scaled).flatten()
mae = mean_absolute_error(test_y, test_predictions)
print(f"Test MAE: {mae:.2f}")

# 9. Optionally save predictions
results_df = test_x.copy()
results_df['actual_apy'] = test_y
results_df['predicted_apy'] = test_predictions
results_df.to_csv("predicted_contracts.csv", index=False)
