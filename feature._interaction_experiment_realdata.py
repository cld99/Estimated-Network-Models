from feature_interaction import FeatureInteractionModel
from scipy.io import arff
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# %%
seed = 42
np.random.seed(seed)

dataset = fetch_openml(
    data_id=44140,
    as_frame=True,
    parser="auto"
)
X = dataset.data.astype(np.float32)
y = dataset.target.astype(np.float32)


X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

X_train_std = torch.tensor(X_train_std, dtype=torch.float32)
X_test_std = torch.tensor(X_test_std, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)


# %%
n,p = X_train_std.shape
d = 2
model = FeatureInteractionModel(p=p, d=d)
model.fit(X_train_std, y_train, epochs=3000)
model.eval()


# %%
model.eval()
with torch.no_grad():
    y_pred = model.predict(X_test_std).squeeze()

y_pred = y_pred.detach().cpu().numpy()

if torch.is_tensor(y_test):
    y_test_np = y_test.detach().cpu().numpy()
else:
    y_test_np = y_test

mse = mean_squared_error(y_test_np, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_np, y_pred)


print("Dataset name:", dataset.details["name"])
print("Dataset id:", dataset.details["id"])
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


