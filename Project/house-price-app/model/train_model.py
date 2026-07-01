import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("data/housing_data.csv")
X = df.drop("price_lakhs", axis=1)
y = df["price_lakhs"]

numeric_features = ["total_sqft", "bhk", "bath", "balcony", "age_years", "parking"]
categorical_features = ["location", "furnishing"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append({"Model": name, "R2": round(r2,4), "MAE": round(mae,2), "RMSE": round(rmse,2)})
    fitted_pipelines[name] = pipe
    print(f"{name} → R2: {round(r2,4)}, MAE: {round(mae,2)}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
best_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_pipelines[best_name]
print(f"\nBest model: {best_name}")

joblib.dump(best_pipeline, "model/house_price_model.pkl")

metadata = {
    "locations": sorted(df["location"].unique().tolist()),
    "furnishing_options": sorted(df["furnishing"].unique().tolist()),
    "best_model_name": best_name,
    "r2_score": float(results_df.iloc[0]["R2"]),
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f)

print("Model saved!")