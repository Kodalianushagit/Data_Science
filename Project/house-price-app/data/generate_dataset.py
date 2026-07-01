import numpy as np
import pandas as pd

np.random.seed(42)
N = 4000

locations = {
    "Whitefield": 5800, "Electronic City": 4200, "Indiranagar": 9500,
    "Koramangala": 10200, "HSR Layout": 8800, "Jayanagar": 8200,
    "Marathahalli": 5600, "Yelahanka": 4800, "Hebbal": 6700,
    "JP Nagar": 7200, "BTM Layout": 6900, "Sarjapur Road": 5400,
}

loc_names = list(locations.keys())
loc_prices = np.array([locations[l] for l in loc_names])
location = np.random.choice(loc_names, size=N, p=loc_prices / loc_prices.sum())

bhk = np.random.choice([1,2,3,4,5], size=N, p=[0.12,0.38,0.33,0.13,0.04])
base_sqft = {1:600, 2:1050, 3:1500, 4:2100, 5:2800}
total_sqft = np.array([base_sqft[b] + np.random.normal(0,150) for b in bhk])
total_sqft = np.clip(total_sqft, 300, 6000).round(0)

bath = np.clip(bhk + np.random.choice([-1,0,0,1], size=N), 1, 6)
balcony = np.clip(np.random.poisson(1.3, size=N), 0, 4)
age_years = np.clip(np.random.exponential(6, size=N), 0, 40).round(1)
furnishing = np.random.choice(["Unfurnished","Semi-Furnished","Furnished"], size=N, p=[0.45,0.35,0.20])
parking = np.random.choice([0,1], size=N, p=[0.3,0.7])

price_per_sqft = np.array([locations[l] for l in location]).astype(float)
furnishing_mult = pd.Series(furnishing).map({"Unfurnished":1.0,"Semi-Furnished":1.08,"Furnished":1.18}).values
age_discount = np.clip(1 - age_years * 0.008, 0.6, 1.0)
parking_bonus = 1 + parking * 0.04
bath_bonus = 1 + (bath - bhk).clip(min=0) * 0.02

price = total_sqft * price_per_sqft * furnishing_mult * age_discount * parking_bonus * bath_bonus
price_lakhs = (price / 100000 * np.random.normal(1, 0.07, size=N)).round(2)
price_lakhs = np.clip(price_lakhs, 8, None)

df = pd.DataFrame({
    "location": location, "total_sqft": total_sqft.astype(int),
    "bhk": bhk, "bath": bath, "balcony": balcony,
    "age_years": age_years, "furnishing": furnishing,
    "parking": parking, "price_lakhs": price_lakhs,
})

df.to_csv("data/housing_data.csv", index=False)
print("Dataset created! Shape:", df.shape)