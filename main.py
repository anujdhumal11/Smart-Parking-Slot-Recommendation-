# 1) Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns
import random

# 2) Synthetic dataset generation
def generate_parking_synthetic(n_zones=10, days=30, capacity=20, seed=42):
    np.random.seed(seed)
    rows = []
    start = pd.to_datetime('2023-01-01')
    for day in range(days):
        for hour in range(24):
            for zone in range(n_zones):
                timestamp = start + pd.Timedelta(days=day, hours=hour)
                day_of_week = timestamp.dayofweek
                # base occupancy pattern: peak midday/afternoon
                base = (np.sin((hour-8)/24*2*np.pi) + 1)/2
                zone_bias = np.random.normal(loc=0, scale=0.1)
                weather = np.random.choice([0,1], p=[0.85,0.15])
                occ = int(np.clip((base + zone_bias + 0.1*weather)*capacity + np.random.normal(0,2), 0, capacity))
                rows.append({
                    'timestamp': timestamp,
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'zone_id': f'zone_{zone}',
                    'weather': weather,
                    'occupancy': occ,
                    'capacity': capacity,
                    'available_slots': capacity - occ
                })
    df = pd.DataFrame(rows)
    df.to_csv('parking_synth.csv', index=False)
    return df

# generate
df = generate_parking_synthetic()
print('Synthetic parking dataset generated:', df.shape)

# 3) Feature engineering for clustering
feat = df.groupby(['zone_id','hour'])['occupancy'].mean().reset_index()
# pivot to zone x hours matrix
pivot = feat.pivot(index='zone_id', columns='hour', values='occupancy').fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(pivot)

# 4) K-Means to cluster zones
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
clusters = pd.DataFrame({'zone_id': pivot.index, 'cluster': kmeans.labels_})
print(clusters)

# 5) Train ANN to predict available_slots
# Prepare features
df_ml = df.copy()
# one-hot zone
zone_ohe = pd.get_dummies(df_ml['zone_id'], prefix='zone')
X_ml = pd.concat([df_ml[['hour','day_of_week','weather','capacity']], zone_ohe], axis=1)
y_ml = df_ml['available_slots']

X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=0)

# simple regression NN
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

# evaluate
preds = model.predict(X_test).flatten()
print('RMSE:', np.sqrt(mean_squared_error(y_test, preds)))

# 6) Recommendation engine (basic): for an incoming request (zone or area), recommend best slot
# Example: incoming driver: hour=15, day_of_week=2, prefers cluster with highest avg availability

def recommend_slot(hour, day_of_week, weather, preferred_zone=None):
    # predict for all zones
    rows = []
    for z in pivot.index:
        row = {'hour': hour, 'day_of_week': day_of_week, 'weather': weather, 'capacity': 20}
        # zone OHE
        for zz in pivot.index:
            row['zone_'+zz] = 1 if zz==z else 0
        rows.append(row)
    Xq = pd.DataFrame(rows)
    preds = model.predict(Xq).flatten()
    best_idx = preds.argmax()
    return pivot.index[best_idx], max(int(round(preds[best_idx])), 0)

print('Recommendation example:', recommend_slot(15,2,0))

# 7) Agent simulation (grid)
# Simple grid world where parking zones are placed. Agent moves to recommended zone using Manhattan distance.

class SimpleAgent:
    def __init__(self, start=(0,0)):
        self.pos = start
    def move_towards(self, target):
        tx, ty = target
        x,y = self.pos
        if x<tx: x+=1
        elif x>tx: x-=1
        elif y<ty: y+=1
        elif y>ty: y-=1
        self.pos = (x,y)

# place zones on grid
zone_positions = {z: (i%5, i//5) for i,z in enumerate(pivot.index)}
agent = SimpleAgent(start=(0,0))
rec_zone,slots = recommend_slot(10,1,0)
print('Recommended zone:', rec_zone, 'slots predicted:', slots, 'position:', zone_positions[rec_zone])
while agent.pos != zone_positions[rec_zone]:
    agent.move_towards(zone_positions[rec_zone])
    print('Agent moved to', agent.pos)

# 8) Visualizations
plt.figure(figsize=(10,4))
plt.title('Zone clusters (by hour occupancy)')
for cluster_id in clusters['cluster'].unique():
    members = clusters[clusters['cluster']==cluster_id]['zone_id']
    for m in members:
        plt.plot(pivot.columns, pivot.loc[m], alpha=0.6)
plt.xlabel('Hour')
plt.ylabel('Average occupancy')
plt.show()
