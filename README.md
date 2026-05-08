03 — Customer Segmentation using K-Means Clustering
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook# ============================================================
# PROJECT 3 — Customer Segmentation using K-Means Clustering
# VTU 2022 Scheme | Data Science Lab | Dept. of ECE
# ============================================================

# STEP 1 — Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# STEP 2 — Create Sample Dataset
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'CustomerID':     range(1, n + 1),
    'Age':            np.random.randint(18, 70, n),
    'Annual_Income':  np.random.randint(15, 120, n),
    'Spending_Score': np.random.randint(1, 100, n),
    'Gender':         np.random.choice(['Male', 'Female'], n)
})

print("Dataset Shape:", data.shape)
print(data.head())

# STEP 3 — EDA
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(data['Annual_Income'], bins=20, color='steelblue')
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 2)
sns.histplot(data['Spending_Score'], bins=20, color='coral')
plt.title('Spending Score Distribution')

plt.subplot(1, 3, 3)
sns.scatterplot(data=data, x='Annual_Income', y='Spending_Score',
                hue='Gender', palette={'Male': 'steelblue', 'Female': 'coral'})
plt.title('Income vs Spending Score')

plt.tight_layout()
plt.show()

# STEP 4 — Feature Selection and Scaling
X = data[['Annual_Income', 'Spending_Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 5 — Elbow Method
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, wcss, marker='o', color='steelblue', linewidth=2)
plt.axvline(x=5, color='red', linestyle='--', label='Optimal K = 5')
plt.title('Elbow Method — Optimal Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.legend()
plt.tight_layout()
plt.show()

# STEP 6 — Apply K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, data['Cluster'])
print(f"Silhouette Score: {sil_score:.4f}")

# STEP 7 — Cluster Visualization
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
labels = [
    'High Income / High Spend',
    'Low Income / Low Spend',
    'Mid Income / Mid Spend',
    'High Income / Low Spend',
    'Low Income / High Spend'
]

plt.figure(figsize=(10, 6))
for i in range(5):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'],
                c=colors[i], label=labels[i], s=60, alpha=0.7)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1],
            c='black', s=250, marker='*', label='Centroids', zorder=5)

plt.title('K-Means Customer Clusters — Income vs Spending Score', fontsize=13)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# STEP 8 — Cluster Summary
print("\n--- Cluster Summary ---")
summary = data.groupby('Cluster')[['Annual_Income', 'Spending_Score', 'Age']].mean().round(1)
print(summary)

actions = {
    0: 'Loyalty programs',
    1: 'Discount offers',
    2: 'Seasonal promotions',
    3: 'Exclusive deals',
    4: 'Targeted bundles'
}

print("\n--- Business Actions per Cluster ---")
for cluster, action in actions.items():
    count = len(data[data['Cluster'] == cluster])
    print(f"Cluster {cluster+1} ({count} customers) → {action}")
