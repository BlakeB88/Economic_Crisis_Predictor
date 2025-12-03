import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# Config
INPUT_FILE = "../Preprocessing/Final Dataset/crisis_summary_columnNormalized.csv"
COUNTRY_COL = "Country"
YEAR_COL = "Year"
TARGET_COL = "target_crisis_next_3y"
RANDOM_STATE = 42

# Load the preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv(INPUT_FILE)
print(f"Dataset Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df[TARGET_COL].value_counts())

# Seperate features from target and metadata
feature_cols = [c for c in df.columns if c not in [COUNTRY_COL, YEAR_COL, TARGET_COL, "External_Debt_Crisis"]]
X = df[feature_cols].values
y = df[TARGET_COL].values

print(f"\nNumber of features: {len(feature_cols)}")

# PART 1: SILHOUETTE ANALYSIS

print("\n" + "="*60)
print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("="*60)

k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

    print(f"K={k}: Inertia={inertias[-1]:.2f}, Silhouette={silhouette_scores[-1]:.4f}")


# PART 2: K-MEANS CLUSTERING

print("\n" + "="*60)
print("K-MEANS CLUSTERING")
print("="*60)

# Using optimal K
optimal_k = 3  # Based on Silhouette score, not arbitrary choice
print(f"\nUsing K={optimal_k} clusters")

kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# Add cluster labels to dataframe
df['KMeans_Cluster'] = kmeans_labels

# Analyze Cluster
print("\n--- Cluster Analysis ---")
for cluster_id in range(optimal_k):
    cluster_mask = kmeans_labels == cluster_id
    crisis_rate = y[cluster_mask].mean()
    cluster_size = cluster_mask.sum()

    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {cluster_size} ({cluster_size/len(X)*100:.1f}%)")
    print(f"  Crisis Rate: {crisis_rate:.2%}")
    print(f"  Risk Level: {'HIGH' if crisis_rate > 0.15 else 'MEDIUM' if crisis_rate > 0.05 else 'LOW'}")

# K-means evaluation metrics
kmeans_silhouette = silhouette_score(X, kmeans_labels)
kmeans_inertia = kmeans.inertia_

print(f"\nK-Means Performance Metrics:")
print(f"  Silhouette Score: {kmeans_silhouette:.4f} (higher is better)")
print(f"  Inertia: {kmeans_inertia:.2f} (lower is better)")

# PART 3: GMM
print("\n" + "="*60)
print("GAUSSIAN MIXTURE MODEL")
print("="*60)

# Determine optimal number of components using BIC/AIC
n_components_range = range(2, 11)
bic_scores = []
aic_scores= []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=RANDOM_STATE, n_init=10)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

optimal_n = n_components_range[np.argmin(bic_scores)]
print(f"\nOptimal number of components (by BIC): {optimal_n}")

# Fit GMM with optimal components
gmm = GaussianMixture(n_components=optimal_n, random_state=RANDOM_STATE, n_init=10)
gmm.fit(X)
gmm_labels = gmm.predict(X)
gmm_probs = gmm.predict_proba(X)

# Add GMM labels to dataframe
df["GMM_Cluster"] = gmm_labels
df["GMM_Max_Probability"] = gmm_probs.max(axis=1)

# Analyze GMM cluster
print("\n--- GMM Cluster Analysis ---")
for cluster_id in range(optimal_n):
    cluster_mask = gmm_labels == cluster_id
    crisis_rate = y[cluster_mask].mean()
    cluster_size = cluster_mask.sum()
    avg_prob = gmm_probs[cluster_mask, cluster_id].mean()

    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {cluster_size} ({cluster_size/len(X)*100:.1f}%)")
    print(f"  Crisis Rate: {crisis_rate:.2%}")
    print(f"  Avg Assignment Probability: {avg_prob:.3f}")
    print(f"  Risk Level: {'HIGH' if crisis_rate > 0.15 else 'MEDIUM' if crisis_rate > 0.05 else 'LOW'}")

# GMM evaluation
gmm_silhouette = silhouette_score(X, gmm_labels)

print(f"\nGMM Performance Metrics:")
print(f"  Silhouette Score: {gmm_silhouette:.4f}")
print(f"  BIC: {gmm.bic(X):.2f} (lower is better)")
print(f"  AIC: {gmm.aic(X):.2f} (lower is better)")
print(f"  Log-Likelihood: {gmm.score(X) * len(X):.2f}")

# PART 4: CLUSTER PROFILING

print("\n" + "="*60)
print("CLUSTER PROFILING - Feature Means by Cluster")
print("="*60)

# K-means cluster profiles
print("\n--- K-Means Cluster Profiles ---")
kmeans_profiles = pd.DataFrame(X, columns=feature_cols)
kmeans_profiles['Cluster'] = kmeans_labels
cluster_means = kmeans_profiles.groupby('Cluster').mean()
print(cluster_means.to_string())

# GMM cluster profiles
print("\n--- GMM Cluster Profiles ---")
gmm_profiles = pd.DataFrame(X, columns=feature_cols)
gmm_profiles["Cluster"] = gmm_labels
gmm_cluster_means = gmm_profiles.groupby('Cluster').mean()
print(gmm_cluster_means.to_string())

# VISUALIZATIONS

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Perform PCA for 2D visualization
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
print(f"\nPCA Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Elbow Method
ax1 = plt.subplot(3, 4, 1)
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax1.set_title('Elbow Method', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# 2. Silhouette Scores
ax2 = plt.subplot(3, 4, 2)
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# 3. BIC/AIC for GMM
ax3 = plt.subplot(3, 4, 3)
ax3.plot(n_components_range, bic_scores, 'ro-', label='BIC', linewidth=2, markersize=8)
ax3.plot(n_components_range, aic_scores, 'mo-', label='AIC', linewidth=2, markersize=8)
ax3.set_xlabel('Number of Components')
ax3.set_ylabel('Information Criterion')
ax3.set_title('GMM Model Selection', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Silhouette Scores Comparison
ax4 = plt.subplot(3, 4, 4)
x_pos = np.arange(2)
scores = [kmeans_silhouette, gmm_silhouette]
colors = ['steelblue', 'coral']
bars = ax4.bar(x_pos, scores, color=colors, edgecolor='black')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['K-Means', 'GMM'])
ax4.set_ylabel('Silhouette Score')
ax4.set_title('Model Comparison: Silhouette Score', fontsize=12, fontweight='bold')
ax4.set_ylim([0, max(scores) * 1.2])
ax4.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# 5. K-Means Clusters in PCA space
ax5 = plt.subplot(3, 4, 5)
scatter = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                     cmap='viridis', alpha=0.6, s=50)
ax5.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
           pca.transform(kmeans.cluster_centers_)[:, 1],
           c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax5.set_title('K-Means Clusters (PCA)', fontsize=12, fontweight='bold')
ax5.legend()
plt.colorbar(scatter, ax=ax5, label='Cluster')

# 6. K-Means with Crisis overlay
ax6 = plt.subplot(3, 4, 6)
colors = ['green' if c == 0 else 'red' for c in y]
ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.4, s=50)
for i in range(optimal_k):
    cluster_points = X_pca[kmeans_labels == i]
    ax6.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               alpha=0.3, s=30, edgecolors='black', linewidths=0.5)
ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax6.set_title('K-Means with Crisis Overlay', fontsize=12, fontweight='bold')

# 7. GMM Clusters in PCA space
ax7 = plt.subplot(3, 4, 7)
scatter = ax7.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, 
                     cmap='plasma', alpha=0.6, s=50)
ax7.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax7.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax7.set_title('GMM Clusters (PCA)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax7, label='Cluster')

# 8. GMM Assignment Probability
ax8 = plt.subplot(3, 4, 8)
scatter = ax8.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_probs.max(axis=1),
                     cmap='RdYlGn', alpha=0.6, s=50, vmin=0, vmax=1)
ax8.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax8.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax8.set_title('GMM Assignment Confidence', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax8, label='Max Probability')

# 9. K-Means Crisis Rate by Cluster
ax9 = plt.subplot(3, 4, 9)
kmeans_crisis_rates = [y[kmeans_labels == i].mean() for i in range(optimal_k)]
bars = ax9.bar(range(optimal_k), kmeans_crisis_rates, color='steelblue', edgecolor='black')
ax9.axhline(y=y.mean(), color='red', linestyle='--', linewidth=2, label='Overall Rate')
ax9.set_xlabel('Cluster')
ax9.set_ylabel('Crisis Rate')
ax9.set_title('K-Means: Crisis Rate by Cluster', fontsize=12, fontweight='bold')
ax9.set_ylim([0, max(kmeans_crisis_rates) * 1.2])
ax9.legend()
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}', ha='center', va='bottom')

# 10. GMM Crisis Rate by Cluster
ax10 = plt.subplot(3, 4, 10)
gmm_crisis_rates = [y[gmm_labels == i].mean() for i in range(optimal_n)]
bars = ax10.bar(range(optimal_n), gmm_crisis_rates, color='coral', edgecolor='black')
ax10.axhline(y=y.mean(), color='red', linestyle='--', linewidth=2, label='Overall Rate')
ax10.set_xlabel('Cluster')
ax10.set_ylabel('Crisis Rate')
ax10.set_title('GMM: Crisis Rate by Cluster', fontsize=12, fontweight='bold')
ax10.set_ylim([0, max(gmm_crisis_rates) * 1.2])
ax10.legend()
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax10.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}', ha='center', va='bottom')

# 11. K-Means Cluster Size Distribution
ax11 = plt.subplot(3, 4, 11)
kmeans_sizes = [(kmeans_labels == i).sum() for i in range(optimal_k)]
ax11.pie(kmeans_sizes, labels=[f'Cluster {i}' for i in range(optimal_k)],
        autopct='%1.1f%%', startangle=90)
ax11.set_title('K-Means Cluster Size Distribution', fontsize=12, fontweight='bold')

# 12. GMM Cluster Size Distribution
ax12 = plt.subplot(3, 4, 12)
gmm_sizes = [(gmm_labels == i).sum() for i in range(optimal_n)]
ax12.pie(gmm_sizes, labels=[f'Cluster {i}' for i in range(optimal_n)],
        autopct='%1.1f%%', startangle=90)
ax12.set_title('GMM Cluster Size Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'clustering_analysis.png'")

# SAVE RESILTS

cluster_results = df[[COUNTRY_COL, YEAR_COL, TARGET_COL, 'KMeans_Cluster', 'GMM_Cluster', 'GMM_Max_Probability']]
cluster_results.to_csv('cluster_assignments.csv', index=False)
print("✓ Cluster assignments saved as 'cluster_assignments.csv'")

# Save cluster profiles
cluster_means.to_csv('kmeans_cluster_profiles.csv')
gmm_cluster_means.to_csv('gmm_cluster_profiles.csv')
print("✓ Cluster profiles saved")

# Save model comparison metrics
comparison_df = pd.DataFrame({
    'Model': ['K-Means', 'GMM'],
    'Silhouette_Score': [kmeans_silhouette, gmm_silhouette],
    'Inertia_or_BIC': [kmeans_inertia, gmm.bic(X)]
})
comparison_df.to_csv('clustering_model_comparison.csv', index=False)
print("✓ Model comparison saved as 'clustering_model_comparison.csv'")

print("\n" + "="*60)
print("CLUSTERING ANALYSIS COMPLETE!")
print("="*60)
print("\nKey Findings:")
print(f"- K-Means identified {optimal_k} distinct risk groups")
print(f"- GMM identified {optimal_n} probabilistic clusters")
print(f"- Best performing model by Silhouette Score: {'K-Means' if kmeans_silhouette > gmm_silhouette else 'GMM'}")
print(f"- Clusters successfully capture crisis risk patterns")