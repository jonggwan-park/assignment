import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Mall_Customers.csv")

# customer id는 유의미한 데이터가 아닌것으로 판단되어
df = df.drop('CustomerID',axis=1)


# 결측치 처리
# print(df.info())
# 결측치 없음

# 데이터 변환
#  성별을 0, 1로 변환
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
'''
## scatterplot Gender vs Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Gender', y='Spending Score (1-100)')
plt.title('Gender vs Spending Score')
plt.show()

## scatterplot Age vs Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)')
plt.title('Age vs Spending Score')
plt.show()


## scatter plot Annual Income vs spending score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')
plt.title('Annual Income vs Spending Score')
plt.show()
'''
# correlate Each feature with spending Score
correlation_matrix = df.corr()
#print(correlation_matrix.head())

#print("\ndf.describe:")
#print(df.describe())

# AGE 별로 세분화

## Create new features with Age(10- 19, 20 - 29, 30- 39, 40- 49, 50- 59, 60- 69, 70 +)
age_bins = [10, 20, 30, 40, 50, 60, 70, 80]

## split by age_bins
# Changed 'age' to 'age_bins' in the bins argument
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'])
df['Age_Group'] = df['Age_Group'].astype(str)
df.head()
# view as histogram
df['Age_Group'].value_counts()


# 각 범주의 값이 몇번 등장했는지 세어줌.
df['Age_Group2'] = df['Age_Group'].replace(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'], ['10', '20', '30', '40', '50', '60', '70'])
## make Age_Group2 as int
df['Age_Group2'] = df['Age_Group2'].astype(int)
# Agegroup2를 int로 만듦



''' 
# 시각화
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Age_Group')
plt.title('Age Group Distribution')
plt.show()

# Spendingscore에 따른 Age Group 분포
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Age_Group', hue='Spending Score (1-100)')
plt.title('Age Group Distribution with Spending Score')

# Gender에 따른 Age Group 분포
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Age_Group', hue='Gender')
plt.title('Age Group Distribution with Gender')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.xticks(rotation=45)
plt.show()
'''

## Make spending score 10,20,30,40,50,60,70,80,90,100 and create new column name Spend
df['Spend'] = pd.cut(df['Spending Score (1-100)'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'])
df

## Annual Income Group by 15- 30, 30- 50, 50-100, 100+
income_bins = [10, 30, 50, 100, 150]
df['Income_Group'] = pd.cut(df['Annual Income (k$)'], bins=income_bins, labels=['10-30', '30-50', '50-100', '100+'])
'''
#시각화
## Create histogram between Annual income group and spending score
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Income_Group', hue='Spend', multiple='stack')
plt.title('Annual Income Group Distribution with Spending Score')
plt.show()

# legend with Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Income_Group', hue='Gender')
plt.title('Annual Income Group Distribution with Gender')
plt.show()


## Create histogram between AgeGroup2 and spend
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age_Group2', hue='Spend', multiple='stack')
plt.title('Age Group Distribution with Spending Score')
plt.show()
# legend by Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Age_Group2', hue='Gender')
plt.title('Age Group Distribution with Gender')
plt.show()
'''
# 표준화(Standardization) 또는 정규화(Normalization)

from sklearn.preprocessing import StandardScaler

# 연속형 데이터 스케일링
scaler = StandardScaler()
numerical_columns = ['Gender','Age','Annual Income (k$)', 'Spending Score (1-100)']
data_numerical = df[numerical_columns]

scaled_df = df[numerical_columns].copy()
scaled_df=scaler.fit_transform(data_numerical)
scaled_df=pd.DataFrame(scaled_df,columns=numerical_columns)
#print(scaled_df.head())
#print(scaled_df.describe())

# 차원 축소

from sklearn.decomposition import PCA
# 여기서 Gender는 뺐다.
numerical_columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
pca = PCA(n_components=3) 
reduced_data = pca.fit_transform(scaled_df)

explained_variance = pca.explained_variance_ratio_
#print("Explained variance ratio by each component:", explained_variance)
#print("Cumulative explained variance:", explained_variance.cumsum())
# 0.8318로 낮은 수준으로 보인다.

# 차원 축소 후 데이터프레임 생성
pca_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2','PCA3'])
#print(pca_df.head())
'''
# 설명 분산 비율 누적 그래프
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()
'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)

'''
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k')
plt.title("PCA: First Two Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# 3D PCA 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=kmeans.labels_, cmap='rainbow', edgecolor='k')
ax.set_title("PCA: First Three Principal Components")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()
'''

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
data_tsne = tsne.fit_transform(scaled_df)
#print(f"Shape of t-SNE result: {data_tsne.shape}")

'''
plt.figure(figsize=(8, 6))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=kmeans.labels_, cmap='rainbow', edgecolor='k', alpha=0.7)
plt.title("t-SNE Visualization (2D)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.colorbar(label="Cluster Label")
plt.show()
'''
# 각 군집의 평균 값을 계산
# Select only numeric columns before calculating the mean

# Add cluster labels to the original dataset for visualization
scaled_df['Cluster']= kmeans.fit_predict(scaled_df)
df['Cluster'] = scaled_df['Cluster']
df.drop(['Gender','Age_Group2'],axis=1,inplace=True)
numeric_df = df.select_dtypes(include=np.number)
cluster_summary = numeric_df.groupby('Cluster').mean()
#print(cluster_summary)


from sklearn.metrics import silhouette_score

# 실루엣 점수 계산
silhouette_avg = silhouette_score(scaled_df, kmeans.labels_)
#print(f"Silhouette Score: {silhouette_avg}")
#0.6075
'''
# 시각화
import seaborn as sns

# Cluster vs Spending Score 분포
sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=df)
plt.title("Spending Score Distribution by Cluster")
plt.show()

# Cluster vs Age 분포
sns.boxplot(x='Cluster', y='Age', data=df)
plt.title("Age Distribution by Cluster")
plt.show()
'''
########### 계층적 군집화

from scipy.cluster.hierarchy import dendrogram, linkage
'''
linkage_matrix = linkage(scaled_df, method='ward')
dendrogram(linkage_matrix)
plt.title("Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()
'''
############# DBSCAN

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_df)
'''
# 결과 시각화
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=df['DBSCAN_Cluster'], cmap='rainbow', edgecolor='k')
plt.title("t-SNE with DBSCAN Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
'''
############# k-means
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
scaled_df['Cluster'] = kmeans.fit_predict(scaled_df)

# Add cluster labels to the original dataset for visualization
df['Cluster'] = scaled_df['Cluster']

'''
# 시각화

# Plot scatter plots for clusters
plt.figure(figsize=(15, 5))

# Scatter plot for Age vs Annual Income
plt.subplot(1, 2, 1)
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Age'], label=f"Cluster {cluster}")
plt.title('Clusters: Age vs Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Age')
plt.legend()


# Scatter plot for Annual Income vs Spending Score
plt.subplot(1, 2, 2)
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f"Cluster {cluster}")
plt.title('Clusters: Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.tight_layout()
plt.show()

'''
'''
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Initialize DBSCAN with a range of epsilon values to test
eps_values = [0.5, 1.0, 1.5, 2.0]
min_samples = 5  # Default minimum samples for DBSCAN

# Dictionary to store results for each epsilon value
dbscan_results = {}

# Use scaled_df
for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_df)  # Changed here

    # Calculate the number of clusters (excluding noise, -1 label)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Silhouette score (only if there are at least 2 clusters)
    if n_clusters > 1:
        silhouette_avg = silhouette_score(scaled_df, labels)  # Changed here
    else:
        silhouette_avg = "N/A"

    dbscan_results[eps] = {"Clusters": n_clusters, "Silhouette Score": silhouette_avg}

# Convert results to DataFrame for clarity
dbscan_results_df = pd.DataFrame(dbscan_results).T
dbscan_results_df.index.name = "Epsilon (eps)"

#print(dbscan_results_df)

#Remove the extra space before dbscan
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(scaled_df)  # Changed here

# Add cluster labels to the DataFrame for visualization
scaled_df['Cluster'] = labels  # Changed here

# Scatter plot of clusters (using the first two dimensions for 2D visualization)
plt.figure(figsize=(8, 6))
plt.scatter(scaled_df['Age'], scaled_df['Annual Income (k$)'], c=labels, cmap='rainbow', s=50)  # Changed here
plt.title('DBSCAN Clustering Visualization')
plt.xlabel('Age (scaled)')
plt.ylabel('Annual Income (scaled)')
plt.colorbar(label='Cluster Label')
plt.show()
'''
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# 차원 축소
pca = PCA(n_components=2)
scaled_df_2d = pca.fit_transform(scaled_df)

# 여러가지 클러스터링 적용
# (1) K-means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_df)

# (2) DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_df)

# (3) Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(scaled_df)

# (4) Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(scaled_df)

# 실루엣스코어 계산
algorithms = ['K-means', 'DBSCAN', 'Hierarchical', 'GMM']
labels_list = [kmeans_labels, dbscan_labels, hierarchical_labels, gmm_labels]

silhouette_scores = []
for labels in labels_list:
    if len(set(labels)) > 1:  # 클러스터가 2개 이상일 때만 계산
        score = silhouette_score(scaled_df, labels)
    else:
        score = np.nan  # DBSCAN처럼 하나의 클러스터만 생성된 경우
    silhouette_scores.append(score)

'''
# 시각화
plt.figure(figsize=(16, 12))

# K-means
plt.subplot(2, 2, 1)
plt.scatter(scaled_df_2d[:, 0], scaled_df_2d[:, 1], c=kmeans_labels, cmap='rainbow', s=30)
plt.title('K-means Clustering')

# DBSCAN
plt.subplot(2, 2, 2)
plt.scatter(scaled_df_2d[:, 0], scaled_df_2d[:, 1], c=dbscan_labels, cmap='rainbow', s=30)
plt.title('DBSCAN Clustering')

# Hierarchical
plt.subplot(2, 2, 3)
plt.scatter(scaled_df_2d[:, 0], scaled_df_2d[:, 1], c=hierarchical_labels, cmap='rainbow', s=30)
plt.title('Hierarchical Clustering')

# GMM
plt.subplot(2, 2, 4)
plt.scatter(scaled_df_2d[:, 0], scaled_df_2d[:, 1], c=gmm_labels, cmap='rainbow', s=30)
plt.title('Gaussian Mixture Model (GMM)')

plt.tight_layout()
plt.show()   

'''
'''
# Silhouette Score 출력
for algo, score in zip(algorithms, silhouette_scores):
    print(f"{algo}: Silhouette Score = {score if not np.isnan(score) else 'Not Applicable'}")
'''



################ 최적 클러스터 수 결정

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Step 1: 데이터 준비 (정규화된 데이터 사용)
data = scaled_df.values  # 정규화된 데이터

# Step 2: 클러스터 수 범위 정의
cluster_range = range(2, 11)  # 2에서 10까지의 클러스터 수 탐색

# 결과 저장용
kmeans_inertia = []
kmeans_silhouette = []
gmm_silhouette = []
hierarchical_silhouette = []

# Step 3: 최적 k 탐색
for k in cluster_range:
    # K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(data)
    kmeans_inertia.append(kmeans.inertia_)
    kmeans_silhouette.append(silhouette_score(data, kmeans_labels))
    
    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm_labels = gmm.fit_predict(data)
    gmm_silhouette.append(silhouette_score(data, gmm_labels))
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=k)
    hierarchical_labels = hierarchical.fit_predict(data)
    hierarchical_silhouette.append(silhouette_score(data, hierarchical_labels))



##################### 결과 시각화
'''
plt.figure(figsize=(14, 6))

# K-means: Elbow Method
plt.subplot(1, 2, 1)
plt.plot(cluster_range, kmeans_inertia, marker='o')
plt.title('K-means: Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid()

# K-means: Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(cluster_range, kmeans_silhouette, marker='o', label='K-means')
plt.plot(cluster_range, gmm_silhouette, marker='o', label='GMM')
plt.plot(cluster_range, hierarchical_silhouette, marker='o', label='Hierarchical')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

'''

# 고객 행동 예측 모델 (구매가능성)


# 시계열 분석