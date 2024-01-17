import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.decomposition import PCA

# Load Dataset
data = pd.read_csv(r"heart_failure_clinical_records_dataset.csv")


# Method 1: use all 12 features
# Drop the target class from the dataset
X = data.drop(['DEATH_EVENT'], axis = 1)

# Method 2: select two of the best features
X2 = data[['ejection_fraction', 'serum_creatinine']]
 
# Independently standardized each individual feature(column) with StandardScaler()
# Fit and transform the data
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
X2_scaled = scalar.fit_transform(X2)
 
# Use elbow method to find the optimal k
# Calculate the sum of squared errors 

'''
# Method 1
sse = []
rng = range(1, 50) 
for k in rng:
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    sse.append(km.inertia_)

# Method 2
sse2 = []
rng = range(1, 50) 
for k in rng:
    km = KMeans(n_clusters=k)
    km.fit(X2_scaled)
    sse2.append(km.inertia_)
    
# Plot the sum of squared errors curve
plt.xlabel('k')
plt.ylabel('Sum of squared error')
plt.plot(rng, sse2, label='Ejection Fraction & Serum Creatinine')
plt.plot(rng, sse, label='All 12 features')
plt.legend() 
plt.show()
'''
# Choose Method 2
# Use K-Means for clustering
num_clusters = 5
km = KMeans(n_clusters=num_clusters, random_state=22)

# Fit and predict the model
ypred = km.fit_predict(X2_scaled)

# Plot the results
# Filter rows of X2_scaled data
# Get the unique labels
clusters = np.unique(ypred)
 
for i in clusters:
    plt.scatter(X2_scaled[ypred == i , 0] , X2_scaled[ypred == i , 1] , label = i)
# Plot the centriods
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], color='black', marker='*')
plt.legend()
plt.show()