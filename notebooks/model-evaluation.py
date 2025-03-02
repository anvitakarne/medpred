import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import scipy.stats as stats
from sklearn.decomposition import PCA
from numpy.linalg import cond, svd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier


import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier

# Read the dataset
data_df = pd.read_csv("medical_malpractice.csv")

print("First five rows of the data: ")
print(data_df.head())

print("Last five rows of the data: ")
print(data_df.tail())

print("Dimensionalty of the dataset: ")
print(data_df.shape)

print("Information related to the dataset: ")
print(data_df.info())

print("Columns of the dataset: ")
print(data_df.columns)

print(data_df.dtypes)

print("Display the value counts of the 'Amount' column: ")
print(data_df['Amount'].value_counts())

print("Display the value counts of the 'Insurance' column: ")
print(data_df['Insurance'].value_counts())

print("Display the value counts of the 'Age' column: ")
print(data_df['Age'].value_counts())

print("Display the value counts of the 'Specialty' column: ")
print(data_df['Specialty'].value_counts())

print("Display the value counts of the 'Severity' column: ")
print(data_df['How_Severe'].value_counts())

print("Display the value counts of the 'Private Attorney' column: ")
print(data_df['Private Attorney'].value_counts())

print("Display the value counts of the 'Gender' column: ")
print(data_df['Gender'].value_counts())

print("Display the value counts of the 'Marital Status' column: ")
print(data_df['Marital Status'].value_counts())

print("Display null values in the dataset: ")
print(data_df.isnull())

print("Display count of missing values in each column of the dataset: ")
print(data_df.isnull().sum())

print("Display number of distinct observations in the dataset: ")
print(data_df.nunique())

print("Identify and display duplicate rows in the dataset: ")
print(data_df.duplicated())


# Display basic statistics of the dataset
print(data_df.describe())

# Display the distribution of the target variable (Severity)
sns.countplot(x='How_Severe', data=data_df)
plt.title('Distribution of Severity')
plt.show()

# Explore the distribution of claim amounts
plt.figure(figsize=(10, 6))
sns.histplot(data_df['Amount'], bins=30, kde=True)
plt.title('Distribution of Claim Amounts')
plt.xlabel('Claim Amount')
plt.ylabel('Frequency')
plt.show()

# Explore the relationship between age and claim amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Amount', data=data_df, hue='How_Severe')
plt.title('Relationship between Age and Claim Amount with Severity')
plt.xlabel('Age')
plt.ylabel('Claim Amount')
plt.show()

# Explore the distribution of claim amounts by specialty
plt.figure(figsize=(14, 8))
sns.boxplot(x='Specialty', y='Amount', data=data_df)
plt.title('Distribution of Claim Amounts by Specialty')
plt.xlabel('Specialty')
plt.ylabel('Claim Amount')
plt.xticks(rotation=45, ha='right')
plt.show()

# Explore the relationship between severity and the presence of a private attorney
plt.figure(figsize=(8, 5))
sns.countplot(x='How_Severe', hue='Private Attorney', data=data_df)
plt.title('Severity vs. Private Attorney Representation')
plt.show()


data_df.replace({'Male':0, 'Female':1},inplace=True)
data_df.replace({'Normal':0, 'High':1},inplace=True)

categorical_data = data_df.select_dtypes(exclude=np.number).columns
print("Categorical features: ", categorical_data)

data_df_encoded = pd.get_dummies(data_df, columns=categorical_data)
print("Encoded features: ", data_df_encoded)
print(data_df_encoded.columns)

print("---------------------------------- K MEANS CLUSTERING ----------------------------------")
print("WCSS Method: ")
selected_columns = ['Amount', 'How_Severe', 'Age']
data = data_df[selected_columns]
data = pd.get_dummies(data, columns=['How_Severe'], prefix=['Severity'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
data_df['Cluster'] = kmeans.fit_predict(scaled_data)
print(data_df[['Amount', 'How_Severe', 'Age', 'Cluster']])

print("Silhouette Method: ")
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
data_df['Cluster'] = kmeans.fit_predict(scaled_data)
print(data_df[['Amount', 'How_Severe', 'Age', 'Cluster']])



print("---------------------------------- APRIORI ALGORITHM ----------------------------------")
pd.set_option('display.max_columns',100)
df = pd.read_csv('medical_malpractice.csv')
# Assuming your dataset is stored in a variable called 'df'
selected_columns = ['Specialty', 'How_Severe', 'Insurance', 'Gender', 'Private Attorney']
data = df[selected_columns]
data_encoded = pd.get_dummies(data, sparse=True)
frequent_itemsets = apriori(data_encoded, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)