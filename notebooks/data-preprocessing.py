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
from sklearn.preprocessing import StandardScaler
from numpy.linalg import cond, svd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

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

# Import necessary libraries for EDA
import matplotlib.pyplot as plt
import seaborn as sns

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

def standardize(data):
  mean = np.mean(data)
  std = np.std(data)
  standardized_data = (data - mean) / std
  return standardized_data
data_df_encoded['Amount'] = standardize(data_df_encoded['Amount'])
data_df_encoded['Age'] = standardize(data_df_encoded['Age'])
data_df_encoded['Amount'].head()

# Create a box plot for the 'Amount' feature
data_df['Amount'].describe()
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.boxplot(x=data_df['Amount'])
plt.title('Box Plot for Outlier Detection - Amount')
plt.show()

Q1 = data_df['Age'].quantile(0.25)
Q3 = data_df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data_df[(data_df['Age'] < lower_bound) | (data_df['Age'] > upper_bound)]
data_df = data_df[(data_df['Age'] >= lower_bound) & (data_df['Age'] <= upper_bound)]
print("Statistics:")
print(data_df['Age'].describe())
plt.figure(figsize=(8, 6))
sns.boxplot(x=data_df['Age'])
plt.title('Box Plot - Age')
plt.show()

# Create a box plot for the 'Amount' feature
data_df['Age'].describe()
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.boxplot(x=data_df['Age'])
plt.title('Box Plot for Outlier Detection - Age')
plt.show()

Q1 = data_df['Amount'].quantile(0.25)
Q3 = data_df['Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data_df[(data_df['Amount'] < lower_bound) | (data_df['Amount'] > upper_bound)]
data_df = data_df[(data_df['Amount'] >= lower_bound) & (data_df['Amount'] <= upper_bound)]
print("Statistics:")
print(data_df['Amount'].describe())
plt.figure(figsize=(8, 6))
sns.boxplot(x=data_df['Amount'])
plt.title('Box Plot - Amount')
plt.show()

categorical_data = data_df.select_dtypes(include=['object']).columns
print(categorical_data)

data_df_encoded = pd.get_dummies(data_df, columns=categorical_data)
print(data_df_encoded)

print(data_df_encoded.columns)

print(data_df_encoded.info())

def standardize(data):
  mean = np.mean(data)
  std = np.std(data)
  standardized_data = (data - mean) / std
  return standardized_data

data_df_encoded['Amount'] = standardize(data_df_encoded['Amount'])
data_df_encoded['Age'] = standardize(data_df_encoded['Age'])

y = data_df_encoded['Amount']
X = data_df_encoded.drop(['Amount'],axis=1)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=5805)
rf_regressor.fit(X, y)

feature_importance = rf_regressor.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.show()

N = 10
top_features = feature_importance_df.head(N)
print(top_features)
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest Model')
plt.show()

print("PCA for Regression: ")
pca = PCA(random_state= 5805)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
num_components_needed = np.argmax(cumulative_explained_variance > 0.90) + 1
print(f"No of principal components needed to explain more than 90% of the variance: {num_components_needed}")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='b')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.xlabel('No of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
plt.axvline(x=num_components_needed, color='g', linestyle='--', label=f'{num_components_needed} Components')
plt.legend()
plt.grid(True)
plt.show()
covariance_matrix = pca.get_covariance()
condition_number = cond(covariance_matrix)
print(f"Condition number before passing n components:{condition_number}")
pca_2 = PCA(n_components=num_components_needed)
X_pca = pca_2.fit_transform(X)
cov_matrix = pca_2.get_covariance()
condition_number = cond(cov_matrix)
print(f"Condition number after passing n components :{condition_number}")

print("SVD for Regression: ")
explained_variance_ratio = []
for n in range(1, X.shape[1]):
    svd = TruncatedSVD(n_components=n)
    svd.fit(X)
    explained_variance_ratio.append(svd.explained_variance_ratio_.sum())
plt.figure(figsize=(10, 6))
plt.plot(range(1, X.shape[1]), explained_variance_ratio, marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratio vs. Number of Components (TruncatedSVD)')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
plt.axvline(x=num_components_needed, color='g', linestyle='--', label=f'{num_components_needed} Components')
plt.legend()
plt.grid(True)
plt.show()

print(X.describe())



threshold = 0.001
selected_features_df = feature_importance_df[feature_importance_df["Importance"] >= threshold]
eliminated_features_df = feature_importance_df[feature_importance_df["Importance"] < threshold]

# Display selected and eliminated features
print("Selected Features:")
print(selected_features_df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)


print("Linear Regression")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
r2_train = round(r2_score(y_train, y_pred_train),2)
r2_test = round(r2_score(y_test, y_pred_test),2)
print(f'R-squared on training set: {r2_train}')
print(f'R-squared on testing set: {r2_test}')
mse_train = round(mean_squared_error(y_train, y_pred_train),2)
mse_test = round(mean_squared_error(y_test, y_pred_test),2)
print(f'Mean Squared Error (MSE) on training set: {mse_train}')
print(f'Mean Squared Error (MSE) on testing set: {mse_test}')
rmse_train = round(mean_squared_error(y_train, y_pred_train, squared=False),2)
rmse_test = round(mean_squared_error(y_test, y_pred_test, squared=False),2)
print(f'Root Mean Squared Error on training set: {rmse_train}')
print(f'Root Mean Squared Error on testing set: {rmse_test}')

print("Stepwise Regression OLS")
trainCols = list(X_train.columns)
featuresDropped = []
threshold = 0.01
resultsTable = pd.DataFrame(columns=['Feature Dropped', 'AIC', 'BIC', 'AdjR2', 'P-Value'])
while len(trainCols) > 0:
    X_train = X_train[trainCols]
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    pValues = model.pvalues[1:]
    maxPVal = pValues.max()
    maxPValIdx = pValues.idxmax()

    if maxPVal > threshold:
        dropFeature = maxPValIdx
        featuresDropped.append(dropFeature)
        trainCols.remove(dropFeature)
        print('Feature Dropped:', dropFeature)
        print(model.summary())

        f_statistic = model.fvalue.round(2)
        f_p_value = model.f_pvalue
        aic = model.aic.round(2)
        bic = model.bic.round(2)
        adj_r_squared = model.rsquared_adj.round(2)
        confidence_intervals = round(model.conf_int()), 2

        resultsTable = resultsTable.append({
            'Feature Dropped': dropFeature,
            'AIC': aic,
            'BIC': bic,
            'AdjR2': adj_r_squared,
            'P-Value': maxPVal,
            'F-Value': f_statistic
        }, ignore_index=True)

        print(f"F-statistic: {f_statistic}")
        print(f"P-value (F-test): {f_p_value}")
        print(f"AIC: {aic}")
        print(f"BIC: {bic}")
        print(f"Adjusted R-squared: {adj_r_squared}")
        print("Confidence Intervals:")
        print(confidence_intervals)

    else:
        break

print("\nResults Table:")
print(resultsTable)
print(X_train.shape)
print(X_test.shape)

common_features = set(X_train.columns) & set(X_test.columns)
X_train = X_train[common_features]
X_test = X_test[common_features]
model1 = sm.OLS(y_train, X_train).fit()

sm_pred = model1.get_prediction(X_test).summary_frame(alpha=0.05)
lower_interval = sm_pred['obs_ci_lower']
upper_interval = sm_pred['obs_ci_upper']
x_range = [i for i in range(len(y_test))]
plt.plot(x_range, lower_interval, alpha=0.4, label='Lower interval')
plt.plot(x_range, upper_interval, alpha=0.4, label='Upper interval')
plt.plot(x_range, y_test, alpha=1.0, label='predicted Values')
plt.title('Predicted Values with Intervals')
plt.ylabel('Amount')
plt.xlabel('No.of samples')
plt.legend()
plt.show()

train_range = [i for i in range(len(y_train))]
test_range = [i for i in range(y_test.shape[0])]
plt.plot(train_range, y_train, label='Training Amount')
plt.plot(test_range, y_test, label='Test Amount')
plt.plot(test_range, y_pred_test, label='Predicted Amount')
plt.xlabel('Observations')
plt.ylabel('Amount')
plt.legend(loc='best')
plt.title('Linear Regression OLS')
plt.show()
