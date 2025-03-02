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
from sklearn.decomposition import TruncatedSVD
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
print(categorical_data)

data_df_encoded = pd.get_dummies(data_df, columns=categorical_data)
print(data_df_encoded)

print(data_df_encoded.columns)

def standardize(data):
  mean = np.mean(data)
  std = np.std(data)
  standardized_data = (data - mean) / std
  return standardized_data

data_df_encoded['Amount'] = standardize(data_df_encoded['Amount'])
data_df_encoded['Age'] = standardize(data_df_encoded['Age'])

print(data_df_encoded['Amount'].head())

# Display the results of Feature importance using Random Forest Analysis
X = data_df_encoded.drop('How_Severe', axis=1)
y = data_df_encoded['How_Severe']
rf_classifier = RandomForestClassifier(n_estimators=100,max_features='log2', random_state=5805)
rf_classifier.fit(X, y)
feature_importance = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.show()

threshold = 0.01
selected_features_df = feature_importance_df[feature_importance_df["Importance"] >= threshold]
eliminated_features_df = feature_importance_df[feature_importance_df["Importance"] < threshold]
# Display selected and eliminated features
print("Selected Features:")
print(selected_features_df)

selected_features = selected_features_df['Feature'].values
data_df = data_df_encoded[selected_features]

print("PCA")
pca = PCA(random_state= 5805)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
num_components_needed = np.argmax(cumulative_explained_variance > 0.90) + 1
print(f"Number of principal components needed to explain more than 90% of the variance: {num_components_needed}")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='b')
plt.title('Cumulative Explained Variance vs. Number of Components (PCA) ')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
plt.axvline(x=num_components_needed, color='g', linestyle='--', label=f'{num_components_needed} Components')
plt.legend()
plt.grid(True)
plt.show()
covariance_matrix = pca.get_covariance()
condition_number = cond(covariance_matrix)
print(f"Condition number before passing n_components:{condition_number}")
pca_2 = PCA(n_components=num_components_needed)
X_pca = pca_2.fit_transform(X)
covariance_matrix = pca_2.get_covariance()
condition_number = cond(covariance_matrix)
print(f"Condition number after passing n_components :{condition_number}")

print("SVD")
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

print(y.value_counts())
print(X.shape)

from imblearn.over_sampling import SMOTE
os  = SMOTE(random_state=5805)
X,y = os.fit_resample(X,y)

from imblearn.under_sampling import RandomUnderSampler
us = RandomUnderSampler(sampling_strategy= {0:10000,1:10000}, random_state=5805)
X, y = us.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=5805)
print(X_train[:5])
print(y_train[:5])
print(X_test.shape)

print("-------------------------  BASIC DECISION TREE -------------------------")
basic_decisiontree_classifier = DecisionTreeClassifier(random_state=5805)
basic_decisiontree_classifier.fit(X_train, y_train)
basic_decisiontree_y_pred = basic_decisiontree_classifier.predict(X_test)
basic_decisiontree_y_prob = basic_decisiontree_classifier.predict_proba(X_test)[::, -1]
basic_decisiontree_accuracy = round(accuracy_score(y_test, basic_decisiontree_y_pred),2)
basic_decisiontree_recall = round(recall_score(y_test, basic_decisiontree_y_pred),2)
basic_decisiontree_roc_auc = round(roc_auc_score(y_test, basic_decisiontree_y_prob),2)
basic_decisiontree_confusion_matrix = confusion_matrix(y_test, basic_decisiontree_y_pred)
print("Basic Tree - Accuracy:", basic_decisiontree_accuracy)
print("Basic Tree - Recall:", basic_decisiontree_recall)
print("Basic Tree - ROC & AUC:", basic_decisiontree_roc_auc)
print("Basic Tree - Confusion Matrix:")
print(basic_decisiontree_confusion_matrix)
sns.heatmap(basic_decisiontree_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Basic Decision Tree')
plt.show()
basic_decisiontree_fpr, basic_decisiontree_tpr, basic_decisiontree_thresholds = roc_curve(y_test, basic_decisiontree_y_pred)
basic_decisiontree_auc = roc_auc_score(y_test, basic_decisiontree_y_pred)
plt.figure(figsize=(8, 6))
plt.plot(basic_decisiontree_fpr, basic_decisiontree_tpr, color='blue', label=f"Basic DT (AUC = {basic_decisiontree_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Basic Decision Tree')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = basic_decisiontree_confusion_matrix[1][1]
TN = basic_decisiontree_confusion_matrix[0][0]
FP = basic_decisiontree_confusion_matrix[0][1]
FN = basic_decisiontree_confusion_matrix[1][0]
basic_decisiontree_precision = round(TP / (TP + FP), 2)
basic_decisiontree_recall = round(TP / (TP + FN), 2)
basic_decisiontree_f1_score = round(2 * (basic_decisiontree_precision * basic_decisiontree_recall) / (basic_decisiontree_precision + basic_decisiontree_recall), 2)
basic_decisiontree_specificity = round(TN / (TN + FP), 2)
print("Basic Decision Tree - Precision :", basic_decisiontree_precision)
print("Basic Decision Tree - Recall :", basic_decisiontree_recall)
print("Basic Decision Tree - F1 score :", basic_decisiontree_f1_score)
print("Basic Decision Tree - Specificity :", basic_decisiontree_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
basic_decisiontree_stratified_k_fold = cross_val_score(basic_decisiontree_classifier, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = basic_decisiontree_stratified_k_fold.mean()
rounded_scores = [round(score, 2) for score in basic_decisiontree_stratified_k_fold]
print("Basic Decision Tree - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Basic Decision Tree - Stratified K-fold Cross Validation Mean AUC Score: ", round(mean_auc_score, 2))

print("-------------------------  PRE PRUNED DECISION TREE -------------------------")
tuned_parameters = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [3, 4, 5, 6, 7],
    'splitter': ['best', 'random'],
    'criterion': ['gini', 'entropy']
}
pre_pruned_GridSearch = DecisionTreeClassifier(random_state=5805)
pre_pruned_grid_search = GridSearchCV(pre_pruned_GridSearch, param_grid=tuned_parameters, cv=5, scoring='roc_auc')
pre_pruned_grid_search.fit(X_train, y_train)
pre_pruned_best_parameters = pre_pruned_grid_search.best_params_
best_pre_pruned_classifier = DecisionTreeClassifier(**pre_pruned_best_parameters, random_state=5805)
best_pre_pruned_classifier.fit(X_train, y_train)
pre_pruned_tree_y_pred = best_pre_pruned_classifier.predict(X_test)
pre_pruned_tree_y_prob = best_pre_pruned_classifier.predict_proba(X_test)[:, 1]
pre_pruned_accuracy = round(accuracy_score(y_test, pre_pruned_tree_y_pred), 2)
pre_pruned_recall = round(recall_score(y_test, pre_pruned_tree_y_pred), 2)
pre_pruned_roc_auc = round(roc_auc_score(y_test, pre_pruned_tree_y_prob), 2)
pre_pruned_confusion_matrix = confusion_matrix(y_test, pre_pruned_tree_y_pred)
print("Best Parameters:", pre_pruned_best_parameters)
print("Pre Pruned Tree - Accuracy", pre_pruned_accuracy)
print("Pre Pruned Tree - Recall", pre_pruned_recall)
print("Pre Pruned Tree - Roc & Auc", pre_pruned_roc_auc)
print("Pre Pruned Tree - Confusion Matrix")
print(pre_pruned_confusion_matrix)
sns.heatmap(pre_pruned_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Pre Pruned Decision Tree')
plt.show()
pre_pruned_decisiontree_fpr, pre_pruned_decisiontree_tpr, pre_pruned_decisiontree_thresholds = roc_curve(y_test, pre_pruned_tree_y_prob)
pre_pruned_decisiontree_auc = roc_auc_score(y_test, pre_pruned_tree_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(pre_pruned_decisiontree_fpr, pre_pruned_decisiontree_tpr, color='blue', label=f"Pre-Pruned DT (AUC = {pre_pruned_decisiontree_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Pre-Pruned Decision Tree')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = pre_pruned_confusion_matrix[1][1]
TN = pre_pruned_confusion_matrix[0][0]
FP = pre_pruned_confusion_matrix[0][1]
FN = pre_pruned_confusion_matrix[1][0]
pre_pruned_decisiontree_precision = round(TP / (TP + FP), 2)
pre_pruned_decisiontree_recall = round(TP / (TP + FN), 2)
pre_pruned_decisiontree_f1_score = round(2 * (pre_pruned_decisiontree_precision * pre_pruned_decisiontree_recall) / (pre_pruned_decisiontree_precision + pre_pruned_decisiontree_recall), 2)
pre_pruned_decisiontree_specificity = round(TN / (TN + FP), 2)
print("Pre Pruned Decision Tree - Precision :", pre_pruned_decisiontree_precision)
print("Pre Pruned Decision Tree - Recall :", pre_pruned_decisiontree_recall)
print("Pre Pruned Decision Tree - F1 score :", pre_pruned_decisiontree_f1_score)
print("Pre Pruned Decision Tree - Specificity :", pre_pruned_decisiontree_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
pre_pruned_decisiontree_stratified_k_fold = cross_val_score(pre_pruned_grid_search, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = pre_pruned_decisiontree_stratified_k_fold.mean()
rounded_scores = [round(score, 2) for score in pre_pruned_decisiontree_stratified_k_fold]
print("Pre Pruned Decision Tree - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Pre Pruned Decision Tree - Stratified K-fold Cross Validation Mean AUC Score: ", round(mean_auc_score, 2))

print("-------------------------  ALPHA -------------------------")
path1 = basic_decisiontree_classifier.cost_complexity_pruning_path(X_train, y_train)
alphas1 = path1['ccp_alphas']
print(alphas1)
print(len(alphas1))
print(alphas1)
f_alphas1 = [alpha for alpha in alphas1 if alpha > 1e-10]
accuracy_train, accuracy_test = [], []
for i in f_alphas1[200:400]:
    dt_classifier = DecisionTreeClassifier(ccp_alpha=i, random_state=5805)
    dt_classifier.fit(X_train, y_train)
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title("1. Accuracy vs alpha for training and testing sets")
ax.plot(f_alphas1[200:400], accuracy_train[0:200], marker="o", label="train", drawstyle="steps-post")
ax.plot(f_alphas1[200:400], accuracy_test[0:200], marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()

path2 = basic_decisiontree_classifier.cost_complexity_pruning_path(X_train, y_train)
alphas2 = path2['ccp_alphas']
print(alphas2)
print(len(alphas2))
print(alphas2)
f_alphas2 = [alpha for alpha in alphas2 if alpha > 1e-10]
accuracy_train, accuracy_test = [], []
for i in f_alphas2[1500:1900]:
    dt_classifier = DecisionTreeClassifier(ccp_alpha=i, random_state=5805)
    dt_classifier.fit(X_train, y_train)
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title("2. Accuracy vs alpha for training and testing sets")
ax.plot(f_alphas2[1500:1900], accuracy_train[0:400], marker="o", label="train", drawstyle="steps-post")
ax.plot(f_alphas2[1500:1900], accuracy_test[0:400], marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()

print("-------------------------  POST PRUNED DECISION TREE -------------------------")
post_pruned_tree = DecisionTreeClassifier(ccp_alpha=0.0001050, random_state=5805)
post_pruned_tree.fit(X_train, y_train)
post_pruned_y_pred = post_pruned_tree.predict(X_test)
post_pruned_y_prob = post_pruned_tree.predict_proba(X_test)[:, -1]
post_pruned_accuracy = round(accuracy_score(y_test, post_pruned_y_pred), 2)
post_pruned_confusion_matrix = confusion_matrix(y_test, post_pruned_y_pred)
post_pruned_recall = round(recall_score(y_test, post_pruned_y_pred), 2)
post_pruned_roc_auc = round(roc_auc_score(y_test, post_pruned_y_prob), 2)
print("Post Pruned Tree - Accuracy :", post_pruned_accuracy)
print("Post Pruned Tree - Recall :", post_pruned_recall)
print("Post Pruned Tree - ROC & AUC :", post_pruned_roc_auc)
print("Post Pruned Tree - Confusion Matrix:")
print(post_pruned_confusion_matrix)
sns.heatmap(post_pruned_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
post_pruned_decisiontree_fpr, post_pruned_decisiontree_tpr, post_pruned_decisiontree_thresholds = roc_curve(y_test, post_pruned_y_prob)
post_pruned_decisiontree_auc = roc_auc_score(y_test, post_pruned_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(post_pruned_decisiontree_fpr, post_pruned_decisiontree_tpr, color='blue', label=f"Post-Pruned DT (AUC = {post_pruned_decisiontree_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Post-Pruned Decision Tree')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = post_pruned_confusion_matrix[1][1]
TN = post_pruned_confusion_matrix[0][0]
FP = post_pruned_confusion_matrix[0][1]
FN = post_pruned_confusion_matrix[1][0]
post_pruned_decisiontree_precision = round(TP / (TP + FP), 2)
post_pruned_decisiontree_recall = round(TP / (TP + FN), 2)
post_pruned_decisiontree_f1_score = round(2 * (post_pruned_decisiontree_precision * post_pruned_decisiontree_recall) / (post_pruned_decisiontree_precision + post_pruned_decisiontree_recall), 2)
post_pruned_decisiontree_specificity = round(TN / (TN + FP), 2)
print("Post Pruned Decision Tree - Precision" , post_pruned_decisiontree_precision)
print("Post Pruned Decision Tree - Recall", post_pruned_decisiontree_recall)
print("Post Pruned Decision Tree - F1 score", post_pruned_decisiontree_f1_score)
print("Post Pruned Decision Tree - Specificity", post_pruned_decisiontree_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
post_pruned_decisiontree_stratified_k_fold = cross_val_score(post_pruned_tree, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(post_pruned_decisiontree_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in post_pruned_decisiontree_stratified_k_fold]
print("Post Pruned Decision Tree - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Post Pruned Decision Tree - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("-------------------------  LOGISTIC REGRESSION -------------------------")
logistic_regression_model = LogisticRegression(random_state=5805)
logistic_regression_model.fit(X_train, y_train)
logistic_regression_y_pred = logistic_regression_model.predict(X_test)
logistic_regression_y_prob = logistic_regression_model.predict_proba(X_test)[::, -1]
logistic_regression_accuracy = round(accuracy_score(y_test, logistic_regression_y_pred),2)
logistic_regression_confusion_matrix = confusion_matrix(y_test, logistic_regression_y_pred)
logistic_regression_recall = round(recall_score(y_test, logistic_regression_y_pred),2)
logistic_regression_roc_auc = round(roc_auc_score(y_test, logistic_regression_y_prob),2)
print("Logistic Regression - Accuracy :", logistic_regression_accuracy)
print("Logistic Regression - Recall :", logistic_regression_recall)
print("Logistic Regression - ROC & AUC :", logistic_regression_roc_auc)
print("Logistic Regression - Confusion Matrix:")
print(logistic_regression_confusion_matrix)
sns.heatmap(logistic_regression_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
logistic_regression_fpr, logistic_regression_tpr, logistic_regression_thresholds = roc_curve(y_test, logistic_regression_y_prob)
logistic_regression_auc = roc_auc_score(y_test, logistic_regression_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(logistic_regression_fpr, logistic_regression_tpr, color='blue', label=f"Logistic Regression (AUC = {logistic_regression_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic Regression')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = logistic_regression_confusion_matrix[1][1]
TN = logistic_regression_confusion_matrix[0][0]
FP = logistic_regression_confusion_matrix[0][1]
FN = logistic_regression_confusion_matrix[1][0]

logistic_regression_precision = round(TP / (TP + FP),2)
logistic_regression_recall = round(TP / (TP + FN),2)
logistic_regression_f1_score = round(2 * (logistic_regression_precision * logistic_regression_recall) / (logistic_regression_precision + logistic_regression_recall),2)
logistic_regression_specificity = round(TN / (TN + FP),2)
print("Logistic Regression - Precision :", logistic_regression_precision)
print("Logistic Regression - Recall :", logistic_regression_recall)
print("Logistic Regression - F1 score :", logistic_regression_f1_score)
print("Logistic Regression - Specificity :", logistic_regression_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
logistic_regression_stratified_k_fold = cross_val_score(logistic_regression_model, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(logistic_regression_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in logistic_regression_stratified_k_fold]
print("Logistic Regression - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Logistic Regression - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("-------------------------  GRID SEARCH LOGISTIC REGRESSION -------------------------")
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'newton-cg', 'liblinear', 'saga']
}
logistic_regression_GridSearch = GridSearchCV(logistic_regression_model, param_grid, cv=5, scoring='roc_auc')
logistic_regression_GridSearch.fit(X_train, y_train)
logistic_regression_best_parameters = logistic_regression_GridSearch.best_params_
best_logistic_regression_model = logistic_regression_GridSearch.best_estimator_
grid_search_logistic_regression_y_pred_train = best_logistic_regression_model.predict(X_train)
grid_search_logistic_regression_y_pred_test = best_logistic_regression_model.predict(X_test)
grid_search_logistic_regression_y_prob = best_logistic_regression_model.predict_proba(X_test)[:, 1]
grid_search_logistic_regression_accuracy_train = round(accuracy_score(y_train, grid_search_logistic_regression_y_pred_train), 2)
grid_search_logistic_regression_accuracy_test = round(accuracy_score(y_test, grid_search_logistic_regression_y_pred_test), 2)
grid_search_logistic_regression_roc_auc = round(roc_auc_score(y_test, logistic_regression_y_prob), 2)
grid_search_logistic_regression_confusion_matrix = confusion_matrix(y_test, grid_search_logistic_regression_y_pred_test)
print("Grid Search Logistic Regression Best Parameters:", logistic_regression_best_parameters)
print("Grid Search Logistic Regression - Accuracy on Training Set:", grid_search_logistic_regression_accuracy_train)
print("Grid Search Logistic Regression - Accuracy on Testing Set:", grid_search_logistic_regression_accuracy_test)
print("Grid Search Logistic Regression - ROC & AUC:", grid_search_logistic_regression_roc_auc)
print("Grid Search Logistic Regression - Confusion Matrix:")
print(grid_search_logistic_regression_confusion_matrix)
sns.heatmap(grid_search_logistic_regression_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
grid_search_logistic_regression_fpr, grid_search_logistic_regression_tpr, grid_search_logistic_regression_thresholds = roc_curve(y_test, grid_search_logistic_regression_y_prob)
grid_search_logistic_regression_auc = roc_auc_score(y_test, grid_search_logistic_regression_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(grid_search_logistic_regression_fpr, grid_search_logistic_regression_tpr, color='blue', label=f"Logistic Regression (AUC = {grid_search_logistic_regression_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic Regression using Grid Search')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = grid_search_logistic_regression_confusion_matrix[1][1]
TN = grid_search_logistic_regression_confusion_matrix[0][0]
FP = grid_search_logistic_regression_confusion_matrix[0][1]
FN = grid_search_logistic_regression_confusion_matrix[1][0]
grid_search_logistic_regression_precision = round(TP / (TP + FP),2)
grid_search_logistic_regression_recall = round(TP / (TP + FN),2)
grid_search_logistic_regression_f1_score = round(2 * (grid_search_logistic_regression_precision * grid_search_logistic_regression_recall) / (grid_search_logistic_regression_precision + grid_search_logistic_regression_recall),2)
grid_search_logistic_regression_specificity = round(TN / (TN + FP),2)
print("Grid Search Logistic Regression - Precision :", grid_search_logistic_regression_precision)
print("Grid Search Logistic Regression - Recall :", grid_search_logistic_regression_recall)
print("Grid Search Logistic Regression - F1 score :", grid_search_logistic_regression_f1_score)
print("Grid Search Logistic Regression - Specificity :", grid_search_logistic_regression_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
grid_search_logistic_regression_stratified_k_fold = cross_val_score(best_logistic_regression_model, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(grid_search_logistic_regression_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in grid_search_logistic_regression_stratified_k_fold]
print("Grid Search Logistic Regression - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Grid Search Logistic Regression - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("-------------------------  K NEAREST NEIGHBOR -------------------------")
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
knn_y_pred_train = knn_classifier.predict(X_train)
knn_y_pred_test = knn_classifier.predict(X_test)
knn_y_prob = knn_classifier.predict_proba(X_test)[:, -1]
knn_accuracy = round(accuracy_score(y_test, knn_y_pred_test), 2)
knn_recall = round(recall_score(y_test, knn_y_pred_test), 2)
knn_roc_auc = round(roc_auc_score(y_test, knn_y_prob), 2)
knn_confusion_matrix = confusion_matrix(y_test, knn_y_pred_test)
print("K Nearest Neighbor - Accuracy", knn_accuracy)
print("K Nearest Neighbor - Recall", knn_recall)
print("K Nearest Neighbor - ROC & AUC", knn_roc_auc)
print("K Nearest Neighbor - Confusion Matrix")
print(knn_confusion_matrix)
sns.heatmap(knn_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_y_prob)
knn_auc = roc_auc_score(y_test, knn_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(knn_fpr, knn_tpr, color='blue', label=f"K Nearest Neighbor (AUC = {knn_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for K Nearest Neighbor before Elbow method')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = knn_confusion_matrix[1][1]
TN = knn_confusion_matrix[0][0]
FP = knn_confusion_matrix[0][1]
FN = knn_confusion_matrix[1][0]
knn_precision = round(TP / (TP + FP),2)
knn_recall = round(TP / (TP + FN),2)
knn_f1_score = round(2 * (knn_precision * knn_recall) / (knn_precision + knn_recall),2)
knn_specificity = round(TN / (TN + FP),2)
print("K Nearest Neighbor - Precision :", knn_precision)
print("K Nearest Neighbor - Recall :", knn_recall)
print("K Nearest Neighbor - F1 score :", knn_f1_score)
print("K Nearest Neighbor - Specificity :", knn_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
knn_stratified_k_fold = cross_val_score(knn_classifier, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(knn_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in knn_stratified_k_fold]
print("K Nearest Neighbor - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("K Nearest Neighbor - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- GRID SEARCH K NEAREST NEIGHBOR -------------------------")
param_grid = {
    'n_neighbors': [4,6,13,16],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean']
}
grid_search_knn = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
knn_best_parameters = grid_search_knn.best_params_
best_knn_classifier = grid_search_knn.best_estimator_
grid_search_knn_y_pred_train = best_knn_classifier.predict(X_train)
grid_search_knn_y_pred_test = best_knn_classifier.predict(X_test)
grid_search_knn_y_prob = best_knn_classifier.predict_proba(X_test)[:, 1]
grid_search_knn_accuracy_train = round(accuracy_score(y_train, grid_search_knn_y_pred_train), 2)
grid_search_knn_accuracy_test = round(accuracy_score(y_test, grid_search_knn_y_pred_test), 2)
grid_search_knn_roc_auc = round(roc_auc_score(y_test, grid_search_knn_y_prob), 2)
grid_search_knn_confusion_matrix = confusion_matrix(y_test, grid_search_knn_y_pred_test)
print("Grid Search KNN Best Parameters:", knn_best_parameters)
print("Grid Search KNN - Accuracy on Training Set:", grid_search_knn_accuracy_train)
print("Grid Search KNN - Accuracy on Testing Set:", grid_search_knn_accuracy_test)
print("Grid Search KNN - ROC & AUC:", grid_search_knn_roc_auc)
print("Grid Search KNN - Confusion Matrix:")
print(grid_search_knn_confusion_matrix)
sns.heatmap(grid_search_knn_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
grid_search_knn_fpr, grid_search_knn_tpr, grid_search_knn_thresholds = roc_curve(y_test, grid_search_knn_y_prob)
grid_search_knn_auc = roc_auc_score(y_test, grid_search_knn_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(grid_search_knn_fpr, grid_search_knn_tpr, color='blue', label=f"K Nearest Neighbor (AUC = {grid_search_knn_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for K Nearest Neighbor using Grid Search')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
grid_search_knn_confusion_matrix = [[6702, 2491], [3648, 5545]]
TP = grid_search_knn_confusion_matrix[1][1]
TN = grid_search_knn_confusion_matrix[0][0]
FP = grid_search_knn_confusion_matrix[0][1]
FN = grid_search_knn_confusion_matrix[1][0]
grid_search_knn_precision = round(TP / (TP + FP),2)
grid_search_knn_recall = round(TP / (TP + FN),2)
grid_search_knn_f1_score = round(2 * (grid_search_knn_precision * grid_search_knn_recall) / (grid_search_knn_precision + grid_search_knn_recall),2)
grid_search_knn_specificity = round(TN / (TN + FP),2)
print("Grid Search K Nearest Neighbor - Precision :", grid_search_knn_precision)
print("Grid Search K Nearest Neighbor - Recall :", grid_search_knn_recall)
print("Grid Search K Nearest Neighbor - F1 score :", grid_search_knn_f1_score)
print("Grid Search K Nearest Neighbor - Specificity :", grid_search_knn_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
grid_search_knn_stratified_k_fold = cross_val_score(best_knn_classifier, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(grid_search_knn_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in grid_search_knn_stratified_k_fold]
print("Grid Search K Nearest Neighbor - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Grid Search K Nearest Neighbor - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- ELBOW METHOD K NEAREST NEIGHBOR -------------------------")
no_of_k = [i for i in range(1, 15)]
error_rate = []
for k in no_of_k:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_y_pred_test = knn_classifier.predict(X_test)
    error_rate.append(np.mean(y_test != knn_y_pred_test))
plt.figure(figsize=(10, 6))
plt.plot(range(1, 15, 1), error_rate, color='orange', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
plt.title('Error Rate vs. K')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()
e_knn_classifier = KNeighborsClassifier(n_neighbors=5)
e_knn_classifier.fit(X_train, y_train)
e_knn_y_pred_train = e_knn_classifier.predict(X_train)
e_knn_y_pred_test = e_knn_classifier.predict(X_test)
e_knn_y_prob_knn = e_knn_classifier.predict_proba(X_test)[:, -1]
e_knn_accuracy = round(accuracy_score(y_test, e_knn_y_pred_test), 2)
e_knn_recall = round(recall_score(y_test, e_knn_y_pred_test), 2)
e_knn_roc_auc = round(roc_auc_score(y_test, e_knn_y_prob_knn), 2)
e_knn_confusion_matrix = confusion_matrix(y_test, e_knn_y_pred_test)
print("Elbow Method K Nearest Neighbor - Accuracy:", e_knn_accuracy)
print("Elbow Method K Nearest Neighbor - Recall:", e_knn_recall)
print("Elbow Method K Nearest Neighbor - ROC & AUC:", e_knn_roc_auc)
print("Elbow Method K Nearest Neighbor - Confusion Matrix:")
print(e_knn_confusion_matrix)
sns.heatmap(e_knn_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_y_prob)
knn_auc = roc_auc_score(y_test, knn_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(knn_fpr, knn_tpr, color='blue', label=f"K Nearest Neighbor (AUC = {knn_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for K Nearest Neighbor after Elbow method')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = e_knn_confusion_matrix[1][1]
TN = e_knn_confusion_matrix[0][0]
FP = e_knn_confusion_matrix[0][1]
FN = e_knn_confusion_matrix[1][0]
e_knn_precision = round(TP / (TP + FP), 2)
e_knn_recall = round(TP / (TP + FN), 2)
e_knn_f1_score = round(2 * (e_knn_precision * e_knn_recall) / (e_knn_precision + e_knn_recall), 2)
e_knn_specificity = round(TN / (TN + FP), 2)
print("K Nearest Neighbor using Elbow Method - Precision:", e_knn_precision)
print("K Nearest Neighbor using Elbow Method - Recall:", e_knn_recall)
print("K Nearest Neighbor using Elbow Method - F1 score:", e_knn_f1_score)
print("K Nearest Neighbor using Elbow Method - Specificity:", e_knn_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
e_knn_stratified_k_fold = cross_val_score(best_knn_classifier, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(e_knn_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in e_knn_stratified_k_fold]
print("K Nearest Neighbor using Elbow Method - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("K Nearest Neighbor using Elbow Method - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- LINEAR SUPPORT VECTOR MACHINE -------------------------")
linear_svm = SVC(C=0.1, kernel='linear',probability=True)
linear_svm.fit(X_train, y_train)
linear_svm_y_pred = linear_svm.predict(X_test)
linear_svm_y_prob = linear_svm.predict_proba(X_test)[::, -1]
linear_svm_accuracy = round(accuracy_score(y_test, linear_svm_y_pred), 2)
linear_svm_recall = round(recall_score(y_test, linear_svm_y_pred), 2)
linear_svm_roc_auc = round(roc_auc_score(y_test, linear_svm_y_prob), 2)
linear_svm_confusion_matrix = confusion_matrix(y_test, linear_svm_y_pred)
print("Linear SVM - Accuracy: ", linear_svm_accuracy)
print("Linear SVM - Recall: ", linear_svm_recall)
print("Linear SVM - ROC & AUC:", linear_svm_roc_auc)
print("Linear SVM - Confusion Matrix:")
print(linear_svm_confusion_matrix)
sns.heatmap(linear_svm_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
linear_svm_fpr, linear_svm_tpr, linear_svm_thresholds = roc_curve(y_test, linear_svm_y_prob)
linear_svm_auc = roc_auc_score(y_test, linear_svm_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(linear_svm_fpr, linear_svm_tpr, color='blue', label=f"Linear SVM (AUC = {linear_svm_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Linear SVM')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = linear_svm_confusion_matrix[1][1]
TN = linear_svm_confusion_matrix[0][0]
FP = linear_svm_confusion_matrix[0][1]
FN = linear_svm_confusion_matrix[1][0]
linear_svm_precision = round(TP / (TP + FP), 2)
linear_svm_recall = round(TP / (TP + FN), 2)
linear_svm_f1_score = round(2 * (linear_svm_precision * linear_svm_recall) / (linear_svm_precision + linear_svm_recall), 2)
linear_svm_specificity = round(TN / (TN + FP), 2)
print("Linear SVM - Precision:", linear_svm_precision)
print("Linear SVM - Recall:", linear_svm_recall)
print("Linear SVM - F1 score:", linear_svm_f1_score)
print("Linear SVM - Specificity:", linear_svm_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
linear_svm_stratified_k_fold = cross_val_score(linear_svm, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(linear_svm_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in linear_svm_stratified_k_fold]
print("Linear SVM - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Linear SVM - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- GRID SEARCH SUPPORT VECTOR MACHINE -------------------------")
grid_search_svm = SVC(random_state=5805, probability=True)
grid_search_svm_parameters = {'C': [0.4, 0.5],
                              'kernel': ['poly', 'rbf']}
grid_search_svm = GridSearchCV(estimator=grid_search_svm, param_grid=grid_search_svm_parameters, scoring='f1_macro', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
grid_search_svm_best_parameters = grid_search_svm.best_params_
best_svm_estimator = grid_search_svm.best_estimator_
grid_search_svm_y_pred = best_svm_estimator.predict(X_test)
grid_search_svm_y_prob = best_svm_estimator.predict_proba(X_test)[:, 1]
grid_search_svm_accuracy = round(accuracy_score(y_test, grid_search_svm_y_pred), 2)
grid_search_svm_recall = round(recall_score(y_test, grid_search_svm_y_pred), 2)
grid_search_svm_roc_auc = round(roc_auc_score(y_test, grid_search_svm_y_prob), 2)
grid_search_svm_confusion_matrix = confusion_matrix(y_test, grid_search_svm_y_pred)
print("Grid Search Support Vector Machine Best Parameters:", grid_search_svm_best_parameters)
print("Grid Search Support Vector Machine - Accuracy:", grid_search_svm_accuracy)
print("Grid Search Support Vector Machine - Recall:", grid_search_svm_recall)
print("Grid Search Support Vector Machine - AUC-ROC:", grid_search_svm_roc_auc)
print("Grid Search Support Vector Machine - Confusion Matrix:")
print(grid_search_svm_confusion_matrix)
sns.heatmap(grid_search_svm_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
grid_search_svm_fpr, grid_search_svm_tpr, grid_search_svm_thresholds = roc_curve(y_test, linear_svm_y_prob)
grid_search_svm_auc = roc_auc_score(y_test, grid_search_svm_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(grid_search_svm_fpr, grid_search_svm_tpr, color='blue', label=f"Grid Search SVM (AUC = {grid_search_svm_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Grid Search SVM')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = grid_search_svm_confusion_matrix[1][1]
TN = grid_search_svm_confusion_matrix[0][0]
FP = grid_search_svm_confusion_matrix[0][1]
FN = grid_search_svm_confusion_matrix[1][0]
grid_search_svm_precision = round(TP / (TP + FP), 2)
grid_search_svm_recall = round(TP / (TP + FN), 2)
grid_search_svm_f1_score = round(2 * (grid_search_svm_precision * grid_search_svm_recall) / (grid_search_svm_precision + grid_search_svm_recall), 2)
grid_search_svm_specificity = round(TN / (TN + FP), 2)
print("Grid Search SVM - Precision:", grid_search_svm_precision)
print("Grid Search SVM - Recall:", grid_search_svm_recall)
print("Grid Search SVM - F1 score:", grid_search_svm_f1_score)
print("Grid Search SVM - Specificity:", grid_search_svm_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
grid_search_svm_stratified_k_fold = cross_val_score(grid_search_svm, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(grid_search_svm_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in grid_search_svm_stratified_k_fold]
print("Grid Search SVM - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Grid Search SVM - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- POLYNOMIAL SUPPORT VECTOR MACHINE -------------------------")
poly_svm = SVC(kernel='poly',probability=True)
poly_svm.fit(X_train, y_train)
poly_svm_y_pred = poly_svm.predict(X_test)
poly_svm_y_prob = poly_svm.predict_proba(X_test)[::, -1]
poly_svm_accuracy = round(accuracy_score(y_test, poly_svm_y_pred), 2)
poly_svm_recall = round(recall_score(y_test, poly_svm_y_pred), 2)
poly_svm_roc_auc = round(roc_auc_score(y_test, poly_svm_y_prob), 2)
poly_svm_confusion_matrix = confusion_matrix(y_test, poly_svm_y_pred)
print("Polynomial SVM - Accuracy: ", poly_svm_accuracy)
print("Polynomial SVM - Recall: ", poly_svm_recall)
print("Polynomial SVM - ROC & AUC:", poly_svm_roc_auc)
print("Polynomial SVM - Confusion Matrix:")
print(poly_svm_confusion_matrix)
sns.heatmap(poly_svm_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
poly_svm_fpr, poly_svm_tpr, poly_svm_thresholds = roc_curve(y_test, poly_svm_y_prob)
poly_svm_auc = roc_auc_score(y_test, poly_svm_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(poly_svm_fpr, poly_svm_tpr, color='blue', label=f"Polynomial SVM (AUC = {poly_svm_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Polynomial SVM')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = poly_svm_confusion_matrix[1][1]
TN = poly_svm_confusion_matrix[0][0]
FP = poly_svm_confusion_matrix[0][1]
FN = poly_svm_confusion_matrix[1][0]
poly_svm_precision = round(TP / (TP + FP), 2)
poly_svm_recall = round(TP / (TP + FN), 2)
poly_svm_f1_score = round(2 * (poly_svm_precision * poly_svm_recall) / (poly_svm_precision + poly_svm_recall), 2)
poly_svm_specificity = round(TN / (TN + FP), 2)
print("Polynomial SVM - Precision:", poly_svm_precision)
print("Polynomial SVM - Recall:", poly_svm_recall)
print("Polynomial SVM - F1 score:", poly_svm_f1_score)
print("Polynomial SVM - Specificity:", poly_svm_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
poly_svm_stratified_k_fold = cross_val_score(poly_svm, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(poly_svm_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in poly_svm_stratified_k_fold]
print("Polynomial SVM - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Polynomial SVM - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- RADIAL BASE SUPPORT VECTOR MACHINE -------------------------")
rbf_svm = SVC(kernel='rbf',probability=True)
rbf_svm.fit(X_train, y_train)
rbf_svm_y_pred = rbf_svm.predict(X_test)
rbf_svm_y_prob = rbf_svm.predict_proba(X_test)[::, -1]
rbf_svm_accuracy = round(accuracy_score(y_test, rbf_svm_y_pred), 2)
rbf_svm_recall = round(recall_score(y_test, rbf_svm_y_pred), 2)
rbf_svm_roc_auc = round(roc_auc_score(y_test, rbf_svm_y_prob), 2)
rbf_svm_confusion_matrix = confusion_matrix(y_test, rbf_svm_y_pred)
print("Radial Base SVM - Accuracy: ", rbf_svm_accuracy)
print("Radial Base SVM - Recall: ", rbf_svm_recall)
print("Radial Base SVM - ROC & AUC:", rbf_svm_roc_auc)
print("Radial Base SVM - Confusion Matrix:")
print(rbf_svm_confusion_matrix)
sns.heatmap(rbf_svm_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
rbf_svm_fpr, rbf_svm_tpr, rbf_svm_thresholds = roc_curve(y_test, rbf_svm_y_prob)
rbf_svm_auc = roc_auc_score(y_test, rbf_svm_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(rbf_svm_fpr, rbf_svm_tpr, color='blue', label=f"Radial Base SVM (AUC = {rbf_svm_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Radial Base SVM')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = rbf_svm_confusion_matrix[1][1]
TN = rbf_svm_confusion_matrix[0][0]
FP = rbf_svm_confusion_matrix[0][1]
FN = rbf_svm_confusion_matrix[1][0]
rbf_svm_precision = round(TP / (TP + FP), 2)
rbf_svm_recall = round(TP / (TP + FN), 2)
rbf_svm_f1_score = round(2 * (rbf_svm_precision * rbf_svm_recall) / (rbf_svm_precision + rbf_svm_recall), 2)
rbf_svm_specificity = round(TN / (TN + FP), 2)
print("Radial Base SVM - Precision:", rbf_svm_precision)
print("Radial Base SVM - Recall:", rbf_svm_recall)
print("Radial Base SVM - F1 score:", rbf_svm_f1_score)
print("Radial Base SVM - Specificity:", rbf_svm_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
rbf_svm_stratified_k_fold = cross_val_score(rbf_svm, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(rbf_svm_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in rbf_svm_stratified_k_fold]
print("Radial Basis SVM - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Radial Basis SVM - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- NAIVE BAYES -------------------------")
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
naive_bayes_y_pred = naive_bayes.predict(X_test)
naive_bayes_y_prob = naive_bayes.predict_proba(X_test)[::, -1]
naive_bayes_accuracy = round(accuracy_score(y_test, naive_bayes_y_pred), 2)
naive_bayes_recall = round(recall_score(y_test, naive_bayes_y_pred, pos_label=0), 2)
naive_bayes_roc_auc = round(roc_auc_score(y_test, naive_bayes_y_prob), 2)
naive_bayes_confusion_matrix = confusion_matrix(y_test, naive_bayes_y_pred)
print("Naive Bayes - Accuracy:", naive_bayes_accuracy)
print("Naive Bayes - Recall:", naive_bayes_recall)
print("Naive Bayes - ROC & AUC:", naive_bayes_roc_auc)
print("Naive Bayes - Confusion Matrix:")
print(naive_bayes_confusion_matrix)
sns.heatmap(naive_bayes_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
naive_bayes_fpr, naive_bayes_tpr, naive_bayes_thresholds = roc_curve(y_test, naive_bayes_y_prob)
naive_bayes_auc = roc_auc_score(y_test, naive_bayes_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(naive_bayes_fpr, naive_bayes_tpr, color='blue', label=f"Naive Bayes (AUC = {naive_bayes_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Naive Bayes')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = naive_bayes_confusion_matrix[0][0]
TN = naive_bayes_confusion_matrix[1][1]
FP = naive_bayes_confusion_matrix[0][1]
FN = naive_bayes_confusion_matrix[1][0]
naive_bayes_precision = round(TP / (TP + FP), 2)
naive_bayes_recall = round(TP / (TP + FN), 2)
naive_bayes_f1_score = round(2 * (naive_bayes_precision * naive_bayes_recall) / (naive_bayes_precision + naive_bayes_recall), 2)
naive_bayes_specificity = round(TN / (TN + FP), 2)
print("Naive Bayes - Precision:", naive_bayes_precision)
print("Naive Bayes - Recall:", naive_bayes_recall)
print("Naive Bayes - F1 score:", naive_bayes_f1_score)
print("Naive Bayes - Specificity:", naive_bayes_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
naive_bayes_stratified_k_fold = cross_val_score(naive_bayes, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(naive_bayes_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in naive_bayes_stratified_k_fold]
print("Naive Bayes - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Naive Bayes - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- GRID SEARCH NAIVE BAYES -------------------------")
param_grid = {
    'priors': [None, [0.2, 0.8], [0.5, 0.5], [0.8, 0.2]]
}
naive_bayes_GridSearch = GaussianNB()
scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'AUC-ROC': make_scorer(roc_auc_score)
}
grid_search_naive_bayes = GridSearchCV(naive_bayes_GridSearch, param_grid, cv=5, scoring=scoring, refit='AUC-ROC', verbose=2)
grid_search_naive_bayes.fit(X_train, y_train)
best_naive_bayes_estimator = grid_search_naive_bayes.best_estimator_
grid_search_naive_bayes_y_pred = best_naive_bayes_estimator.predict(X_test)
grid_search_naive_bayes_y_prob = best_naive_bayes_estimator.predict_proba(X_test)[:, 1]
grid_search_naive_bayes_best_parameters = grid_search_naive_bayes.best_params_
grid_search_naive_bayes_accuracy = round(accuracy_score(y_test, grid_search_naive_bayes_y_pred), 3)
grid_search_naive_bayes_recall = round(recall_score(y_test, grid_search_naive_bayes_y_pred, pos_label=0), 3)
grid_search_naive_bayes_roc_auc = round(roc_auc_score(y_test, grid_search_naive_bayes_y_prob), 3)
grid_search_naive_bayes_confusion_matrix = confusion_matrix(y_test, grid_search_naive_bayes_y_pred)
print("Grid Search Naive Bayes Best Parameters:", grid_search_naive_bayes_best_parameters)
print("Grid Search Naive Bayes - Accuracy:", grid_search_naive_bayes_accuracy)
print("Grid Search Naive Bayes - Recall:", grid_search_naive_bayes_recall)
print("Grid Search Naive Bayes - AUC-ROC:", grid_search_naive_bayes_roc_auc)
print("Grid Search Naive Bayes - Confusion Matrix:")
print(grid_search_naive_bayes_confusion_matrix)
sns.heatmap(grid_search_naive_bayes_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
grid_search_naive_bayes_fpr, grid_search_naive_bayes_tpr, grid_search_naive_bayes_thresholds = roc_curve(y_test, grid_search_naive_bayes_y_prob)
grid_search_naive_bayes_auc = roc_auc_score(y_test, grid_search_naive_bayes_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(grid_search_naive_bayes_fpr, grid_search_naive_bayes_tpr, color='blue', label=f"Naive Bayes using Grid Search (AUC = {grid_search_naive_bayes_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Naive Bayes using Grid Search')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = grid_search_naive_bayes_confusion_matrix[0][0]
TN = grid_search_naive_bayes_confusion_matrix[1][1]
FP = grid_search_naive_bayes_confusion_matrix[0][1]
FN = grid_search_naive_bayes_confusion_matrix[1][0]
grid_search_naive_bayes_precision = round(TP / (TP + FP), 2)
grid_search_naive_bayes_recall = round(TP / (TP + FN), 2)
grid_search_naive_bayes_f1_score = round(2 * (grid_search_naive_bayes_precision * grid_search_naive_bayes_recall) / (grid_search_naive_bayes_precision + grid_search_naive_bayes_recall), 2)
grid_search_naive_bayes_specificity = round(TN / (TN + FP), 2)
print("Naive Bayes - Precision:", grid_search_naive_bayes_precision)
print("Naive Bayes - Recall:", grid_search_naive_bayes_recall)
print("Naive Bayes - F1 score:", grid_search_naive_bayes_f1_score)
print("Naive Bayes - Specificity:", grid_search_naive_bayes_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
grid_search_naive_bayes_stratified_k_fold = cross_val_score(best_naive_bayes_estimator, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(grid_search_naive_bayes_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in grid_search_naive_bayes_stratified_k_fold]
print("Grid Search Naive Bayes - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Grid Search Naive Bayes - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- RANDOM FOREST -------------------------")
random_forest = RandomForestClassifier(n_estimators=100, random_state=5805)
random_forest.fit(X_train, y_train)
random_forest_y_pred = random_forest.predict(X_test)
random_forest_y_prob = random_forest.predict_proba(X_test)[::, -1]
random_forest_accuracy = round(accuracy_score(y_test, random_forest_y_pred), 2)
random_forest_recall = round(recall_score(y_test, random_forest_y_pred), 2)
random_forest_roc_auc = round(roc_auc_score(y_test, random_forest_y_prob), 2)
random_forest_confusion_matrix = confusion_matrix(y_test, random_forest_y_pred)
print("Random Forest - Accuracy:", random_forest_accuracy)
print("Random Forest - Recall:", random_forest_recall)
print("Random Forest - ROC & AUC:", random_forest_roc_auc)
print("Random Forest - Confusion Matrix:")
print(random_forest_confusion_matrix)
sns.heatmap(random_forest_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
random_forest_fpr, random_forest_tpr, random_forest_thresholds = roc_curve(y_test, random_forest_y_prob)
random_forest_auc = roc_auc_score(y_test, random_forest_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(random_forest_fpr, random_forest_tpr, color='blue', label=f"Random Forest (AUC = {random_forest_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = random_forest_confusion_matrix[1][1]
TN = random_forest_confusion_matrix[0][0]
FP = random_forest_confusion_matrix[0][1]
FN = random_forest_confusion_matrix[1][0]
random_forest_precision = round(TP / (TP + FP), 2)
random_forest_recall = round(TP / (TP + FN), 2)
random_forest_f1_score = round(2 * (random_forest_precision * random_forest_recall) / (random_forest_precision + random_forest_recall), 2)
random_forest_specificity = round(TN / (TN + FP), 2)
print("Random Forest - Precision:", random_forest_precision)
print("Random Forest - Recall:", random_forest_recall)
print("Random Forest - F1 score:", random_forest_f1_score)
print("Random Forest - Specificity:", random_forest_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
random_forest_stratified_k_fold = cross_val_score(random_forest, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(random_forest_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in random_forest_stratified_k_fold]
print("Random Forest - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Random Forest - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- GRID SEARCH RANDOM FOREST -------------------------")
parameter_grid = {'n_estimators': [100, 125],
                  'criterion': ['gini', 'entropy']}
grid_search_random_forest = GridSearchCV(estimator=random_forest, param_grid=parameter_grid, scoring='f1', n_jobs=-1)
grid_search_random_forest.fit(X_train, y_train)
grid_search_random_forest_best_parameters = grid_search_random_forest.best_params_
best_random_forest_classifier = grid_search_random_forest.best_estimator_
grid_search_random_forest_y_pred = best_random_forest_classifier.predict(X_test)
grid_search_random_forest_y_prob = best_random_forest_classifier.predict_proba(X_test)[:, 1]
grid_search_random_forest_accuracy = round(accuracy_score(y_test, grid_search_random_forest_y_pred), 2)
grid_search_random_forest_recall = round(recall_score(y_test, grid_search_random_forest_y_pred), 2)
grid_search_random_forest_roc_auc = round(roc_auc_score(y_test, grid_search_random_forest_y_prob), 2)
grid_search_random_forest_confusion_matrix = confusion_matrix(y_test, grid_search_random_forest_y_pred)
print("Grid Search Random Forest Best Parameters:", grid_search_random_forest_best_parameters)
print("Grid Search Random Forest - Accuracy:", grid_search_random_forest_accuracy)
print("Grid Search Random Forest - Recall:", grid_search_random_forest_recall)
print("Grid Search Random Forest - ROC & AUC:", grid_search_random_forest_roc_auc)
print("Grid Search Random Forest - Confusion Matrix:")
print(grid_search_random_forest_confusion_matrix)
sns.heatmap(grid_search_random_forest_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
grid_search_random_forest_fpr, grid_search_random_forest_tpr, grid_search_random_forest_thresholds = roc_curve(y_test, grid_search_random_forest_y_prob)
grid_search_random_forest_auc = roc_auc_score(y_test, grid_search_random_forest_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(grid_search_random_forest_fpr, grid_search_random_forest_tpr, color='blue', label=f"Grid Search Random Forest (AUC = {grid_search_random_forest_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest using grid Search')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = grid_search_random_forest_confusion_matrix[1][1]
TN = grid_search_random_forest_confusion_matrix[0][0]
FP = grid_search_random_forest_confusion_matrix[0][1]
FN = grid_search_random_forest_confusion_matrix[1][0]
grid_search_random_forest_precision = round(TP / (TP + FP), 2)
grid_search_random_forest_recall = round(TP / (TP + FN), 2)
grid_search_random_forest_f1_score = round(2 * (grid_search_random_forest_precision * grid_search_random_forest_recall) / (grid_search_random_forest_precision + grid_search_random_forest_recall), 2)
grid_search_random_forest_specificity = round(TN / (TN + FP), 2)
print("Random Forest - Precision:", grid_search_random_forest_precision)
print("Random Forest - Recall:", grid_search_random_forest_recall)
print("Random Forest - F1 score:", grid_search_random_forest_f1_score)
print("Random Forest - Specificity:", grid_search_random_forest_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
grid_search_random_forest_stratified_k_fold = cross_val_score(random_forest, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(grid_search_random_forest_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in grid_search_random_forest_stratified_k_fold]
print("Grid Search Random Forest - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Grid Search Random Forest - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- RANDOM FOREST BAGGING -------------------------")
random_forest_bagging = BaggingClassifier(random_forest,random_state=5805)
random_forest_bagging.fit(X_train, y_train)
random_forest_bagging_y_pred = random_forest_bagging.predict(X_test)
random_forest_bagging_y_prob = random_forest_bagging.predict_proba(X_test)[::, -1]
random_forest_bagging_accuracy = round(accuracy_score(y_test, random_forest_bagging_y_pred), 2)
random_forest_bagging_recall = round(recall_score(y_test, random_forest_bagging_y_pred), 2)
random_forest_bagging_roc_auc = round(roc_auc_score(y_test, random_forest_bagging_y_prob), 2)
random_forest_bagging_confusion_matrix = confusion_matrix(y_test, random_forest_bagging_y_pred)
print("Random Forest Bagging - Accuracy:", random_forest_bagging_accuracy)
print("Random Forest Bagging - Recall:", random_forest_bagging_recall)
print("Random Forest Bagging - ROC & AUC:", random_forest_bagging_roc_auc)
print("Random Forest Bagging - Confusion Matrix:")
print(random_forest_bagging_confusion_matrix)
sns.heatmap(random_forest_bagging_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
random_forest_bagging_fpr, random_forest_bagging_tpr, _ = roc_curve(y_test, random_forest_bagging_y_prob)
random_forest_bagging_auc = roc_auc_score(y_test, random_forest_bagging_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(random_forest_bagging_fpr, random_forest_bagging_tpr, color='blue', label=f"Random Forest - Bagging (AUC = {random_forest_bagging_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest - Bagging')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = random_forest_bagging_confusion_matrix[1][1]
TN = random_forest_bagging_confusion_matrix[0][0]
FP = random_forest_bagging_confusion_matrix[0][1]
FN = random_forest_bagging_confusion_matrix[1][0]
random_forest_bagging_precision = round(TP / (TP + FP), 2)
random_forest_bagging_recall = round(TP / (TP + FN), 2)
random_forest_bagging_f1_score = round(2 * (random_forest_bagging_precision * random_forest_bagging_recall) / (random_forest_bagging_precision + random_forest_bagging_recall), 2)
random_forest_bagging_specificity = round(TN / (TN + FP), 2)
print("Random Forest Bagging - Precision:", random_forest_bagging_precision)
print("Random Forest Bagging - Recall:", random_forest_bagging_recall)
print("Random Forest Bagging F1 - score:", random_forest_bagging_f1_score)
print("Random Forest Bagging - Specificity:", random_forest_bagging_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
random_forest_bagging_stratified_k_fold = cross_val_score(random_forest_bagging, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
mean_auc_score = round(random_forest_bagging_stratified_k_fold.mean(), 2)
rounded_scores = [round(score, 2) for score in random_forest_bagging_stratified_k_fold]
print("Random Forest Bagging - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Random Forest Bagging - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- RANDOM FOREST BOOSTING -------------------------")
random_forest_boosting = AdaBoostClassifier(random_forest,n_estimators=75, random_state=5805)
random_forest_boosting.fit(X_train, y_train)
random_forest_boosting_y_pred = random_forest_boosting.predict(X_test)
random_forest_boosting_y_prob = random_forest_boosting.predict_proba(X_test)[:, -1]
random_forest_boosting_accuracy = round(accuracy_score(y_test, random_forest_boosting_y_pred), 2)
random_forest_boosting_recall = round(recall_score(y_test, random_forest_boosting_y_pred), 2)
random_forest_boosting_roc_auc = round(roc_auc_score(y_test, random_forest_boosting_y_prob), 2)
random_forest_boosting_confusion_matrix = confusion_matrix(y_test, random_forest_boosting_y_pred)
print("Random Forest Boosting - Accuracy:", random_forest_boosting_accuracy)
print("Random Forest Boosting - Recall:", random_forest_boosting_recall)
print("Random Forest Boosting - ROC & AUC:", random_forest_boosting_roc_auc)
print("Random Forest Boosting - Confusion Matrix:")
print(random_forest_boosting_confusion_matrix)
sns.heatmap(random_forest_boosting_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
random_forest_boosting_fpr, random_forest_boosting_tpr, _ = roc_curve(y_test, random_forest_boosting_y_prob)
random_forest_boosting_auc = roc_auc_score(y_test, random_forest_boosting_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(random_forest_boosting_fpr, random_forest_boosting_tpr,color='blue', label=f"Random Forest - Boosting(AUC = {random_forest_boosting_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest - Boosting')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = random_forest_boosting_confusion_matrix[1][1]
TN = random_forest_boosting_confusion_matrix[0][0]
FP = random_forest_boosting_confusion_matrix[0][1]
FN = random_forest_boosting_confusion_matrix[1][0]
random_forest_boosting_precision = round(TP / (TP + FP), 2)
random_forest_boosting_recall = round(TP / (TP + FN), 2)
random_forest_boosting_f1_score = round(2 * (random_forest_boosting_precision * random_forest_boosting_recall) / (random_forest_boosting_precision + random_forest_boosting_recall), 2)
random_forest_boosting_specificity = round(TN / (TN + FP), 2)
print("Random Forest Boosting - Precision:", random_forest_boosting_precision)
print("Random Forest Boosting - Recall:", random_forest_boosting_recall)
print("Random Forest Boosting - F1 score:", random_forest_boosting_f1_score)
print("Random Forest Boosting - Specificity:", random_forest_boosting_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
random_forest_boosting_stratified_k_fold = cross_val_score(random_forest_boosting, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
rounded_auc_scores = [round(score, 2) for score in random_forest_boosting_stratified_k_fold]
mean_auc_score = round(random_forest_boosting_stratified_k_fold.mean(), 2)
print("Random Forest Boosting - Stratified K-fold Cross Validation AUC Scores: ", rounded_auc_scores)
print("Random Forest Boosting - Stratified K-fold Cross Validation Mean AUC Score: ", mean_auc_score)

print("------------------------- RANDOM FOREST STACKING -------------------------")
estimators = [('BC', BaggingClassifier()),
              ('BO', AdaBoostClassifier(n_estimators=100)),
              ('GBO', GradientBoostingClassifier(n_estimators=100))]
random_forest_stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
random_forest_stacking.fit(X_train, y_train)
random_forest_stacking_y_pred = random_forest_stacking.predict(X_test)
random_forest_stacking_y_prob = random_forest_stacking.predict_proba(X_test)[::, -1]
random_forest_stacking_accuracy = accuracy_score(y_test, random_forest_stacking_y_pred)
random_forest_stacking_recall = recall_score(y_test, random_forest_stacking_y_pred)
random_forest_stacking_roc_auc = roc_auc_score(y_test, random_forest_stacking_y_prob)
random_forest_stacking_accuracy = round(random_forest_stacking_accuracy, 2)
random_forest_stacking_recall = round(random_forest_stacking_recall, 2)
random_forest_stacking_roc_auc = round(random_forest_stacking_roc_auc, 2)
random_forest_stacking_confusion_matrix = confusion_matrix(y_test, random_forest_stacking_y_pred)
print("Random Forest Stacking - Accuracy:", random_forest_stacking_accuracy)
print("Random Forest Stacking - Recall:", random_forest_stacking_recall)
print("Random Forest Stacking - ROC & AUC:", random_forest_stacking_roc_auc)
print("Random Forest Stacking - Confusion Matrix:")
print(random_forest_stacking_confusion_matrix)
sns.heatmap(random_forest_stacking_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
random_forest_stacking_fpr, random_forest_stacking_tpr, _ = roc_curve(y_test, random_forest_stacking_y_prob)
random_forest_stacking_auc= roc_auc_score(y_test, random_forest_stacking_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(random_forest_stacking_fpr, random_forest_stacking_tpr,color='blue', label=f"Random Forest - Stacking(AUC = {random_forest_stacking_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest - Stacking')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = random_forest_stacking_confusion_matrix[1][1]
TN = random_forest_stacking_confusion_matrix[0][0]
FP = random_forest_stacking_confusion_matrix[0][1]
FN = random_forest_stacking_confusion_matrix[1][0]
random_forest_stacking_precision = round(TP / (TP + FP), 2)
random_forest_stacking_recall = round(TP / (TP + FN), 2)
random_forest_stacking_f1_score = round(2 * (random_forest_stacking_precision * random_forest_stacking_recall) / (random_forest_stacking_precision + random_forest_stacking_recall), 2)
random_forest_stacking_specificity = round(TN / (TN + FP), 2)
print("Random Forest Stacking - Precision:", random_forest_stacking_precision)
print("Random Forest Stacking - Recall:", random_forest_stacking_recall)
print("Random Forest Stacking - F1 score:", random_forest_stacking_f1_score)
print("Random Forest Stacking - Specificity:", random_forest_stacking_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
random_forest_stacking_stratified_k_fold = cross_val_score(random_forest_stacking, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
rounded_scores = [round(score, 2) for score in random_forest_stacking_stratified_k_fold]
print("Random Forest Stacking - Stratified K-fold Cross Validation AUC Scores: ", rounded_scores)
print("Random Forest Stacking - Stratified K-fold Cross Validation Mean AUC Score: ", round(random_forest_stacking_stratified_k_fold.mean(), 2))

print("------------------------- NEURAL NETWORK -------------------------")
neural_network_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,solver='sgd',activation='tanh', random_state=5805)
neural_network_mlp.fit(X_train, y_train)
neural_network_mlp_y_pred = neural_network_mlp.predict(X_test)
neural_network_mlp_y_prob = neural_network_mlp.predict_proba(X_test)[::, -1]
neural_network_mlp_accuracy = round(accuracy_score(y_test, neural_network_mlp_y_pred),2)
neural_network_mlp_recall = round(recall_score(y_test, neural_network_mlp_y_pred), 2)
neural_network_mlp_roc_auc = round(roc_auc_score(y_test, neural_network_mlp_y_prob), 2)
neural_network_mlp_confusion_matrix = confusion_matrix(y_test, neural_network_mlp_y_pred)
print("Neural Network - Accuracy:", neural_network_mlp_accuracy)
print("Neural Network - Recall:", neural_network_mlp_recall)
print("Neural Network - ROC & AUC:", neural_network_mlp_roc_auc)
print("Neural Network - Confusion Matrix:")
print(neural_network_mlp_confusion_matrix)
sns.heatmap(neural_network_mlp_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
neural_network_mlp_fpr, neural_network_mlp_tpr, _ = roc_curve(y_test, neural_network_mlp_y_prob)
neural_network_mlp_auc = roc_auc_score(y_test, random_forest_boosting_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(neural_network_mlp_fpr, neural_network_mlp_tpr, color='blue', label=f"Neural Network - MLP (AUC = {neural_network_mlp_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Neural Network')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
TP = neural_network_mlp_confusion_matrix[1][1]
TN = neural_network_mlp_confusion_matrix[0][0]
FP = neural_network_mlp_confusion_matrix[0][1]
FN = neural_network_mlp_confusion_matrix[1][0]
neural_network_mlp_precision = round(TP / (TP + FP), 2)
neural_network_mlp_recall = round(TP / (TP + FN), 2)
neural_network_mlp_f1_score = round(2 * (neural_network_mlp_precision * neural_network_mlp_recall) / (neural_network_mlp_precision + neural_network_mlp_recall), 2)
neural_network_mlp_specificity = round(TN / (TN + FP), 2)
print("Neural Network - Precision:", neural_network_mlp_precision)
print("Neural Network - Recall:", neural_network_mlp_recall)
print("Neural Network - F1 score:", neural_network_mlp_f1_score)
print("Neural Network - Specificity:", neural_network_mlp_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
neural_network_mlp_stratified_k_fold = cross_val_score(neural_network_mlp, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
neural_network_mlp_stratified_k_fold = np.round(neural_network_mlp_stratified_k_fold, 2)
print("Neural Network - Stratified K-fold Cross Validation AUC Score: ", neural_network_mlp_stratified_k_fold)
print("Neural Network - Stratified K-fold Cross Validation Mean AUC Score: ", neural_network_mlp_stratified_k_fold.mean())

print("------------------------- GRID SEARCH NEURAL NETWORK -------------------------")
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'lbfgs'],
    'random_state': [5805]
}
grid_search_neural_network_mlp = GridSearchCV(neural_network_mlp, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_neural_network_mlp.fit(X_train, y_train)
best_neural_network_mlp = grid_search_neural_network_mlp.best_estimator_
grid_search_neural_network_mlp_best_parameters = grid_search_neural_network_mlp.best_params_
grid_search_neural_network_mlp_y_pred = best_neural_network_mlp.predict(X_test)
grid_search_neural_network_mlp_y_prob = best_neural_network_mlp.predict_proba(X_test)[:, 1]
grid_search_neural_network_mlp_accuracy = round(accuracy_score(y_test, grid_search_neural_network_mlp_y_pred), 2)
grid_search_neural_network_roc_auc = round(roc_auc_score(y_test, grid_search_neural_network_mlp_y_prob), 2)
grid_search_neural_network_confusion_matrix = confusion_matrix(y_test, grid_search_neural_network_mlp_y_pred)
print("Grid Search Neural Network Best Parameters:", grid_search_neural_network_mlp_best_parameters)
print("Grid Search Neural Network - Accuracy:", grid_search_neural_network_mlp_accuracy)
print("Grid Search Neural Network - ROC & AUC:", grid_search_neural_network_roc_auc)
print("Grid Search Neural Network - Confusion Matrix:")
print(grid_search_neural_network_confusion_matrix)
sns.heatmap(grid_search_neural_network_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
grid_search_neural_network_mlp_fpr, grid_search_neural_network_mlp_tpr, _ = roc_curve(y_test, grid_search_neural_network_mlp_y_prob)
grid_search_neural_network_mlp_auc= roc_auc_score(y_test, grid_search_neural_network_mlp_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(grid_search_neural_network_mlp_fpr, grid_search_neural_network_mlp_tpr, color='blue', label=f"NN (AUC = {grid_search_neural_network_mlp_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Neural Network - MLP using Grid Search')
plt.legend(loc="upper left")
plt.tight_layout()
plt.grid(True)
plt.show()
TP = grid_search_neural_network_confusion_matrix[1][1]
TN = grid_search_neural_network_confusion_matrix[0][0]
FP = grid_search_neural_network_confusion_matrix[0][1]
FN = grid_search_neural_network_confusion_matrix[1][0]
grid_search_neural_network_mlp_precision = round(TP / (TP + FP), 2)
grid_search_neural_network_mlp_recall = round(TP / (TP + FN), 2)
grid_search_neural_network_mlp_f1_score = round(2 * (grid_search_neural_network_mlp_precision * grid_search_neural_network_mlp_recall) / (grid_search_neural_network_mlp_precision + grid_search_neural_network_mlp_recall), 2)
grid_search_neural_network_mlp_specificity = round(TN / (TN + FP), 2)
print("Grid Search Neural Network - Precision:", grid_search_neural_network_mlp_precision)
print("Grid Search Neural Network - Recall:", grid_search_neural_network_mlp_recall)
print("Grid Search Neural Network - F1 score:", grid_search_neural_network_mlp_f1_score)
print("Grid Search Neural Network - Specificity:", grid_search_neural_network_mlp_specificity)
n_splits = 3
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
grid_search_neural_network_mlp_stratified_k_fold = cross_val_score(grid_search_neural_network_mlp, X_train, y_train, cv=stratified_k_fold, scoring='roc_auc')
rounded_auc_scores = [round(score, 2) for score in grid_search_neural_network_mlp_stratified_k_fold]
print("Grid Search Neural Network - Stratified K-fold Cross Validation AUC Scores: ", rounded_auc_scores)
print("Grid Search Neural Network - Stratified K-fold Cross Validation Mean AUC Score: ", round(grid_search_neural_network_mlp_stratified_k_fold.mean(), 2))

print("------------------------- TABLE -------------------------")

from prettytable import PrettyTable
data = [
    ["Decision Tree", basic_decisiontree_confusion_matrix, basic_decisiontree_precision, basic_decisiontree_recall, basic_decisiontree_specificity, basic_decisiontree_f1_score, basic_decisiontree_roc_auc],
    ["Pre Pruned Tree", pre_pruned_confusion_matrix, pre_pruned_decisiontree_precision, pre_pruned_decisiontree_recall, pre_pruned_decisiontree_specificity, pre_pruned_decisiontree_f1_score, pre_pruned_roc_auc],
    ["Post Pruned Tree", post_pruned_confusion_matrix, post_pruned_decisiontree_precision, post_pruned_decisiontree_recall, post_pruned_decisiontree_specificity, post_pruned_decisiontree_f1_score, post_pruned_roc_auc],

    ["Logistic Regression", logistic_regression_confusion_matrix, logistic_regression_precision, logistic_regression_recall, logistic_regression_specificity, logistic_regression_f1_score, logistic_regression_roc_auc],
    ["Grid Search Logistic Regression", grid_search_logistic_regression_confusion_matrix, grid_search_logistic_regression_precision, grid_search_logistic_regression_recall, grid_search_logistic_regression_specificity, grid_search_logistic_regression_f1_score, grid_search_logistic_regression_roc_auc],

    ["K Nearest Neighbor", knn_confusion_matrix, knn_precision, knn_recall, knn_specificity, knn_f1_score, knn_roc_auc],
    ["Elbow Method K Nearest Neighbor", e_knn_confusion_matrix, e_knn_precision, e_knn_recall, e_knn_specificity, e_knn_f1_score, e_knn_roc_auc],


    ["Linear SVM", linear_svm_confusion_matrix, linear_svm_precision, linear_svm_recall, linear_svm_specificity, linear_svm_f1_score, linear_svm_roc_auc],
    ["Grid Search SVM", grid_search_svm_confusion_matrix, grid_search_svm_precision, grid_search_svm_recall, grid_search_svm_specificity, grid_search_svm_f1_score, grid_search_svm_roc_auc],
    ["Polynomial SVM", poly_svm_confusion_matrix, poly_svm_precision, poly_svm_recall, poly_svm_specificity, poly_svm_f1_score, poly_svm_roc_auc],
    ["Radial Basis SVM", rbf_svm_confusion_matrix, rbf_svm_precision, rbf_svm_recall, rbf_svm_specificity, rbf_svm_f1_score, rbf_svm_roc_auc],


    ["Naive Bayes", naive_bayes_confusion_matrix, naive_bayes_precision, naive_bayes_recall, naive_bayes_specificity, naive_bayes_f1_score, naive_bayes_roc_auc],
    ["Grid Search Naive Bayes", grid_search_naive_bayes_confusion_matrix, grid_search_naive_bayes_precision, grid_search_naive_bayes_recall, grid_search_naive_bayes_specificity, grid_search_naive_bayes_f1_score, grid_search_naive_bayes_roc_auc],


    ["Random Forest", random_forest_confusion_matrix, random_forest_precision, random_forest_recall, random_forest_specificity, random_forest_f1_score, random_forest_roc_auc],
    ["Grid Search Random Forest", grid_search_random_forest_confusion_matrix, grid_search_random_forest_precision, grid_search_random_forest_recall, grid_search_random_forest_specificity, grid_search_random_forest_f1_score, grid_search_random_forest_roc_auc],
    ["Random Forest", random_forest_confusion_matrix, random_forest_precision, random_forest_recall, random_forest_specificity, random_forest_f1_score, random_forest_roc_auc],
    ["Random Forest - Bagging", random_forest_bagging_confusion_matrix, random_forest_bagging_precision, random_forest_bagging_recall, random_forest_bagging_specificity, random_forest_bagging_f1_score, random_forest_bagging_roc_auc],
    ["Random Forest - Boosting", random_forest_boosting_confusion_matrix, random_forest_boosting_precision, random_forest_boosting_recall, random_forest_boosting_specificity, random_forest_boosting_f1_score, random_forest_boosting_roc_auc],
    ["Random Forest - Stacking", random_forest_stacking_confusion_matrix, random_forest_stacking_precision, random_forest_stacking_recall, random_forest_stacking_specificity, random_forest_stacking_f1_score, random_forest_stacking_roc_auc],

    ["Neural Network", neural_network_mlp_confusion_matrix, neural_network_mlp_precision, neural_network_mlp_recall, neural_network_mlp_specificity, neural_network_mlp_f1_score, neural_network_mlp_roc_auc],
    ["Grid Search Neural Network", grid_search_neural_network_confusion_matrix, grid_search_neural_network_mlp_precision, grid_search_neural_network_mlp_recall, grid_search_neural_network_mlp_specificity, grid_search_neural_network_mlp_f1_score, grid_search_neural_network_mlp_auc],
]
table = PrettyTable()
table.field_names = ["Classifier", "Confusion Matrix", "Precision", "Recall", "Specificity", "F1 Score", "ROC-AUC"]


