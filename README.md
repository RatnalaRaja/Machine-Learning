# EDA and Feature Engineering Revision Guide

This document provides a quick and practical reference for revising **Exploratory Data Analysis (EDA)** and **Feature Engineering** in data science and machine learning workflows.

---

##  Exploratory Data Analysis (EDA)

EDA helps understand the dataset, detect anomalies, discover patterns, and form hypotheses using summary statistics and visualizations.

### 1. **Understanding the Data**
- Load the dataset: `pandas.read_csv()`, etc.
- View structure: `.head()`, `.info()`, `.shape`, `.columns`, `.dtypes`
- Check summary statistics: `.describe()`, `.value_counts()`

### 2. **Univariate Analysis**
- Numerical features: Histogram, Boxplot, KDE plot
- Categorical features: Bar plot, Pie chart
- Check for outliers: Boxplots, IQR method

### 3. **Bivariate/Multivariate Analysis**
- Correlation matrix: `df.corr()`, `sns.heatmap()`
- Scatter plots: `sns.scatterplot()`, `pairplot()`
- Grouped statistics: `groupby()`, pivot tables
- Categorical vs Target: `sns.boxplot(x=cat_feature, y=target)`

### 4. **Missing Value Analysis**
- Detect missing values: `df.isnull().sum()`
- Visualize missingness: `missingno.matrix()`, `heatmap()`

### 5. **Data Imbalance**
- Check class distribution
- Use visual tools: Count plot, Pie chart

### 6. **EDA Tools & Libraries**
- `pandas`, `matplotlib`, `seaborn`
- Profiling tools: `pandas_profiling`, `sweetviz`

---

## 🧰 Feature Engineering

Feature engineering transforms raw data into meaningful features to improve model performance.

### 1. **Handling Missing Values**
- Numerical: Mean, median, interpolation
- Categorical: Mode, “Unknown” category
- Advanced: KNN imputation, regression imputation

### 2. **Encoding Categorical Variables**
- Label Encoding
- One-Hot Encoding
- Ordinal Encoding
- Target Encoding (careful of data leakage)

### 3. **Feature Scaling**
- Standardization: `(x - mean) / std`
- Normalization: `(x - min) / (max - min)`
- Tools: `StandardScaler`, `MinMaxScaler`, `RobustScaler`

### 4. **Creating New Features**
- Date-time: Extract day, month, year, weekday, etc.
- Aggregated stats: Mean, median, sum over groups
- Domain-specific features (ratios, differences)
- Binning: Convert continuous to categorical bins

### 5. **Dealing with Outliers**
- IQR Method, Z-score method
- Cap and floor (winsorizing)
- Remove or transform (e.g., log transform)


---

##  Best Practices
- Always perform EDA before modeling.
- Document your assumptions and findings.
- Beware of data leakage during feature engineering.
- Perform feature engineering with training data only, and apply the same on test/validation sets.

---

##  Useful Code Snippets


# Correlation Heatmap
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# One-hot Encoding
pd.get_dummies(df, columns=['Category'])

# Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Income']])
