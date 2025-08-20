# DATA-PIPELINE-DEVELOPMENT
Company: CODTECH IT SOLUTIONS

NAME: Kashish Rathi

INTERN ID:CT06DH2160

DOMAIN:Data Science

DURACTION: 6 WEEKS

MENTOR: NEELA SANTOSH
---

### 📝 Script Summary: Data Preprocessing Pipeline using Scikit-Learn

This Python script demonstrates how to preprocess a dataset using `pandas` and `scikit-learn`. It covers key steps in preparing data for machine learning, including handling missing values, scaling numerical features, encoding categorical data, and splitting the dataset into training and test sets.

#### 📊 Dataset Overview

The dataset includes four columns:

* `age` and `salary` (numerical, with missing values)
* `city` (categorical, with a missing value)
* `purchased` (target variable: Yes/No)

The target (`purchased`) is separated from the input features (`X`), which include both numerical and categorical data.

#### 🔧 Preprocessing Steps

The script builds two separate pipelines:

1. **Numerical Pipeline**:

   * Imputes missing values with the column mean.
   * Applies standard scaling (zero mean, unit variance).

2. **Categorical Pipeline**:

   * Fills missing values with the most frequent category.
   * Uses one-hot encoding to convert categories into binary columns.

These are combined using a `ColumnTransformer` to apply the correct processing to each column type.

#### 🔄 Transformation & Splitting

The combined pipeline is then applied to the feature set using `.fit_transform()`, resulting in a fully numeric and clean dataset ready for model training. The transformed data is then split into training and test sets using `train_test_split`.

#### ✅ Outcome

The script ensures:

* Missing values are handled
* Data is scaled and encoded
* Output is suitable for machine learning models

---

#OUTPUT

<img width="1913" height="1073" alt="Image" src="https://github.com/user-attachments/assets/7c8c0797-aa72-4910-8131-da12a470820f" />



