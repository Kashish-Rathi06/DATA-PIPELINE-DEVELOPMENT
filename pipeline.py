import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    'age': [25, 30, None, 22, 28],
    'salary': [50000, 60000, 55000, None, 58000],
    'city': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', None],
    'purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

print("Original Data:")
print(data)

X = data.drop('purchased', axis=1)
y = data['purchased']

numeric_features = ['age', 'salary']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['city']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_processed = pipeline.fit_transform(X)

print("\nProcessed Data (Ready for Model Training):")
print(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print("\nTrain and Test Sets Prepared!")