from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

#
# Dataset fields
#
numeric_features = ["if" + str(i) for i in range(1, 14)]
categorical_features = ["cf" + str(i) for i in range(1, 27)] 
good_cf_num = [1,2,5,6,8,9,14,17,19,20,22,23,25]
# good_cf = ["cf" + str(i) for i in range(1,27) if i in good_cf_num]
good_cf = categorical_features
bad_cf = ["cf" + str(i) for i in range(1,27) if i not in good_cf_num]
fields = ["id", "label"] + numeric_features + categorical_features

#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, good_cf)
    ]
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logregression', LogisticRegression(max_iter=10000)),
])
