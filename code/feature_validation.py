import pandas as pd
import pytest


#X_train = pd.read_csv("data/X_train.csv")
#X_test = pd.read_csv("data/X_test.csv")
#y_train = pd.read_csv("data/y_train.csv")
#y_test = pd.read_csv("data/y_test.csv") 

X_train = pd.read_csv("../data/X_train.csv")
X_test = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv")
y_test = pd.read_csv("../data/y_test.csv") 


def validate_credit_score_range(df):
    assert df['credit_score'].between(300, 850).all(), "Credit scores should be between 300 and 850"

def validate_age_range(df):
    assert df['age'].between(18, 92).all(), "Age should be between 18 and 92"

def validate_tenure_range(df):
    assert df['tenure'].between(0, 10).all(), "Tenure should be between 0 and 10 years"

def validate_balance_non_negative(df):
    assert (df['balance'] >= 0).all(), "Balance should be non-negative"

def validate_products_number_range(df):
    assert df['products_number'].between(1, 4).all(), "Number of products should be between 1 and 4"

def validate_credit_card_binary(df):
    assert df['credit_card'].isin([0, 1]).all(), "Credit card flag should be 0 or 1"

def validate_active_member_binary(df):
    assert df['active_member'].isin([0, 1]).all(), "Active member flag should be 0 or 1"

def validate_gender_values(df):
    assert df['gender'].isin(['Male', 'Female']).all(), "Gender should be 'Male' or 'Female'"

def validate_country_values(df):
    valid_countries = ['France', 'Spain', 'Germany']
    assert df['country'].isin(valid_countries).all(), "Country should be one of the valid countries"

def validate_customer_id_unique(df):
    assert df['customer_id'].is_unique, "Customer ID should be unique"

def validate_labels(df):
    assert df['churn'].isin([0, 1]).all(), "The label should be binary"

validation_functions_features = [
    validate_credit_score_range,
    validate_age_range,
    validate_tenure_range,
    validate_balance_non_negative,
    validate_products_number_range,
    validate_credit_card_binary,
    validate_active_member_binary,
    validate_gender_values,
    validate_country_values,
    validate_customer_id_unique
]

validation_functions_labels = [
    validate_labels
]


@pytest.mark.parametrize("dataset", [X_train, X_test])
def test_features(dataset):
    for func in validation_functions_features:
        func(dataset)

@pytest.mark.parametrize("dataset", [y_train, y_test])
def test_labels(dataset):
    for func in validation_functions_labels:
        func(dataset)

if __name__ == "__main__":
    pytest.main()


