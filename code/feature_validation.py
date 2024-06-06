from utils import load_data


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


def test_features(dataset):
    for func in validation_functions_features:
        func(dataset)

def test_labels(dataset):
    for func in validation_functions_labels:
        func(dataset)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    test_features(X_train)
    test_features(X_test)
    test_labels(y_train)
    test_labels(y_test)


