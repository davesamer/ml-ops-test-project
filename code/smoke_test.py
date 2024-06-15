import requests
import json
import sys


def main(base_url):
    url = f"{base_url}/invocations"
    headers = {"Content-Type": "application/json"}

    data = {
        "dataframe_records": [
            {
                "customer_id": 1,
                "credit_score": 779,
                "country": "Spain",
                "gender": "Female",
                "age": 34,
                "tenure": 5,
                "balance": 0.0,
                "products_number": 2,
                "credit_card": 0,
                "active_member": 1,
                "estimated_salary": 111676.63,
            },
            {
                "customer_id": 445,
                "credit_score": 748,
                "country": "France",
                "gender": "Female",
                "age": 26,
                "tenure": 1,
                "balance": 77780.29,
                "products_number": 1,
                "credit_card": 0,
                "active_member": 1,
                "estimated_salary": 183049.41,
            },
            {
                "customer_id": 9505,
                "credit_score": 651,
                "country": "France",
                "gender": "Male",
                "age": 28,
                "tenure": 10,
                "balance": 79562.98,
                "products_number": 1,
                "credit_card": 1,
                "active_member": 1,
                "estimated_salary": 74687.37,
            },
            {
                "customer_id": 332,
                "credit_score": 663,
                "country": "Germany",
                "gender": "Male",
                "age": 44,
                "tenure": 2,
                "balance": 117028.6,
                "products_number": 2,
                "credit_card": 0,
                "active_member": 1,
                "estimated_salary": 144680.18,
            },
            {
                "customer_id": 4168,
                "credit_score": 1,
                "country": "France",
                "gender": "Male",
                "age": 37,
                "tenure": 8,
                "balance": 0.0,
                "products_number": 1,
                "credit_card": 1,
                "active_member": 0,
                "estimated_salary": 101834.58,
            },
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    print("Response Code:", response.status_code)
    if response.status_code == 200:
        print("Response Data:", response.json())
    else:
        print("Error:", response.text)
        sys.exit(1)  # Exit with error code if the response is not successful


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoke_test.py <base_url>")
        sys.exit(1)

    base_url = sys.argv[1]
    main(base_url)
