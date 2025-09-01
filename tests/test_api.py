import requests
import json
import os

# Portu ortam değişkeninden al, yoksa 5001 kullan
PORT = os.getenv("API_PORT", "5001")
API_URL = f"http://localhost:{PORT}"


def test_api_health():
    print(f"\nTesting health endpoint at {API_URL}/health")
    response = requests.get(f"{API_URL}/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200
    health_data = response.json()
    assert health_data['status'] == 'healthy'
    assert health_data['model_available'] is True


def test_api_prediction():
    print(f"\nTesting prediction endpoint at {API_URL}/predict")
    sample_data = {
        "features": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]
    }

    # Test valid prediction
    response = requests.post(f"{API_URL}/predict", json=sample_data)
    print(f"Prediction status code: {response.status_code}")
    print(f"Prediction response: {response.text}")

    assert response.status_code == 200
    result = response.json()
    assert 'prediction' in result
    assert result['prediction'] in [0, 1, 2]

    # Test invalid input
    invalid_data = {"features": [1, 2, 3]}
    response = requests.post(f"{API_URL}/predict", json=invalid_data)
    print(f"Invalid input status code: {response.status_code}")
    print(f"Invalid input response: {response.text}")
    assert response.status_code == 400

    # Test metrics endpoint
    metrics_response = requests.get(f"{API_URL}/metrics")
    print(f"Metrics status code: {metrics_response.status_code}")
    assert metrics_response.status_code == 200
    assert 'api_request_count' in metrics_response.text


if __name__ == "__main__":
    print("Running API tests...")
    test_api_health()
    test_api_prediction()
    print("All tests completed!")