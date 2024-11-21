import requests
import pytest
import argparse
import time

# Default values for the base URL and port
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
DEFAULT_API_PATH = "/condense"


def get_args():
    """
    Function to parse command-line arguments for test configuration.
    """
    parser = argparse.ArgumentParser(description="Test API Endpoints.")
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="API host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="API port (default: 8080)"
    )
    parser.add_argument(
        "--api-path",
        type=str,
        default=DEFAULT_API_PATH,
        help="API path (default: /condense)",
    )

    parser.add_argument(
        "--target-model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
    )

    args, _ = parser.parse_known_args()  # Avoid conflict with pytest's arguments
    return args


# Get arguments from the command line
args = get_args()

# Construct the base URL using the provided arguments
BASE_URL = f"http://{args.host}:{args.port}{args.api_path}"


@pytest.fixture
def api_url():
    """
    Fixture to provide the full API URL based on the host, port, and path.
    """
    return BASE_URL


def test_api_prediction(api_url):
    """
    Test the prediction endpoint by sending a valid context and model request.
    """
    with open("test.txt","r+") as file:
        context_data = file.read()
    payload = {
        "context": context_data,
        "target_model": args.target_model,
    }

    time_elapsed = []
    for trial in range(10):
        t1 = time.time()
        response = requests.post(api_url, json=payload)

        # Ensure the response status is OK
        assert (
            response.status_code == 200
        ), f"Expected status code 200 but got {response.status_code}"

        # Parse the response JSON
        data = response.json()

        # Check that the necessary fields are in the response
        assert "compressed_tokens_b64" in data, "Response should contain compressed_context."

        # Ensure the compressed context is not empty
        assert len(data["compressed_tokens_b64"]) > 0, "Compressed context should not be empty."

        time_elapsed.append(time.time() - t1)

        print(f"Trial {trial+1}: Inference time: {time.time() - t1:.2f} seconds")
    print("Average inference time:", sum(time_elapsed) / len(time_elapsed))
