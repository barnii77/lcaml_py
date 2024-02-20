import os
import requests
from urllib.parse import urlparse


def download_gist_files(gist_url, local_folder):
    # Check if local folder exists
    if os.path.exists(local_folder):
        raise RuntimeError("Folder already exists.")

    # Create local folder
    os.makedirs(local_folder)

    # Extract the Gist ID from the URL
    gist_id = os.path.basename(urlparse(gist_url).path)

    # Make GET request to fetch the gist metadata
    metadata_url = f"https://api.github.com/gists/{gist_id}"
    response = requests.get(metadata_url)
    if response.status_code != 200:
        raise RuntimeError("Failed to fetch gist metadata.")

    # Parse response JSON
    gist_data = response.json()

    # Iterate over files and download each one
    for filename, file_info in gist_data["files"].items():
        content_url = file_info["raw_url"]
        content_response = requests.get(content_url)

        if content_response.status_code != 200:
            raise RuntimeError(f"Failed to download file: {filename}")

        # Save content to local file
        with open(os.path.join(local_folder, filename), "w") as f:
            f.write(content_response.text)

    print("Files downloaded successfully.")


gist_url = (
    "https://gist.githubusercontent.com/Schimeltuer144/453f6ba8d667dbfdb73e55ce5ba72b99"
)
local_folder = "el_carml_interpreter"

download_gist_files(gist_url, local_folder)
