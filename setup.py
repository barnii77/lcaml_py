import os
import requests


def download_gist_files(gist_url, local_folder):
    # Check if local folder exists
    if os.path.exists(local_folder):
        raise RuntimeError("Folder already exists.")

    # Create local folder
    os.makedirs(local_folder)

    # Make GET request to fetch the gist
    response = requests.get(gist_url)
    if response.status_code != 200:
        raise RuntimeError("Failed to fetch gist.")

    # Parse response JSON
    files = response.json()["files"]

    # Iterate over files and download each one
    for filename, file_info in files.items():
        content_url = file_info["raw_url"]
        content_response = requests.get(content_url)

        if content_response.status_code != 200:
            raise RuntimeError(f"Failed to download file: {filename}")

        # Save content to local file
        with open(os.path.join(local_folder, filename), "w") as f:
            f.write(content_response.text)

    print("Files downloaded successfully.")


# FIXME: Replace with your own gist URL
gist_url = "https://gist.githubusercontent.com/username/gist_id"
local_folder = "lcaml_interpreter"

download_gist_files(gist_url, local_folder)
