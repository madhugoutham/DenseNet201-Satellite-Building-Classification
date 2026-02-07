
import os
import requests
import sys

# Zenodo DOI/Record details
ZENODO_RECORD_ID = "18513710"
FILE_NAME = "DenseNet201_Best_Model.h5"
# Construct download URL (Note: This URL works after the record is published)
DOWNLOAD_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files/{FILE_NAME}?download=1"
DESTINATION = os.path.join(os.path.dirname(__file__), FILE_NAME)

def download_file(url, dest):
    print(f"Downloading {FILE_NAME} from Zenodo...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded to {dest}")
        print(f"Size: {os.path.getsize(dest) / (1024*1024):.2f} MB")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("Error: File not found. The Zenodo record might not be published yet.")
            print(f"Please check: https://zenodo.org/record/{ZENODO_RECORD_ID}")
        else:
            print(f"Error downloading file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if os.path.exists(DESTINATION):
        print(f"File {FILE_NAME} already exists in models/ folder.")
        sys.exit(0)
        
    download_file(DOWNLOAD_URL, DESTINATION)
