import os
import requests
import sys

def download_dataset(url, dest_path):
    """
    Downloads a dataset from a URL to a destination path with progress logs.
    """
    if os.path.exists(dest_path):
        print(f"[INFO] Dataset already exists at {dest_path}. Skipping download.")
        return True

    print(f"[INFO] Creating directory: {os.path.dirname(dest_path)}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"[INFO] Starting download from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"[DOWNLOAD] {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)", flush=True)
                else:
                    print(f"[DOWNLOAD] {downloaded / (1024*1024):.1f}MB downloaded", flush=True)
                    
        print(f"[SUCCESS] Dataset successfully saved to {dest_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    DATA_URL = "https://huggingface.co/datasets/JEFFREY-VERDIERE/Creditcard/resolve/main/creditcard.csv"
    DEST_PATH = os.path.join("data", "creditcard.csv")
    download_dataset(DATA_URL, DEST_PATH)
