import requests

def drive_download_bytes(file_id: str) -> bytes:
    """
    Downloads a file from Google Drive given its file ID and returns its content as bytes.
    Args:
        file_id (str): The unique identifier of the file on Google Drive.
    Returns:
        bytes: The content of the downloaded file.
    """

    session = requests.Session()
    base = "https://drive.google.com/uc?export=download"

    r = session.get(base, params={"id": file_id}, stream=True)
    r.raise_for_status()

    # Grab confirmation token if present
    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        r = session.get(base, params={"id": file_id, "confirm": token}, stream=True)
        r.raise_for_status()

    return r.content
