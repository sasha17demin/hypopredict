import re
import html
import requests
from urllib.parse import urlparse, parse_qs
from io import BytesIO

def drive_download_bytes(file_id: str) -> bytes:
    session = requests.Session()
    base = "https://drive.google.com/uc?export=download"
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1) First request (often returns HTML virus warning)
    r = session.get(base, params={"id": file_id}, headers=headers, allow_redirects=True)
    r.raise_for_status()

    text = r.text or ""

    # If we got HTML, try to extract confirm token or direct download URL
    if text.lstrip().startswith("<"):

        # A) Look for a direct download URL embedded in scripts (downloadUrl)
        m = re.search(r'"downloadUrl"\s*:\s*"([^"]+)"', text)
        if m:
            download_url = html.unescape(m.group(1)).replace("\\u003d", "=").replace("\\u0026", "&").replace("\\/", "/")
            rr = session.get(download_url, headers=headers, stream=True)
            rr.raise_for_status()
            return _read_bytes_or_fail(rr)

        # B) Look for href containing confirm=...
        m = re.search(r'href="([^"]+)"', text)
        if m:
            # There can be multiple hrefs; scan all
            for href in re.findall(r'href="([^"]+)"', text):
                if "confirm=" in href and ("uc?" in href or "download" in href):
                    href = html.unescape(href).replace("&amp;", "&")
                    # Make absolute if needed
                    if href.startswith("/"):
                        href = "https://drive.google.com" + href
                    qs = parse_qs(urlparse(href).query)
                    if "confirm" in qs:
                        token = qs["confirm"][0]
                        rr = session.get(base, params={"id": file_id, "confirm": token}, headers=headers, stream=True)
                        rr.raise_for_status()
                        return _read_bytes_or_fail(rr)

        # C) Look for hidden input: name="confirm" value="..."
        m = re.search(r'name="confirm"\s+value="([^"]+)"', text)
        if m:
            token = m.group(1)
            rr = session.get(base, params={"id": file_id, "confirm": token}, headers=headers, stream=True)
            rr.raise_for_status()
            return _read_bytes_or_fail(rr)

        # D) Fallback: cookie-based token (your original approach)
        token = None
        for k, v in r.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        if token:
            rr = session.get(base, params={"id": file_id, "confirm": token}, headers=headers, stream=True)
            rr.raise_for_status()
            return _read_bytes_or_fail(rr)

        # Nothing worked -> show a helpful snippet
        snippet = text[:500]
        raise RuntimeError(
            "Got an HTML interstitial from Google Drive (virus scan warning), "
            "but couldn't extract a confirm token or download URL.\n"
            f"HTML starts with:\n{snippet}"
        )

    # If not HTML, it’s already the file
    rr = session.get(base, params={"id": file_id}, headers=headers, stream=True)
    rr.raise_for_status()
    return _read_bytes_or_fail(rr)


def _read_bytes_or_fail(response: requests.Response) -> bytes:
    buf = BytesIO()
    for chunk in response.iter_content(chunk_size=65536):
        if chunk:
            buf.write(chunk)
    data = buf.getvalue()

    if data.lstrip().startswith(b"<"):
        snippet = data[:300].decode("utf-8", errors="replace")
        raise RuntimeError("Still received HTML, not the file. Preview:\n" + snippet)

    return data
