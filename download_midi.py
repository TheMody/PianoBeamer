import os
import urllib.parse
import requests


def download_first_midi(download_href: str, dest_dir: str = "midi_files/") -> str:
    """
    Search BitMidi for *query*, grab the first result, and download its
    .mid file to *dest_dir* (defaults to the current working directory).

    Returns the local path of the saved file.

    Requirements
    ------------
    pip install requests beautifulsoup4

    Example
    -------
    >>> path = download_first_midi("tetris theme")
    >>> print(path)
    ./123456.mid
    """
    session = requests.Session()
    # 1) Search BitMidi

    
    # 4) Stream the file to disk
    filename = os.path.basename(urllib.parse.urlparse(download_href).path)
    local_path = os.path.join(dest_dir, filename)

    with session.get(download_href, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    return local_path


if __name__ == "__main__":
    # Example usage
    dest_dir = "midi_files/"  # Current directory
    download_href = "https://bitmidi.com/uploads/112561.mid"
    try:
        midi_path = download_first_midi(download_href, dest_dir)
        print(f"Downloaded MIDI file to: {midi_path}")
    except Exception as e:
        print(f"Error: {e}")