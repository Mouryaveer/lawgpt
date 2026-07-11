"""Safe JSON writing and resumability helpers."""
import os
import json


def safe_json_write(file_path, data):
    """Safely write JSON atomically."""
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)
    except Exception as e:
        print(f"Error writing JSON to {file_path}: {e}")


def load_year_data(year_file):
    """Load an existing year JSON, or return a template."""
    if os.path.exists(year_file):
        try:
            with open(year_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"year": None, "acts": []}
    return {"year": None, "acts": []}


def get_scraped_act_urls(year_data):
    """Return set of act URLs already scraped in this year."""
    return {act["url"] for act in year_data.get("acts", []) if "url" in act}
