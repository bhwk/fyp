from pathlib import Path
from bs4 import BeautifulSoup
import ray


def extract_sections(path_dict: dict) -> list:
    path = path_dict["path"]

    page = open(path)

    soup = BeautifulSoup(page, "html.parser")

    list = []

    for section in soup.find_all("section"):
        list.append(section)

    return list


sample_html_fp = Path("./ray_docs/docs.ray.io/en/master/rllib/rllib-env.html")
print(extract_sections({"path": sample_html_fp})[1])

# DOCS_DIR = Path("./ray_docs/docs.ray.io/en/master/")
# ds = ray.data.from_items(
#     [{"path": path} for path in DOCS_DIR.rglob("*.html") if not path.is_dir()]
# )
#
# print(f"{ds.count()} documents")
