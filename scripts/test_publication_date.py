import requests
from bs4 import BeautifulSoup

url = (
    "https://aninews.in/news/business/"
    "government-slashes-domestic-natural-gas-prices-amid-major-pricing-overhaul20240601175320/"
)

response = requests.get(
    url,
    headers={
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        )
    },
    timeout=20,
)

print("HTTP status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

for prop in ("article:published_time", "article:published"):
    tag = soup.find("meta", property=prop)
    if tag and tag.get("content"):
        print("Publication date:", tag["content"])
        print("Date source: publisher_metadata")
        break
else:
    print("Publication date: no matching publisher metadata found")
