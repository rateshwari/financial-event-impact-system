
import requests
from bs4 import BeautifulSoup

URL = (
    "https://aninews.in/news/business/"
    "government-slashes-domestic-natural-gas-prices-amid-major-pricing-overhaul"
    "20240601175320/"
)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
}

response = requests.get(URL, headers=headers, timeout=20)
print("HTTP status:", response.status_code)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

print("\nTitle:")
print(soup.title.get_text(" ", strip=True) if soup.title else "MISSING")

print("\nDate-related metadata:")
for tag in soup.find_all("meta"):
    key = tag.get("property") or tag.get("name") or tag.get("itemprop", "")
    value = tag.get("content", "")

    if any(term in key.lower() for term in ("date", "published", "modified", "time")):
        print(f"{key}: {value}")

print("\nDate-related JSON-LD values:")
for tag in soup.find_all("script", type="application/ld+json"):
    text = tag.get_text(" ", strip=True)
    if any(term in text.lower() for term in ("datepublished", "datemodified")):
        print(text[:3000])
