import requests

def search_gdelt(query: str):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        articles = []

        for item in data.get("articles", [])[:10]:
            articles.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "source": item.get("sourceCountry"),
                "published": item.get("seendate", "")
            })

        return articles

    except Exception as e:
        return []