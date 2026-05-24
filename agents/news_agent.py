from tools.gdelt_news import search_gdelt
from tools.rss_news import fetch_rss_news

COMMODITIES = ["gold", "silver", "platinum"]

def run_news_agent():
    print(">> Agent 1 started")

    all_articles = []

    # GDELT
    for commodity in COMMODITIES:
        print(f">> Fetching GDELT for {commodity}")

        query = f"{commodity} price"

        results = search_gdelt(query)

        print("   GDELT results:", len(results))

        for r in results:
            r["commodity"] = commodity
            all_articles.append(r)

    # RSS
    print(">> Fetching RSS")
    rss_results = fetch_rss_news()

    print("   RSS results:", len(rss_results))

    for r in rss_results:
        r["commodity"] = "general"
        all_articles.append(r)

    print(">> Total before dedupe:", len(all_articles))

    # simple dedupe
    seen = set()
    unique = []

    for a in all_articles:
        url = a.get("url")
        if url and url not in seen:
            seen.add(url)
            unique.append(a)

    print(">> Final output:", len(unique))

    return unique