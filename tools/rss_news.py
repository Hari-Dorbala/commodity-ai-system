import feedparser

RSS_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.investing.com/rss/news_25.rss"
]

def fetch_rss_news():
    articles = []

    for url in RSS_FEEDS:
        feed = feedparser.parse(url)

        for entry in feed.entries[:10]:
            articles.append({
                "title": entry.title,
                "url": entry.link,
                "published": getattr(entry, "published", ""),
                "source": url
            })

    return articles