print("START MAIN")

from agents.news_agent import run_news_agent
from agents.summarizer_agent import summarize_articles

print("IMPORT OK")

def main():
    print("\nRUNNING AGENT 1 (NEWS)")

    articles = run_news_agent()

    print("AGENT 1 RETURNED:", len(articles))

    print("\nRUNNING AGENT 2 (SUMMARIZER)\n")

    summary = summarize_articles(articles)

    print("\nFINAL INTELLIGENCE OUTPUT:\n")

    for commodity, data in summary.items():
        print("-----")
        print("Commodity:", commodity)
        print("Sentiment:", data.get("sentiment", "N/A"))
        print("Drivers:", data.get("drivers", []))
        print("Summary:", data.get("summary", data.get("analysis")))
        print("Articles:", data.get("article_count", len(articles)))

if __name__ == "__main__":
    main()