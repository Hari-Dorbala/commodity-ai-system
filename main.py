print("START MAIN")

from agents.news_agent import run_news_agent
from agents.summarizer_agent import summarize_articles
from agents.rag_agent import run_rag_agent

print("IMPORT OK")

def main():
    print("\nRUNNING AGENT 1 (NEWS)")

    articles = run_news_agent()

    print("AGENT 1 RETURNED:", len(articles))

    print("\nRUNNING AGENT 2 (SUMMARIZER)\n")

    summary = summarize_articles(articles)

    print("\nRUNNING AGENT 3 (RAG - LITERATURE REVIEW)\n")
    
    intelligence = run_rag_agent(summary)

    print("\n" + "="*60)
    print("FINAL INTELLIGENCE REPORT WITH LITERATURE INSIGHTS")
    print("="*60 + "\n")

    for commodity, data in intelligence.items():
        print("-----")
        print(f"Commodity: {commodity}")
        
        analysis = data.get("market_analysis", {})
        print(f"Sentiment: {analysis.get('sentiment', 'N/A')}")
        print(f"Drivers: {analysis.get('drivers', [])}")
        print(f"Summary: {analysis.get('summary', analysis.get('analysis', 'N/A'))}")
        print(f"Articles: {analysis.get('article_count', 'N/A')}")
        
        # Literature findings
        print("\n  [LITERATURE REVIEW FINDINGS]")
        findings = data.get("literature_review_findings", [])
        
        if data.get("has_supporting_evidence"):
            for i, finding in enumerate(findings, 1):
                print(f"    {i}. Source: {finding.get('source')}")
                print(f"       Content: {finding.get('content')}")
                print(f"       Relevance: {finding.get('relevance_score'):.2%}")
        else:
            print("    No relevant literature found.")
        
        print()

if __name__ == "__main__":
    main()