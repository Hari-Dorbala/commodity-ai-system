import streamlit as st
import requests
from typing import List, Dict, Any
from agents.rag_agent import RAGAgent
# Single-tab simplified UI: select commodity, show news summary, then 30d and 180d predictions
from tools.rss_news import fetch_rss_news
from tools.gdelt_news import search_gdelt
from tools.commodity_predictor import CommodityPredictor, COMMODITY_TICKERS
import plotly.graph_objects as go


# Initialize RAG Agent (cached)
@st.cache_resource
def load_rag_agent():
    try:
        with st.spinner("Loading RAG model... This may take a moment..."):
            rag = RAGAgent(pdf_directory="data/literature_reviews")
            return rag
    except Exception as e:
        st.error(f"Error loading RAG agent: {e}")
        return None

# Load the agent once
rag_agent = load_rag_agent()
if rag_agent is None:
    st.warning("⚠️ RAG agent not initialized. Some features may be limited.")
else:
    st.success("✅ RAG agent loaded successfully!")

st.markdown("### 📈 Commodity Analysis & Prediction")

commodity = st.selectbox(
    "Select commodity:",
    options=list(COMMODITY_TICKERS.keys()),
    index=0,
    help="Choose a commodity to analyze (news + predictions)"
)

analyze = st.button("🔎 Analyze Commodity")

def simple_news_summary(title_list):
    # basic sentiment heuristic
    pos_words = ['up','increase','rise','surge','gain','higher','bull','positive','rally']
    neg_words = ['down','drop','fall','decline','lower','bear','negative','slump']
    pos = neg = 0
    for t in title_list:
        text = t.lower()
        for w in pos_words:
            if w in text:
                pos += 1
        for w in neg_words:
            if w in text:
                neg += 1
    return pos, neg

if analyze:
    if rag_agent is None:
        st.error("RAG agent not initialized. Ensure PDFs are available in the configured directory.")
    else:
        with st.spinner("Fetching news and running predictions..."):
            # Fetch recent news from RSS + GDELT
            rss = fetch_rss_news()
            gdelt = search_gdelt(commodity)
            # Collect headlines
            headlines = []
            for a in rss[:8]:
                headlines.append(a.get('title',''))
            for a in gdelt[:8]:
                headlines.append(a.get('title',''))

            if not headlines:
                st.info("No recent news found for this commodity.")
            else:
                st.markdown("---")
                st.markdown("### 🗞️ Recent News Summary")
                st.write("Top headlines:")
                for h in headlines[:10]:
                    st.write(f"- {h}")

                pos, neg = simple_news_summary(headlines)
                if pos > neg:
                    news_trend = 'uptick'
                    news_note = 'News leans positive / bullish.'
                elif neg > pos:
                    news_trend = 'downtick'
                    news_note = 'News leans negative / bearish.'
                else:
                    news_trend = 'neutral'
                    news_note = 'News is mixed or neutral.'

                st.info(f"{news_note} (positive headlines: {pos}, negative headlines: {neg})")

            # Run prediction (uses 1 year lookback by default)
            predictor = CommodityPredictor(commodity, lookback_days=365)
            summary = predictor.full_pipeline()

            if not summary:
                st.warning(f"⚠️ Could not generate predictions for {commodity}")
            else:
                # Display current price and last date
                current_price = summary['current_price']
                last_date = summary.get('last_date', '')

                st.markdown("---")
                st.markdown("### 📊 Predictions (Model)")
                st.write(f"Current price: **${current_price:.2f}** (as of {last_date})")

                p1 = summary['predictions']['1_month']
                p6 = summary['predictions']['6_months']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("30-Day Prediction", f"${p1['price']:.2f}", f"Change: {p1['change']:+.2f}%")
                    st.write(f"Range: ${p1['min']:.2f} - ${p1['max']:.2f}")
                with col2:
                    st.metric("180-Day Prediction", f"${p6['price']:.2f}", f"Change: {p6['change']:+.2f}%")
                    st.write(f"Range: ${p6['min']:.2f} - ${p6['max']:.2f}")

                # Combine news trend with prediction ranges to give advisory
                # Simple adjustment: +/- 0.5% per net headline, capped at 5%
                adj_pct = 0
                if 'headlines' in locals() and headlines:
                    adj_pct = min(5, max(-5, (pos - neg) * 0.5))

                suggested_1m = p1['price'] * (1 + adj_pct/100)
                suggested_6m = p6['price'] * (1 + adj_pct/100)

                st.markdown("---")
                st.markdown("### 🤖 Combined Insight (News + Model)")
                insight_lines = []
                insight_lines.append(f"Model 30-day prediction (point): ${p1['price']:.2f} — range ${p1['min']:.2f} to ${p1['max']:.2f}.")
                insight_lines.append(f"Model 180-day prediction (point): ${p6['price']:.2f} — range ${p6['min']:.2f} to ${p6['max']:.2f}.")

                if adj_pct > 0:
                    insight_lines.append(f"News signals suggest a positive tilt (estimated adjustment +{adj_pct:.1f}%), so prices could move toward ${suggested_1m:.2f} in 30 days and ${suggested_6m:.2f} in 180 days.")
                elif adj_pct < 0:
                    insight_lines.append(f"News signals suggest a negative tilt (estimated adjustment {adj_pct:.1f}%), so prices could move toward ${suggested_1m:.2f} in 30 days and ${suggested_6m:.2f} in 180 days.")
                else:
                    insight_lines.append(f"News signals are neutral; no adjustment suggested to model point predictions.")

                # If model lower-end is notably below suggested, mention it
                if suggested_1m > p1['max']:
                    insight_lines.append(f"Note: model's upper bound (${p1['max']:.2f}) is below news-suggested level ${suggested_1m:.2f}.")
                elif suggested_1m < p1['min']:
                    insight_lines.append(f"Note: model's lower bound (${p1['min']:.2f}) is above news-suggested level ${suggested_1m:.2f}.")

                for line in insight_lines:
                    st.write(f"- {line}")

                # Simple chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=['Current', '30 Days', '180 Days'],
                    y=[current_price, p1['price'], p6['price']],
                    mode='lines+markers',
                    name='Price Forecast',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=10)
                ))
                fig.update_layout(title=f"{commodity.upper()} Forecast (News-aware)", xaxis_title="Timeframe", yaxis_title="Price ($)", height=400)
                st.plotly_chart(fig, use_container_width=True)
