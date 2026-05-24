from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
# MODEL LOAD (QWEN)
# -----------------------------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Qwen model... (first run will download ~2-4GB)")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # safe for CPU
    device_map="auto"
)

print("Qwen loaded successfully.")


# -----------------------------
# LLM CALL FUNCTION
# -----------------------------
def run_llm(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a senior commodity market analyst. You analyze financial news and extract structured insights."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# -----------------------------
# PROMPT BUILDER
# -----------------------------
def build_prompt(commodity, text):
    return f"""
Analyze the following commodity news.

Commodity: {commodity}

News Headlines:
{text}

Return clearly:

1. Sentiment: bullish / bearish / neutral
2. Key drivers (bullet points)
3. Short market outlook (2-3 lines)

Be concise and financial-focused.
"""


# -----------------------------
# MAIN AGENT FUNCTION
# -----------------------------
def summarize_articles(articles):
    grouped = defaultdict(list)

    # Group by commodity
    for a in articles:
        grouped[a.get("commodity", "general")].append(a)

    results = {}

    for commodity, items in grouped.items():

        # LIMIT noise (VERY IMPORTANT for LLM quality)
        headlines = "\n".join(
            ["- " + i.get("title", "") for i in items[:10]]
        )

        prompt = build_prompt(commodity, headlines)

        print(f"\n[Agent 2] Analyzing {commodity}...")

        output = run_llm(prompt)

        results[commodity] = {
            "analysis": output,
            "article_count": len(items)
        }

    return results