import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import onnxruntime as ort
from tokenizers import Tokenizer
from rapidfuzz import process, fuzz
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="StoreScout | Geographic Market & Store Location Intelligence",
    page_icon="🏪",
    layout="wide"
)

# -----------------------------------------------------------------------------
# RESOURCE CACHING (In-Process, Zero Torch)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Analytical Data & Models...")
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    onnx_dir = os.path.join(base_dir, "onnx")
    
    # Load Analytical DataFrames
    df_regions = pd.read_csv(os.path.join(data_dir, "regions.csv"))
    df_rc = pd.read_csv(os.path.join(data_dir, "region_category.csv"))
    
    # Load Lookups
    with open(os.path.join(data_dir, "categories_en.json"), "r", encoding="utf-8") as f:
        categories = json.load(f)
    with open(os.path.join(data_dir, "states.json"), "r", encoding="utf-8") as f:
        states = json.load(f)
    with open(os.path.join(data_dir, "cities.json"), "r", encoding="utf-8") as f:
        cities = json.load(f)
    with open(os.path.join(data_dir, "cluster_names.json"), "r", encoding="utf-8") as f:
        cluster_names = json.load(f)
    with open(os.path.join(data_dir, "features.json"), "r", encoding="utf-8") as f:
        features_order = json.load(f)
        
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    kmeans = joblib.load(os.path.join(models_dir, "kmeans.joblib"))
    
    # Load ONNX INT8 Model & Fast Tokenizer (with Hugging Face Hub fallback)
    hf_repo = os.getenv("HF_MODEL_REPO", "GitGuru-sudo/storescout-intent-onnx")
    onnx_path = os.path.join(onnx_dir, "model_int8.onnx")
    tok_path = os.path.join(onnx_dir, "tokenizer.json")
    labels_path = os.path.join(onnx_dir, "labels.json")
    
    if not (os.path.exists(onnx_path) and os.path.exists(tok_path)):
        try:
            onnx_path = hf_hub_download(repo_id=hf_repo, filename="onnx/model_int8.onnx")
            tok_path = hf_hub_download(repo_id=hf_repo, filename="onnx/tokenizer.json")
            labels_path = hf_hub_download(repo_id=hf_repo, filename="onnx/labels.json")
        except Exception:
            pass # Fall back to local paths
            
    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_padding(length=64)
    tokenizer.enable_truncation(max_length=64)
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    with open(labels_path, "r", encoding="utf-8") as f:
        lbls = json.load(f)
        id2intent = {int(k): v for k, v in lbls["id2intent"].items()}
        
    return {
        "df_regions": df_regions,
        "df_rc": df_rc,
        "categories": categories,
        "states": states,
        "cities": cities,
        "cluster_names": cluster_names,
        "features_order": features_order,
        "scaler": scaler,
        "kmeans": kmeans,
        "tokenizer": tokenizer,
        "ort_session": session,
        "id2intent": id2intent
    }

res = load_resources()

# Category synonyms dictionary for multi-lingual and slang queries
CATEGORY_SYNONYMS = {
    "electronics": "telephony",
    "mobile": "telephony",
    "phone": "telephony",
    "gadgets": "computers_accessories",
    "laptop": "computers_accessories",
    "computer": "computers_accessories",
    "khilona": "toys",
    "toy": "toys",
    "furniture": "furniture_decor",
    "sofa": "furniture_decor",
    "beauty": "health_beauty",
    "makeup": "health_beauty",
    "cosmetics": "health_beauty",
    "dawa": "health_beauty",
    "sports": "sports_leisure",
    "gym": "sports_leisure",
    "bed": "bed_bath_table",
    "chadar": "bed_bath_table",
    "watch": "watches_gifts",
    "ghadi": "watches_gifts",
    "car": "auto",
    "bike": "auto"
}

# -----------------------------------------------------------------------------
# SLOT EXTRACTION ENGINE
# -----------------------------------------------------------------------------
def extract_slots(text, previous_slots=None):
    slots = previous_slots.copy() if previous_slots else {
        "zip": None,
        "state": None,
        "city": None,
        "category": None,
        "max_delivery_days": None,
        "n": 5
    }
    
    # 1. ZIP code extraction (5 consecutive digits)
    zip_match = re.search(r'\b(\d{5})\b', text)
    if zip_match:
        slots["zip"] = int(zip_match.group(1))
        
    # 2. Delivery days regex ("5 din", "within 7 days", "1 week", "ek hafte")
    days_match = re.search(r'(\d+)\s*(?:din|day|days)', text, re.IGNORECASE)
    if days_match:
        slots["max_delivery_days"] = int(days_match.group(1))
    week_match = re.search(r'(\d+)\s*(?:week|weeks|hafte|hafta)', text, re.IGNORECASE)
    if week_match:
        slots["max_delivery_days"] = int(week_match.group(1)) * 7
    if re.search(r'\b(?:ek|one)\s*(?:week|hafta)\b', text, re.IGNORECASE):
        slots["max_delivery_days"] = 7
        
    # 3. Top N regex ("top 5", "3 zones", "top 10")
    top_match = re.search(r'\btop\s*(\d+)\b', text, re.IGNORECASE)
    if top_match:
        slots["n"] = int(top_match.group(1))
        
    # 4. State extraction (Exact match against 2-letter state codes)
    tokens = re.findall(r'\b[A-Za-z]{2}\b', text)
    for tok in tokens:
        up = tok.upper()
        if up in res["states"]:
            slots["state"] = up
            break
            
    # 5. Category extraction via synonyms & rapidfuzz
    for syn, mapped_cat in CATEGORY_SYNONYMS.items():
        if re.search(r'\b' + syn + r'\b', text, re.IGNORECASE):
            slots["category"] = mapped_cat
            break
            
    if not slots["category"]:
        cat_match = process.extractOne(text, res["categories"], scorer=fuzz.partial_ratio)
        if cat_match and cat_match[1] >= 75:
            slots["category"] = cat_match[0]
            
    # 6. City extraction via rapidfuzz
    city_match = process.extractOne(text, res["cities"], scorer=fuzz.partial_ratio)
    if city_match and city_match[1] >= 80:
        slots["city"] = city_match[0]
        
    return slots

# -----------------------------------------------------------------------------
# ONNX INTENT CLASSIFICATION
# -----------------------------------------------------------------------------
def classify_intent(text):
    encoding = res["tokenizer"].encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    
    outputs = res["ort_session"].run(None, {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    })[0]
    
    exp_logits = np.exp(outputs[0] - np.max(outputs[0]))
    probs = exp_logits / exp_logits.sum()
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    return res["id2intent"][pred_idx], conf

# -----------------------------------------------------------------------------
# DETERMINISTIC ANALYTICAL TOOLS
# -----------------------------------------------------------------------------
def get_region_by_zip(zip_prefix):
    match = res["df_regions"][res["df_regions"]['zip_prefix'] == zip_prefix]
    if match.empty:
        return None, f"ZIP prefix **{zip_prefix:05d}** ke liye Brazilian e-commerce dataset me koi data nahi mila."
    row = match.iloc[0]
    c_info = res["cluster_names"].get(str(row['cluster']), {"name": f"Cluster {row['cluster']}", "recommendation": "Maintain operations."})
    
    ans = (
        f"**Region Summary: ZIP {zip_prefix:05d} ({row['city'].title()}, {row['state']})**\n\n"
        f"- **Assigned Market Cluster:** {c_info['name']} (Cluster {row['cluster']})\n"
        f"- **Unique Customers:** {int(row['customer_density']):,} customers\n"
        f"- **Total Historic Revenue:** R$ {row['total_revenue']:,.2f}\n"
        f"- **Average Order Value (AOV):** R$ {row['avg_order_value']:.2f}\n"
        f"- **Order Frequency:** {row['order_frequency']:.2f} orders/cust\n"
        f"- **Avg Delivery Duration:** {row['avg_delivery_days']:.1f} days\n\n"
        f"**Strategic Recommendation:** {c_info['recommendation']}"
    )
    return match[['lat', 'lng', 'city', 'state', 'total_revenue', 'avg_delivery_days']], ans

def top_regions(state_or_city=None, n=5):
    df = res["df_regions"].copy()
    if state_or_city:
        sub = df[(df['state'].str.upper() == state_or_city.upper()) | 
                 (df['city'].str.lower() == state_or_city.lower())]
        if not sub.empty:
            df = sub
    df_top = df.sort_values(by='total_revenue', ascending=False).head(n)
    if df_top.empty:
        return None, f"**{state_or_city}** ke liye koi region nahi mila (Brazil data only)."
        
    ans = f"**Top {len(df_top)} High-Revenue Commercial Zones"
    if state_or_city:
        ans += f" in {state_or_city.upper()}"
    ans += ":**\n\n"
    for i, (_, r) in enumerate(df_top.iterrows(), 1):
        ans += f"{i}. **ZIP {int(r['zip_prefix']):05d}** ({r['city'].title()}, {r['state']}): R$ {r['total_revenue']:,.2f} | Delivery: {r['avg_delivery_days']:.1f} days | Cluster {r['cluster']}\n"
    return df_top[['lat', 'lng', 'city', 'state', 'total_revenue', 'avg_delivery_days']], ans

def category_top_regions(category, max_delivery_days=None, n=5):
    df_rc = res["df_rc"].copy()
    sub = df_rc[df_rc['product_category'].str.lower() == category.lower()]
    if sub.empty:
        # Nearest feasible category suggestion
        close_cat = process.extractOne(category, res["categories"])[0]
        return None, f"Category **{category}** ka direct record nahi mila. Nearest match: **{close_cat}**."
        
    if max_delivery_days is not None:
        sub = sub[sub['avg_delivery_days'] <= max_delivery_days]
        if sub.empty:
            min_avail = df_rc[df_rc['product_category'].str.lower() == category.lower()]['avg_delivery_days'].min()
            return None, f"**{category}** ke liye <= {max_delivery_days} din delivery wala koi zone nahi mila. Fastest delivery available: **{min_avail:.1f} din**. Try increasing delivery threshold."
            
    merged = sub.merge(res["df_regions"][['zip_prefix', 'city', 'state', 'lat', 'lng', 'cluster']], on='zip_prefix', how='inner')
    df_top = merged.sort_values(by='revenue', ascending=False).head(n)
    
    ans = f"**Top {len(df_top)} Hubs for '{category}'"
    if max_delivery_days:
        ans += f" (Delivered within {max_delivery_days} days)"
    ans += ":**\n\n"
    for i, (_, r) in enumerate(df_top.iterrows(), 1):
        ans += f"{i}. **ZIP {int(r['zip_prefix']):05d}** ({r['city'].title()}, {r['state']}): R$ {r['revenue']:,.2f} | Delivery: {r['avg_delivery_days']:.1f} din | Cluster {r['cluster']}\n"
    return df_top[['lat', 'lng', 'city', 'state', 'revenue', 'avg_delivery_days']], ans

def explain_clusters():
    ans = "**Overview of Brazilian E-Commerce Market Clusters (K-Means K=4):**\n\n"
    for cid, info in res["cluster_names"].items():
        ans += f"**Cluster {cid}: {info['name']}**\n"
        ans += f"- *Strategic Focus:* {info['strategy']}\n"
        ans += f"- *Action:* {info['recommendation']}\n\n"
    return ans

# -----------------------------------------------------------------------------
# STREAMLIT UI & SIDEBAR
# -----------------------------------------------------------------------------
st.title("StoreScout")
st.caption("Geographic Market & Store Location Intelligence | Brazil Quick-Commerce (Zero-LLM In-Process ONNX INT8)")

# Sidebar for example prompts & Direct GUI fallback
with st.sidebar:
    st.header("Quick Query Starters")
    st.markdown(
        "Click or copy any example query:\n"
        "- `zip 01001 kaisa hai?`\n"
        "- `sao paulo me electronics ke liye 5 din delivery wale best zones`\n"
        "- `top 5 regions in SP`\n"
        "- `cluster 2 ka kya matlab hai?`\n"
        "- `RJ me sabse zyada revenue wale pin codes`\n"
        "- `what can you do?`"
    )
    
    st.divider()
    st.header("Direct GUI Filter (No-ML)")
    with st.form("direct_tool_form"):
        sel_cat = st.selectbox("Product Category", options=res["categories"])
        sel_state = st.selectbox("State Filter", options=["ALL"] + res["states"])
        sel_days = st.slider("Max Delivery Days", min_value=1, max_value=30, value=7)
        sel_n = st.slider("Top N Results", min_value=1, max_value=15, value=5)
        submitted = st.form_submit_button("Query Locations")
        if submitted:
            geo_df, tool_ans = category_top_regions(sel_cat, max_delivery_days=sel_days, n=sel_n)
            st.write(tool_ans)
            if geo_df is not None and not geo_df.empty:
                st.map(geo_df[['lat', 'lng']].rename(columns={'lng': 'lon'}))

# Multi-turn conversation state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Namaste! Main StoreScout Assistant hoon. Brazil quick-commerce market me zip codes, product demand, delivery speeds, aur store expansion recommendations ke baare me poochiye!"}
    ]
if "pending_slots" not in st.session_state:
    st.session_state["pending_slots"] = {}

# Display message history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "map_data" in msg and msg["map_data"] is not None:
            st.map(msg["map_data"][['lat', 'lng']].rename(columns={'lng': 'lon'}))

# Chat Input & Orchestration Loop
user_input = st.chat_input("Enter your retail inquiry (English / Hinglish)...")

if user_input:
    # Append user prompt
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Step 1: Intent Classification
    intent, confidence = classify_intent(user_input)
    
    # Confidence Guardrail (Threshold 0.45)
    CONFIDENCE_THRESHOLD = 0.45
    map_data = None
    
    if confidence < CONFIDENCE_THRESHOLD and intent != "help_greeting":
        bot_response = (
            f"Mujhe aapki query poori tarah samajh nahi aayi (Confidence: {confidence*100:.1f}%). "
            f"Kripya thoda clearly batayein. Example: *'SP me top 5 regions'* ya *'zip 01001 details'*."
        )
    else:
        # Step 2: Slot Extraction with Multi-Turn Memory
        slots = extract_slots(user_input, st.session_state["pending_slots"])
        st.session_state["pending_slots"] = slots
        
        # Step 3: Tool Dispatch
        if intent == "help_greeting":
            bot_response = (
                "Main store expansion aur retail logistics ka assistant hoon. Aap ye pooch sakte hain:\n"
                "1. **ZIP details:** *'zip 01001 kaisa hai?'*\n"
                "2. **Top regions:** *'SP me top 5 revenue zones'*\n"
                "3. **Category demand:** *'toys ke liye 7 din delivery wale best areas'*\n"
                "4. **Clusters explanation:** *'cluster 2 ka kya matlab hai?'*\n"
                "*(Note: All data reflects genuine Brazilian e-commerce operations).*"
            )
        elif intent == "explain_clusters":
            bot_response = explain_clusters()
        elif intent == "zip_lookup":
            if slots["zip"]:
                map_data, bot_response = get_region_by_zip(slots["zip"])
                st.session_state["pending_slots"] = {} # Clear slots on completion
            else:
                bot_response = "Aapne 5-digit zip prefix nahi bataya. Kripya postal prefix likhein (e.g. *zip 01001*)."
        elif intent == "category_top_regions":
            if not slots["category"]:
                bot_response = "Aap kis product category ke liye dekh rahe hain? (e.g. *telephony, toys, health_beauty, furniture*)"
            else:
                map_data, bot_response = category_top_regions(
                    category=slots["category"], 
                    max_delivery_days=slots["max_delivery_days"], 
                    n=slots["n"]
                )
                if slots["max_delivery_days"]:
                    st.session_state["pending_slots"] = {} # Completed
        elif intent == "top_regions":
            filter_loc = slots["state"] or slots["city"]
            map_data, bot_response = top_regions(state_or_city=filter_loc, n=slots["n"])
            st.session_state["pending_slots"] = {}
        elif intent == "out_of_scope":
            bot_response = (
                "Ye query store location intelligence aur Brazilian retail scope se bahar hai. "
                "Main retail market expansion, postal density, aur delivery speed queries me specialize karta hoon!"
            )
        else:
            bot_response = "Kripya query doobara likhein."
            
    # Append assistant response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        if map_data is not None and not map_data.empty:
            st.map(map_data[['lat', 'lng']].rename(columns={'lng': 'lon'}))
            
    st.session_state["messages"].append({
        "role": "assistant", 
        "content": bot_response,
        "map_data": map_data
    })
