import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import base64


st.set_page_config(page_title="Cake Cost & Quality Predictor")
st.title("🎂 The Intelligent Baker \n" \
"– Cake Cost, Quality & Optimization Analyzer")
st.markdown(
    "_Intelligent Baker is a smart cake analysis tool that predicts the quality of your cake "
    "based on ingredient quantities and flavor, calculates the total cost, "
    "and suggests optimized ingredient combinations to maintain quality while minimizing cost._"
)
st.set_page_config(page_title="Intelligent Baker", layout="centered")
st.title("🍰 Intelligent Baker")
st.markdown("_An AI-powered cake analyzer that predicts quality, estimates cost, and optimizes ingredients for best results._")

# ============================================================
# BACKGROUND IMAGE WITH DARK OVERLAY
# ============================================================
def get_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

image_base64 = get_base64("bg.jpg")

if image_base64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                        url("data:image/jpg;base64,{image_base64}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
st.markdown(
    """
    <style>
    /* Background overlay on top of background image */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("bg.jpg");
        background-size: cover;
        background-position: center;
    }

    /* Headings and text color for better visibility */
    h1, h2, h3, h4, h5, h6, label, div, span {
        color: #fff !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 182, 193, 0.5) !important;  /* light pink transparent */
        color: #000 !important;
        border-radius: 8px;
    }
    div[data-baseweb="select"] > div:hover {
        background-color: rgba(255, 182, 193, 0.7) !important;
    }

    /* Number inputs / text inputs */
    input[type="number"], input[type="text"] {
        background-color: rgba(255, 182, 193, 0.5) !important;
        color: #000 !important;
        border-radius: 6px !important;
    }

    /* Spin buttons (+/-) in number inputs */
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button {
        background-color: rgba(255, 182, 193, 0.5) !important;
        border-radius: 6px !important;
    }

    /* Streamlit buttons */
    div.stButton > button {
        background-color: rgba(255, 182, 193, 0.5) !important;
        color: #000 !important;
        border-radius: 8px !important;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: rgba(255, 182, 193, 0.7) !important;
        color: #000 !important;
    }

    /* Sidebar background (optional) */
    .css-1d391kg {
        background-color: rgba(0,0,0,0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

image_base64 = get_base64("bg.jpg")
if image_base64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                        url("data:image/jpg;base64,{image_base64}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def load_pickle(path):
    if Path(path).exists():
        return pickle.load(open(path, "rb"))
    else:
        st.error(f"File not found: {path}")
        st.stop()

model = load_pickle("model.pkl")       
scaler = load_pickle("scaler.pkl")     
lab = load_pickle("lab.pkl")           


ingredient_columns = [
    "Flour_Quantity (g)",
    "Sugar_Content (g)",
    "Butter_Quantity (g)",
    "Egg_Count",
    "Milk_Quantity (ml)"
]

price_per_unit = {
    "Flour_Quantity (g)": 0.045,
    "Sugar_Content (g)": 0.045,
    "Butter_Quantity (g)": 0.85,
    "Egg_Count": 7.0,
    "Milk_Quantity (ml)": 0.06
}

OVERHEAD = 160.0
N_ITER = 50  

def total_cake_cost(arr):
    return np.sum(np.array(arr) * np.array([price_per_unit[c] for c in ingredient_columns])) + OVERHEAD


cake_flavors = [
    'Butterscotch', 'White Forest', 'Strawberry', 'Chocolate',
    'Chocolate Truffle', 'Black Forest', 'Pineapple', 'Vanilla',
    'Red Velvet'
]

st.header("🎨 Select Cake Flavor")
selected_flavor = st.selectbox("Choose Cake Flavor", cake_flavors)
flavor_code = cake_flavors.index(selected_flavor)


st.header("🧁 Enter Ingredients")

col1, col2 = st.columns(2)
with col1:
    flour = st.number_input("Flour (g)", 0.0, 1000.0, 250.0)
    sugar = st.number_input("Sugar (g)", 0.0, 1000.0, 150.0)
    butter = st.number_input("Butter (g)", 0.0, 500.0, 100.0)
with col2:
    eggs = st.number_input("Egg Count", 1, 12, 2)
    milk = st.number_input("Milk (ml)", 0.0, 1000.0, 100.0)


st.header("💰 Ingredient Prices")
flour_price = st.number_input("Flour per 100g (₹)", 0.0, 100.0, 4.5)
sugar_price = st.number_input("Sugar per 100g (₹)", 0.0, 100.0, 4.5)
butter_price = st.number_input("Butter per 100g (₹)", 0.0, 200.0, 85.0)
milk_price = st.number_input("Milk per 100ml (₹)", 0.0, 50.0, 6.0)
egg_price = st.number_input("Egg Price (₹)", 0.0, 50.0, 7.0)
overhead = st.number_input("Overhead (₹)", 0.0, 500.0, 160.0)

def compute_cost(f, s, b, e, m):
    return round(
        (f*price_per_unit["Flour_Quantity (g)"]) +
        (s*price_per_unit["Sugar_Content (g)"]) +
        (b*price_per_unit["Butter_Quantity (g)"]) +
        (e*price_per_unit["Egg_Count"]) +
        (m*price_per_unit["Milk_Quantity (ml)"]) +
        OVERHEAD, 2
    )


def predict_quality(flavor_code, arr):
    f, s, b, e, m = arr

    row = {
        "Cake_Flavor": flavor_code,
        "Flour_Quantity (g)": f,
        "Sugar_Content (g)": s,
        "Butter_Quantity (g)": b,
        "Egg_Count": e,
        "Milk_Quantity (ml)": m,
    }

    df_row = pd.DataFrame([row])

    # scale only numeric columns (flavor is NOT scaled)
    scale_cols = scaler.feature_names_in_
    df_row[scale_cols] = scaler.transform(df_row[scale_cols])

    pred_encoded = model.predict(df_row)[0]
    pred_label = lab.inverse_transform([pred_encoded])[0]
    return pred_label



if st.button("Predict Quality"):
    arr = [flour, sugar, butter, eggs, milk]
    quality = predict_quality(flavor_code, arr)
    cost = compute_cost(flour, sugar, butter, eggs, milk)
    st.success(f"🎯 Predicted Quality: {quality}")
    st.info(f"💰 Total Cost: ₹{cost:.2f}")


st.header("🛠 Optimize Ingredients for Cost Savings")

def optimize_ingredients(flavor_code, arr):
    orig_quality = predict_quality(flavor_code, arr)
    orig_cost = compute_cost(*arr)
    best_ing = arr.copy()
    best_cost = orig_cost

    arr = np.array(arr, dtype=float)
    egg_idx = ingredient_columns.index("Egg_Count")
    low = arr * 0.90
    high = arr * 1.10
    low[egg_idx] = max(1, int(low[egg_idx]))
    high[egg_idx] = max(1, int(high[egg_idx]))

    quality_order = {"Poor":1, "Average":2, "Good":3, "Excellent":4}
    min_required_quality = quality_order[orig_quality]

    for _ in range(N_ITER):
        candidate = np.random.uniform(low, high)
        candidate[egg_idx] = max(1, int(round(candidate[egg_idx])))
       
        candidate_label = predict_quality(flavor_code, candidate)
        if quality_order[candidate_label] < min_required_quality:
            continue
        candidate_cost = compute_cost(*candidate)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_ing = candidate.copy()
            best_quality_label = candidate_label

    return best_ing, best_cost, best_quality_label


if st.button("Optimize Cost"):
    arr = [flour, sugar, butter, eggs, milk]
    best_ing, best_cost, best_quality = optimize_ingredients(flavor_code, arr)
    orig_cost = compute_cost(*arr)

    st.subheader("🍰 Cake Flavor")
    st.write(f"**{selected_flavor}**")

    st.subheader("🧁 Optimized Ingredients")
    for name, val in zip(ingredient_columns, best_ing):
        if name == "Egg_Count":
            st.write(f"- {name}: **{int(val)}**")
        else:
            st.write(f"- {name}: **{round(val,2)}**")

    st.subheader("💰 Cost Summary")
    st.write(f"- Original Cost: ₹{orig_cost:.2f}")
    st.write(f"- Optimized Cost: ₹{best_cost:.2f}")
    st.success(f"🎉 Savings: ₹{orig_cost - best_cost:.2f}")

    st.subheader("🎯 Predicted Quality After Optimization")
    st.success(f"**{best_quality}**")