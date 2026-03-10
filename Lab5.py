import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

# ── Load & feature engineering (from lab notebook) ───────────────────────────
df = pd.read_csv("talabat_enhanced_orders.csv")
df_fe = df.copy()

df_fe["Order_Time"] = pd.to_datetime(df_fe["Order_Time"], errors="coerce")
df_fe["order_hour"]       = df_fe["Order_Time"].dt.hour
df_fe["order_dayofweek"]  = df_fe["Order_Time"].dt.dayofweek
df_fe["is_weekend"]       = df_fe["order_dayofweek"].isin([5, 6]).astype(int)
df_fe["is_peak_hour"]     = df_fe["order_hour"].isin(list(range(12, 16)) + list(range(19, 24))).astype(int)
df_fe["price_per_item"]   = df_fe["Total_Price"] / df_fe["Quantity"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df_fe["haversine_rest_to_cust_km"] = haversine_km(
    df_fe["Restaurant_Lat"], df_fe["Restaurant_Lon"],
    df_fe["Customer_Lat"],   df_fe["Customer_Lon"]
)

top_items = df_fe["Item_Name"].value_counts().head(20).index
df_fe["Item_Name_reduced"] = np.where(df_fe["Item_Name"].isin(top_items), df_fe["Item_Name"], "Other")

df_fe["price_tier"] = pd.cut(
    df_fe["Total_Price"],
    bins=[0, 100, 250, 500, np.inf],
    labels=["low", "medium", "high", "very_high"]
)

# ── Helper: build and evaluate a pipeline ────────────────────────────────────
def build_and_eval(X_tr, X_te, y_tr, y_te, label=""):
    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    pipe = Pipeline([
        ("preprocess", pre),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42,
                                      n_jobs=-1, class_weight="balanced_subsample"))
    ])
    pipe.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_te, pipe.predict(X_te), zero_division=0))
    return pipe, acc


# ── Base feature matrix ───────────────────────────────────────────────────────
drop_cols = ["Order_ID", "User_ID", "Restaurant_ID", "Driver_ID",
             "Order_Time", "Delivery_Time", "Delivery_Duration_Minutes", "Item_Name"]
drop_cols = [c for c in drop_cols if c in df_fe.columns]

target_col = "Order_Status"
X_base = df_fe.drop(columns=drop_cols + [target_col])
y      = df_fe[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 – New engineered feature: distance_per_price
# ─────────────────────────────────────────────────────────────────────────────
# Justification:
# "distance_per_price" captures how far a delivery travels relative to its
# order value. A high-value order covering a short distance is operationally
# very different from a cheap order spanning many kilometres. Deliveries with
# an unusually high distance-to-price ratio may face higher cancellation risk
# (driver unwilling to travel far for a small payout) or remain In-Transit
# longer, giving the model an additional signal for those minority classes.

df_fe["distance_per_price"] = df_fe["Delivery_Distance_km"] / (df_fe["Total_Price"] + 1e-9)

X_task1 = df_fe.drop(columns=drop_cols + [target_col])
Xtr1, Xte1, ytr1, yte1 = train_test_split(X_task1, y, test_size=0.2, random_state=42, stratify=y)
pipe1, acc1 = build_and_eval(Xtr1, Xte1, ytr1, yte1, "Task 1 – with distance_per_price")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 – Different is_peak_hour rule
# ─────────────────────────────────────────────────────────────────────────────
# Original rule  : lunch 12-15, dinner 19-23
# New rule       : add late-night 0-2 (late-night snack orders that often
#                  show delivery stress / cancellations in urban markets)

df_fe["is_peak_hour_v2"] = df_fe["order_hour"].isin(
    list(range(12, 16)) + list(range(19, 24)) + [0, 1, 2]
).astype(int)

# Swap the old flag for the new one
X_task2 = df_fe.drop(columns=drop_cols + [target_col, "is_peak_hour"])
Xtr2, Xte2, ytr2, yte2 = train_test_split(X_task2, y, test_size=0.2, random_state=42, stratify=y)
pipe2, acc2 = build_and_eval(Xtr2, Xte2, ytr2, yte2, "Task 2 – is_peak_hour_v2 (+ late night 0-2)")

print(f"\nTask 2 comparison – original peak rule acc: {acc1:.4f}  |  new rule acc: {acc2:.4f}")
print("Discussion: adding the late-night window introduces ~3 extra hours marked as peak.")
print("If accuracy changes meaningfully, those hours carry cancellation/transit signal.")
print("If unchanged, peak-hour is already low-importance and the extra window adds no value.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 – Vary top_k for Item_Name_reduced
# ─────────────────────────────────────────────────────────────────────────────
results_task3 = {}

for k in [10, 20, 30, 50]:
    df_tmp = df_fe.copy()
    top_k_items = df_tmp["Item_Name"].value_counts().head(k).index
    df_tmp["Item_Name_reduced"] = np.where(df_tmp["Item_Name"].isin(top_k_items),
                                            df_tmp["Item_Name"], "Other")
    X_tmp = df_tmp.drop(columns=drop_cols + [target_col])
    Xtr, Xte, ytr, yte = train_test_split(X_tmp, y, test_size=0.2, random_state=42, stratify=y)

    cat_c = Xtr.select_dtypes(include=["object", "category"]).columns.tolist()
    num_c = Xtr.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    pre   = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_c),
                               ("num", "passthrough", num_c)])
    pipe  = Pipeline([("pre", pre),
                      ("rf",  RandomForestClassifier(n_estimators=300, random_state=42,
                                                     n_jobs=-1, class_weight="balanced_subsample"))])
    pipe.fit(Xtr, ytr)
    acc = accuracy_score(yte, pipe.predict(Xte))
    results_task3[k] = acc
    print(f"  top_k={k:>3}  →  accuracy={acc:.4f}")

print("\nTask 3 – accuracy vs top_k:")
for k, a in results_task3.items():
    print(f"  top_k={k}: {a:.4f}")
print("Observation: since there are only 9 unique items in the dataset, top_k >= 9")
print("produces identical results. top_k < 9 collapses some items into 'Other'.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 – Feature selection with SelectFromModel
# ─────────────────────────────────────────────────────────────────────────────
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

model_fs = Pipeline([
    ("preprocess", preprocess),
    ("select", SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=300, random_state=42,
                                         n_jobs=-1, class_weight="balanced_subsample"),
        threshold="median"
    )),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42,
                                   n_jobs=-1, class_weight="balanced_subsample"))
])

model_fs.fit(X_train, y_train)
y_pred_fs = model_fs.predict(X_test)
acc_fs = accuracy_score(y_test, y_pred_fs)

print(f"\n{'='*55}")
print(f"  Task 4 – with SelectFromModel (threshold=median)")
print(f"  Accuracy: {acc_fs:.4f}")
print(classification_report(y_test, y_pred_fs, zero_division=0))

n_selected = model_fs.named_steps["select"].get_support().sum()
print(f"  Features selected: {n_selected}")
print("\nExplanation:")
print("  SelectFromModel keeps features above the median importance score,")
print("  cutting roughly half the features. If accuracy stays similar to the")
print("  baseline (~0.85), the removed features carried little signal and the")
print("  simpler model is preferred. A drop in accuracy would indicate that")
print("  the discarded features (e.g. some GPS columns) were actually useful.")
