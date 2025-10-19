
# --- Jupyter-friendly Datavizmain workflow ---
import google.generativeai as genai
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score
from pydantic import BaseModel

# Set your Gemini API key directly
GEMINI_API_KEY = "AIzaSyBNwx44BKFWTinOWXlST9tzmKhzodvkyGE"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

# Example synthetic data (replace with your own or Gemini-generated)
x_val = ["January", "February", "March", "April", "May", "June", "July", "August"]
y_val = [15000.0, 16200.0, 14800.0, 17500.0, 18300.0, 19100.0, 17900.0, 20500.0]
x_label = "Month"
y_label = "Sales"
graph_type = "line_plot"

class GraphData(BaseModel):
    x_val: list
    y_val: list
    x_label: str
    y_label: str
    graph_type: str
    summary: str = ""
    insights: str = ""
    df: pd.DataFrame = None

    class Config:
        arbitrary_types_allowed = True

    def descriptive_stats(self):
        s = pd.Series(self.y_val)
        desc = s.describe()
        mean = desc["mean"] if "mean" in desc else None
        median = s.median()
        mode = s.mode().iloc[0] if not s.mode().empty else None
        std = s.std()
        variance = std ** 2 if std is not None else None
        quartiles = s.quantile([0.25, 0.5, 0.75]).to_dict()
        raw = (
            f"Mean: {mean}\nMedian: {median}\nMode: {mode}\n"
            f"Std Dev: {std}\nVariance: {variance}\nQuartiles: {quartiles}"
        )
        print("\n=== Descriptive Statistics ===")
        print(raw)
        return raw

    def visualise(self):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.x_val, y=self.y_val, palette="muted")
        plt.title(f"{self.graph_type.title()} of {self.y_label} over {self.x_label}")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()

# Create the data object directly (no input())
data = GraphData(
    x_val=x_val,
    y_val=y_val,
    x_label=x_label,
    y_label=y_label,
    graph_type=graph_type
)
print("[Debug] Data object created:", data)

# Run analysis and visualization
desc_stats = data.descriptive_stats()
data.visualise()

# Example Gemini API call for summary
combined_prompt = (
    f"You are an expert data analyst. Given the following results, provide a concise summary and 2-3 key insights.\n"
    f"Descriptive statistics: {desc_stats}\n"
    f"Summarize the overall trends, relationships, and any anomalies."
)
try:
    response = gemini_model.generate_content(combined_prompt)
    summary = response.text.strip()
    print("\n=== Gemini Combined Summary/Insights ===")
    print(summary)
except Exception as e:
    print(f"Gemini API error: {e}")
