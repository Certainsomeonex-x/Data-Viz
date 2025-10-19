# ------------------------------------------------------------
# stats_dataviz.py
# ------------------------------------------------------------
# A single-file extension of your original Datavizmain.py.
# It adds:
#   • Descriptive & inferential statistics
#   • Time-series analysis (moving average, ARIMA, decomposition)
#   • Clustering (K-Means) & classification (Decision Tree)
#   • Visualisation helpers (scatter, histogram, bar, box, heatmap)
#   • Polynomial regression
#   • Generalized Additive Model (GAM) via statsmodels
#   • Simple piecewise-linear MARS-style approximation
#   • Gemini (genai) integration for auto-generated summaries/insights
#
# Save this file next to your original Datavizmain.py (or replace it
# if you prefer).  Make sure you have a .env file with:
#   GEMINI_API_KEY=your_actual_key_here
# ------------------------------------------------------------

from dotenv import load_dotenv
import os
import json
from typing import List, Optional

# Gemini / GenAI
import google.generativeai as genai   # <-- correct import name

# Data / visualisation libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Stats / ML libraries
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score

# Pydantic model
from pydantic import BaseModel

# ----------------------------------------------------------------------
# 1. Load environment & configure Gemini
# ----------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")   # you can change the model name

# ----------------------------------------------------------------------
# 2. GraphData model – extended with all statistical methods
# ----------------------------------------------------------------------
class GraphData(BaseModel):
    x_val: List[float]
    y_val: List[float]
    x_label: str
    y_label: str
    graph_type: str
    summary: str = ""
    insights: str = ""
    df: Optional[pd.DataFrame] = None   # optional full dataframe for multivariate work

    # ------------------------------------------------------------------
    # 2.1 Factory methods (CSV / JSON)
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        file_path: str,
        x_col: str,
        y_col: str,
        x_label: str = "",
        y_label: str = "",
        graph_type: str = "",
        summary: str = "",
        insights: str = "",
    ):
        df = pd.read_csv(file_path)
        return cls(
            x_val=df[x_col].tolist(),
            y_val=df[y_col].tolist(),
            x_label=x_label or x_col,
            y_label=y_label or y_col,
            graph_type=graph_type,
            summary=summary,
            insights=insights,
            df=df,
        )

    @classmethod
    def from_json(
        cls,
        file_path: str,
        x_key: str,
        y_key: str,
        x_label: str = "",
        y_label: str = "",
        graph_type: str = "",
        summary: str = "",
        insights: str = "",
    ):
        with open(file_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame({x_key: data[x_key], y_key: data[y_key]})
        return cls(
            x_val=data[x_key],
            y_val=data[y_key],
            x_label=x_label or x_key,
            y_label=y_label or y_key,
            graph_type=graph_type,
            summary=summary,
            insights=insights,
            df=df,
        )

    # ------------------------------------------------------------------
    # 2.2 Helper: ask Gemini for a concise insight / summary
    # ------------------------------------------------------------------
    def _gemini_insight(self, analysis_type: str, raw_results: str) -> str:
        prompt = (
            f"Analyse the following {analysis_type} results for a dataset "
            f"with X = {self.x_label} and Y = {self.y_label}.\n"
            f"Results:\n{raw_results}\n"
            "Provide a short, plain-English summary (≤ 2 sentences) and 1-2 key insights."
        )
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    # ------------------------------------------------------------------
    # 2.3 Descriptive statistics
    # ------------------------------------------------------------------
    def descriptive_stats(self):
        df = self.df[[self.x_label, self.y_label]] if self.df is not None else pd.DataFrame(
            {self.x_label: self.x_val, self.y_label: self.y_val}
        )
        desc = df.describe()
        mean = desc.loc["mean"].to_dict()
        median = df.median().to_dict()
        mode = df.mode().iloc[0].to_dict()
        std = desc.loc["std"].to_dict()
        variance = {k: v ** 2 for k, v in std.items()}
        quartiles = df.quantile([0.25, 0.5, 0.75]).to_dict()

        raw = (
            f"Mean: {mean}\nMedian: {median}\nMode: {mode}\n"
            f"Std Dev: {std}\nVariance: {variance}\nQuartiles: {quartiles}"
        )
        self.summary = self._gemini_insight("descriptive statistics", raw)
        print("\n=== Descriptive Statistics ===")
        print(self.summary)
        return raw

    # ------------------------------------------------------------------
    # 2.4 Inferential statistics (t-test, confidence intervals, correlation)
    # ------------------------------------------------------------------
    def inferential_stats(self, confidence: float = 0.95, test: str = "t-test"):
        # t-test (independent samples)
        if test == "t-test":
            t_stat, p_val = stats.ttest_ind(self.x_val, self.y_val)
            t_res = f"T-stat: {t_stat:.4f}, p-value: {p_val:.4f}"
        else:
            t_res = "No test performed"

        # Confidence intervals
        ci_x = stats.norm.interval(confidence, loc=np.mean(self.x_val), scale=stats.sem(self.x_val))
        ci_y = stats.norm.interval(confidence, loc=np.mean(self.y_val), scale=stats.sem(self.y_val))
        ci_res = f"CI ({confidence*100:.0f}%): X={ci_x}, Y={ci_y}"

        # Pearson correlation
        corr, _ = stats.pearsonr(self.x_val, self.y_val)
        corr_res = f"Pearson correlation: {corr:.4f}"

        raw = f"{t_res}\n{ci_res}\n{corr_res}"
        self.insights = self._gemini_insight("inferential statistics", raw)
        print("\n=== Inferential Statistics ===")
        print(self.insights)
        return raw

    # ------------------------------------------------------------------
    # 2.5 Time-series analysis (moving avg, ARIMA, decomposition)
    # ------------------------------------------------------------------
    def time_series_analysis(self, period: int = 7, forecast_steps: int = 5):
        # Build a pandas Series with a datetime index
        dates = pd.date_range(start="2023-01-01", periods=len(self.y_val), freq="D")
        ts = pd.Series(self.y_val, index=dates, name=self.y_label)

        # Moving average
        ma = ts.rolling(window=period).mean()

        # ARIMA (simple (1,1,1) – adjust as needed)
        arima = ARIMA(ts, order=(1, 1, 1)).fit()
        forecast = arima.forecast(steps=forecast_steps)

        # Seasonal decomposition (additive)
        decomp = seasonal_decompose(ts, model="additive", period=period)

        raw = (
            f"Moving average (last 5): {ma[-5:].to_dict()}\n"
            f"ARIMA forecast: {forecast.to_dict()}\n"
            f"Decomposition trend (last 5): {decomp.trend[-5:].to_dict()}"
        )
        self.summary = self._gemini_insight("time-series analysis", raw)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(ts, label="Original")
        plt.plot(ma, label="Moving Avg")
        plt.plot(forecast, label="ARIMA Forecast")
        plt.title("Time-Series Analysis")
        plt.legend()
        plt.show()

        print("\n=== Time-Series Summary ===")
        print(self.summary)
        return raw

    # ------------------------------------------------------------------
    # 2.6 Clustering (K-Means) + optional classification (Decision Tree)
    # ------------------------------------------------------------------
    def clustering_classification(self, n_clusters: int = 3, classify: bool = False):
        data = np.column_stack((self.x_val, self.y_val))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        labels = kmeans.labels_

        raw = f"K-Means centroids:\n{kmeans.cluster_centers_}\nSample labels: {labels[:10]}"

        if classify:
            tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(data, labels)
            raw += "\nDecision tree fitted (use plot_tree(tree) to visualise)."

        self.insights = self._gemini_insight("clustering/classification", raw)

        # Plot clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.x_val, y=self.y_val, hue=labels, palette="viridis")
        plt.title("K-Means Clustering")
        plt.show()

        print("\n=== Clustering Insights ===")
        print(self.insights)
        return raw

    # ------------------------------------------------------------------
    # 2.7 General visualisation helpers
    # ------------------------------------------------------------------
    def visualise(self, viz_type: str = "scatter"):
        plt.figure(figsize=(10, 6))
        sns.set_style("darkgrid")

        if viz_type == "histogram":
            sns.histplot(self.y_val, kde=True, color="skyblue")
            plt.title("Histogram")
        elif viz_type == "bar":
            # Show first 10 points as an example
            sns.barplot(x=self.x_val[:10], y=self.y_val[:10], palette="muted")
            plt.title("Bar Chart (first 10)")
        elif viz_type == "scatter":
            sns.scatterplot(x=self.x_val, y=self.y_val, color="teal")
            plt.title("Scatter Plot")
        elif viz_type == "box":
            sns.boxplot(data=pd.DataFrame({self.x_label: self.x_val, self.y_label: self.y_val}))
            plt.title("Box Plot")
        elif viz_type == "heatmap":
            corr = pd.DataFrame({self.x_label: self.x_val, self.y_label: self.y_val}).corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
        else:
            raise ValueError(f"Unsupported viz_type: {viz_type}")

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()

    # ------------------------------------------------------------------
    # 2.8 Polynomial regression
    # ------------------------------------------------------------------
    def polynomial_regression(self, degree: int = 2):
        x = np.array(self.x_val)
        y = np.array(self.y_val)
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        r2 = r2_score(y, y_pred)

        raw = f"Coefficients (degree {degree}): {coeffs}\nR²: {r2:.4f}"
        self.summary = self._gemini_insight("polynomial regression", raw)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, color="steelblue")
        plt.plot(np.sort(x), poly(np.sort(x)), color="red", label=f"Degree {degree} fit")
        plt.title("Polynomial Regression")
        plt.legend()
        plt.show()

        print("\n=== Polynomial Regression Summary ===")
        print(self.summary)
        return raw

    # ------------------------------------------------------------------
    # 2.9 Generalized Additive Model (GAM) – using B-splines
    # ------------------------------------------------------------------
    def gam(self, spline_df: int = 10):
        x = np.array(self.x_val).reshape(-1, 1)
        y = np.array(self.y_val)

        # B-spline basis
        bs = BSplines(x, df=[spline_df], degree=[3])
        # Fit GAM (GLM with Gaussian family)
        gam_model = GLMGam(y, sm.add_constant(x), smoother=bs).fit()

        # Predict on original points
        y_pred = gam_model.predict(sm.add_constant(x), exog_smooth=x)

        raw = (
            f"Intercept: {gam_model.params[0]:.4f}\n"
            f"R² (pseudo): {gam_model.pseudo_rsquared():.4f}\n"
            f"Deviance: {gam_model.deviance:.2f}"
        )
        self.insights = self._gemini_insight("GAM", raw)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.x_val, y=self.y_val, color="steelblue")
        plt.plot(self.x_val, y_pred, color="red", label="GAM fit")
        plt.title("Generalized Additive Model (GAM)")
        plt.legend()
        plt.show()

        print("\n=== GAM Insights ===")
        print(self.insights)
        return raw

    # ------------------------------------------------------------------
    # 2.10 Simple MARS-style piecewise-linear approximation
    # ------------------------------------------------------------------
    def mars_approx(self, knots: Optional[List[float]] = None):
        """
        Very lightweight MARS approximation:
        - Sort data by x
        - Choose knots (default: quartiles)
        - Fit a separate linear segment between each pair of knots
        """
        x = np.sort(self.x_val)
        y = np.array(self.y_val)[np.argsort(self.x_val)]

        if knots is None:
            knots = np.percentile(x, [25, 50, 75])

        # Ensure knots are sorted and within data range
        knots = sorted([k for k in knots if x.min() <= k <= x.max()])

        # Build segments
        segments = []
        prev = x.min()
        for knot in knots:
            mask = (x >= prev) & (x <= knot)
            if mask.sum() > 1:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                segments.append((slope, intercept, prev, knot))
            prev = knot
        # last segment to end of data
        mask = x > prev
        if mask.sum() > 1:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            segments.append((slope, intercept, prev, x.max()))

        # Predict
        y_pred = np.zeros_like(x)
        for slope, intercept, start, end in segments:
            seg_mask = (x >= start) & (x <= end)
            y_pred[seg_mask] = slope * x[seg_mask] + intercept

        r2 = r2_score(y, y_pred)

        raw = f"Knots: {knots}\nSegments: {segments}\nR²: {r2:.4f}"
        self.summary = self._gemini_insight("MARS approximation", raw)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.x_val, y=self.y_val, color="steelblue")
        plt.plot(x, y_pred, color="red", label="Piecewise-linear fit")
        plt.title("Simple MARS-style Approximation")
        plt.legend()
        plt.show()

        print("\n=== MARS Approximation Summary ===")
        print(self.summary)
        return raw


# ----------------------------------------------------------------------
# 3. Example usage (uncomment & adapt to your data)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Replace the path / column names with your actual CSV file
    # ------------------------------------------------------------------
    data_path = "data.csv"          # <-- put your CSV file here
    x_col_name = "time"             # column that will be X
    y_col_name = "value"            # column that will be Y

    # Load data
    data = GraphData.from_csv(
        file_path=data_path,
        x_col=x_col_name,
        y_col=y_col_name,
        graph_type="line",
    )

    # ------------------------------------------------------------------
    # Run the various analyses (feel free to comment out any you don't need)
    # ------------------------------------------------------------------
    data.descriptive_stats()
    data.inferential_stats()
    data.time_series_analysis(period=7, forecast_steps=5)
    data.clustering_classification(n_clusters=3, classify=False)
    data.visualise(viz_type="scatter")
    data.visualise(viz_type="heatmap")
    data.polynomial_regression(degree=2)
    data.gam(spline_df=10)
    data.mars_approx()   # uses quartiles as knots by default