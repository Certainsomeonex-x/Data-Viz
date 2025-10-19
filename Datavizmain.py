from dotenv import load_dotenv
import os
import json
from typing import List, Optional
import google.generativeai as genai
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score
from pydantic import BaseModel

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)
# Use the Gemini 2.5 flash model for less quota-intensive usage
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- GraphData model with all features ---
class GraphData(BaseModel):
    def run_full_analysis(self, ts_start_date: str = "2023-01-01", classify: bool = False, use_gemini: bool = True):
        self.all_insights = {}
        # Collect all stats and plot descriptions
        try:
            desc_stats = self.descriptive_stats()
            desc_plot_desc = "Histogram and box plot show the distribution and spread of the data."
        except Exception as e:
            desc_stats = f"Descriptive stats error: {e}"
            desc_plot_desc = ""
        try:
            inf_stats = self.inferential_stats()
            inf_plot_desc = "Scatter plot visualizes the relationship between X and Y."
        except Exception as e:
            inf_stats = f"Inferential stats error: {e}"
            inf_plot_desc = ""
        try:
            ts_stats = self.time_series_analysis(start_date=ts_start_date)
            ts_plot_desc = "Line plot shows original series, moving average, and ARIMA forecast."
        except Exception as e:
            ts_stats = f"Time-series analysis error: {e}"
            ts_plot_desc = ""
        try:
            clust_stats = self.clustering_classification(classify=classify)
            clust_plot_desc = "Scatter plot with color-coded clusters from K-Means."
        except Exception as e:
            clust_stats = f"Clustering/classification error: {e}"
            clust_plot_desc = ""
        try:
            poly_stats = self.polynomial_regression()
            poly_plot_desc = "Scatter plot with polynomial regression curve."
        except Exception as e:
            poly_stats = f"Polynomial regression error: {e}"
            poly_plot_desc = ""
        try:
            gam_stats = self.gam()
            gam_plot_desc = "Scatter plot with smooth GAM fit line."
        except Exception as e:
            gam_stats = f"GAM error: {e}"
            gam_plot_desc = ""
        try:
            mars_stats = self.mars_approx()
            mars_plot_desc = "Scatter plot with piecewise-linear (MARS-style) fit."
        except Exception as e:
            mars_stats = f"MARS approximation error: {e}"
            mars_plot_desc = ""

        # Combine all results for a single Gemini call
        if use_gemini:
            combined_prompt = (
                f"You are an expert data analyst. Given the following results, provide a concise summary and 2-3 key insights.\n"
                f"Descriptive statistics: {desc_stats}\n{desc_plot_desc}\n"
                f"Inferential statistics: {inf_stats}\n{inf_plot_desc}\n"
                f"Time-series analysis: {ts_stats}\n{ts_plot_desc}\n"
                f"Clustering/classification: {clust_stats}\n{clust_plot_desc}\n"
                f"Polynomial regression: {poly_stats}\n{poly_plot_desc}\n"
                f"GAM: {gam_stats}\n{gam_plot_desc}\n"
                f"MARS approximation: {mars_stats}\n{mars_plot_desc}\n"
                f"Summarize the overall trends, relationships, and any anomalies."
            )
            try:
                response = gemini_model.generate_content(combined_prompt)
                summary = response.text.strip()
                print("\n=== Gemini Combined Summary/Insights ===")
                print(summary)
                self.all_insights['combined'] = summary
            except Exception as e:
                print(f"Gemini API error: {e}")
        else:
            print("\nGemini API calls are disabled. Only local analysis results are shown.")
            self.all_insights['combined'] = "Gemini API not used. See local analysis results above."

    x_val: List
    y_val: List[float]
    x_label: str
    y_label: str
    graph_type: str
    summary: str = ""
    insights: str = ""
    df: Optional[pd.DataFrame] = None
    all_insights: dict = {}

    model_config = {
        "arbitrary_types_allowed": True
    }

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        x_col: str,
        y_col: str,
        x_label: str = None,
        y_label: str = None,
        graph_type: str = "",
        summary: str = "",
        insights: str = "",
    ):
        df = pd.read_csv(file_path)
        # Always use the exact column names unless explicitly overridden
        x_label_final = x_col if x_label is None else x_label
        y_label_final = y_col if y_label is None else y_label
        return cls(
            x_val=df[x_col].tolist(),
            y_val=df[y_col].tolist(),
            x_label=x_label_final,
            y_label=y_label_final,
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
        x_label: str = None,
        y_label: str = None,
        graph_type: str = "",
        summary: str = "",
        insights: str = "",
    ):
        with open(file_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame({x_key: data[x_key], y_key: data[y_key]})
        # Always use the exact key names unless explicitly overridden
        x_label_final = x_key if x_label is None else x_label
        y_label_final = y_key if y_label is None else y_label
        return cls(
            x_val=data[x_key],
            y_val=data[y_key],
            x_label=x_label_final,
            y_label=y_label_final,
            graph_type=graph_type,
            summary=summary,
            insights=insights,
            df=df,
        )

    def _gemini_insight(self, analysis_type: str, raw_results: str, plot_desc: str = "") -> str:
        prompt = (
            f"Analyse the following {analysis_type} results for a dataset "
            f"with X = {self.x_label} and Y = {self.y_label}.\n"
            f"Statistical Results:\n{raw_results}\n"
            + (f"Graphical Analysis: {plot_desc}\n" if plot_desc else "")
            + "Provide a short, plain-English summary (≤ 2 sentences) and 1-2 key insights that consider both the statistics and the visual pattern."
        )
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    def descriptive_stats(self):
        # Handle categorical x_val by only describing y_val
        if all(isinstance(x, (int, float)) for x in self.x_val):
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
        else:
            # x_val is categorical, only describe y_val
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
        self.summary = self._gemini_insight("descriptive statistics", raw)
        print("\n=== Descriptive Statistics ===")
        print(self.summary)
        return raw

    def inferential_stats(self, confidence: float = 0.95, test: str = "t-test"):
        # Only perform inferential stats if x_val is numeric
        x = self._convert_months_to_numbers(self.x_val)
        y = self.y_val
        if all(isinstance(val, (int, float)) for val in x):
            if len(set(x)) < 2 or len(set(y)) < 2:
                raw = "Inferential statistics not applicable: all values are the same or not enough variance."
                self.insights = self._gemini_insight("inferential statistics", raw)
                print("\n=== Inferential Statistics ===")
                print(self.insights)
                return raw
            try:
                t_stat, p_val = stats.ttest_ind(x, y)
                t_res = f"T-stat: {t_stat:.4f}, p-value: {p_val:.4f}"
            except Exception as e:
                t_res = f"T-test error: {e}"
            try:
                ci_x = stats.norm.interval(confidence, loc=np.mean(x), scale=stats.sem(x))
                ci_y = stats.norm.interval(confidence, loc=np.mean(y), scale=stats.sem(y))
                ci_res = f"CI ({confidence*100:.0f}%): X={ci_x}, Y={ci_y}"
            except Exception as e:
                ci_res = f"CI error: {e}"
            try:
                corr, _ = stats.pearsonr(x, y)
                corr_res = f"Pearson correlation: {corr:.4f}"
            except Exception as e:
                corr_res = f"Pearson correlation error: {e}"
            raw = f"{t_res}\n{ci_res}\n{corr_res}"
        else:
            raw = "Inferential statistics not applicable for categorical X values."
        self.insights = self._gemini_insight("inferential statistics", raw)
        print("\n=== Inferential Statistics ===")
        print(self.insights)
        return raw

    def time_series_analysis(self, period: int = 7, forecast_steps: int = 5, start_date: str = "2023-01-01"):
        # Only perform time series analysis if x_val is numeric
        x = self._convert_months_to_numbers(self.x_val)
        if all(isinstance(val, (int, float)) for val in x):
            try:
                dates = pd.date_range(start=start_date, periods=len(self.y_val), freq="D")
                ts = pd.Series(self.y_val, index=dates, name=self.y_label)
            except Exception as e:
                print(f"Date range error: {e}")
                ts = pd.Series(self.y_val, name=self.y_label)
            ma = ts.rolling(window=period).mean()
            arima = ARIMA(ts, order=(1, 1, 1)).fit()
            forecast = arima.forecast(steps=forecast_steps)
            # Check for enough data points for seasonal_decompose
            if len(ts) >= 2 * period:
                decomp = seasonal_decompose(ts, model="additive", period=period)
                decomp_trend = decomp.trend[-5:].to_dict()
            else:
                print(f"[Warning] Not enough data points for seasonal decomposition (need {2*period}, got {len(ts)}). Skipping decomposition.")
                decomp_trend = "Not enough data for decomposition"

            raw = (
                f"Moving average (last 5): {ma[-5:].to_dict()}\n"
                f"ARIMA forecast: {forecast.to_dict()}\n"
                f"Decomposition trend (last 5): {decomp_trend}"
            )
            self.summary = self._gemini_insight("time-series analysis", raw)

            plt.figure(figsize=(12, 6))
            plt.plot(ts, label="Original")
            plt.plot(ma, label="Moving Avg")
            plt.plot(forecast, label="ARIMA Forecast")
            plt.title("Time-Series Analysis")
            plt.legend()
            plt.show()
            print("[Info] Plot displayed for visualisation.")
            print("\n=== Time-Series Summary ===")
            print(self.summary)
            return raw
        else:
            raw = "Time-series analysis not applicable for categorical X values."
            print("\n=== Time-Series Summary ===")
            print(raw)
            return raw

    def clustering_classification(self, n_clusters: int = 3, classify: bool = False):
        # Only perform clustering if x_val is numeric
        x = self._convert_months_to_numbers(self.x_val)
        if all(isinstance(val, (int, float)) for val in x):
            data = np.column_stack((x, self.y_val))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
            labels = kmeans.labels_

            raw = f"K-Means centroids:\n{kmeans.cluster_centers_}\nSample labels: {labels[:10]}"

            if classify:
                tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(data, labels)
                raw += "\nDecision tree fitted (use plot_tree(tree) to visualise)."

            self.insights = self._gemini_insight("clustering/classification", raw)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.x_val, y=self.y_val, hue=labels, palette="viridis")
            plt.title("K-Means Clustering")
            plt.show()

            print("\n=== Clustering Insights ===")
            print(self.insights)
            return raw
        else:
            raw = "Clustering/classification not applicable for categorical X values."
            print("\n=== Clustering Insights ===")
            print(raw)
            return raw

    def visualise(self, viz_type: str = "scatter"):
        sns.set_theme(style="darkgrid", palette="mako")
        plt.figure(figsize=(10, 6))
        if viz_type == "histogram":
            sns.histplot(self.y_val, kde=True, color="#0099cc")
            plt.title("Histogram", fontsize=16)
        elif viz_type == "bar":
            sns.barplot(x=self.x_val[:10], y=self.y_val[:10], palette="crest")
            plt.title("Bar Chart (first 10)", fontsize=16)
        elif viz_type == "scatter":
            if all(isinstance(x, (int, float)) for x in self.x_val):
                sns.scatterplot(x=self.x_val, y=self.y_val, color="#00bfae", s=80, edgecolor="black")
                plt.title("Scatter Plot", fontsize=16)
            else:
                x_jitter = np.arange(len(self.x_val)) + np.random.uniform(-0.2, 0.2, len(self.x_val))
                sns.scatterplot(x=x_jitter, y=self.y_val, hue=self.x_val, palette="mako", s=80, edgecolor="black")
                plt.xticks(np.arange(len(self.x_val)), self.x_val, rotation=30)
                plt.title("Scatter Plot (categorical X with jitter)", fontsize=16)
        elif viz_type == "box":
            if all(isinstance(x, (int, float)) for x in self.x_val):
                sns.boxplot(data=pd.DataFrame({self.x_label: self.x_val, self.y_label: self.y_val}), palette="rocket")
            else:
                sns.boxplot(y=self.y_val, palette="rocket")
            plt.title("Box Plot", fontsize=16)
        elif viz_type == "heatmap":
            if all(isinstance(x, (int, float)) for x in self.x_val):
                df_corr = pd.DataFrame({self.x_label: self.x_val, self.y_label: self.y_val})
                corr = df_corr.corr()
                if corr.isnull().values.any() or (corr.nunique().max() == 1):
                    plt.text(0.5, 0.5, "Correlation not meaningful (all values same)", ha='center', va='center', fontsize=14)
                    plt.title("Correlation Heatmap", fontsize=16)
                else:
                    sns.heatmap(corr, annot=True, cmap="mako", linewidths=1, linecolor="white")
                    plt.title("Correlation Heatmap", fontsize=16)
            else:
                df = pd.DataFrame({self.x_label: self.x_val, self.y_label: self.y_val})
                pivot = pd.pivot_table(df, index=self.x_label, values=self.y_label, aggfunc='count')
                if pivot.nunique().max() == 1:
                    plt.text(0.5, 0.5, "Frequency not meaningful (all values same)", ha='center', va='center', fontsize=14)
                    plt.title("Frequency Heatmap (categorical X)", fontsize=16)
                else:
                    sns.heatmap(pivot, annot=True, cmap="crest", linewidths=1, linecolor="white")
                    plt.title("Frequency Heatmap (categorical X)", fontsize=16)
        else:
            raise ValueError(f"Unsupported viz_type: {viz_type}")
        plt.xlabel(self.x_label, fontsize=13)
        plt.ylabel(self.y_label, fontsize=13)
        plt.tight_layout()
        plt.show()

    def polynomial_regression(self, degree: int = 2):
        x = self._convert_months_to_numbers(self.x_val)
        if not all(isinstance(val, (int, float)) for val in x):
            print("[Warning] Polynomial regression requires numeric x values. Could not convert all months.")
            return "Polynomial regression not applicable for categorical X values."
        y = np.array(self.y_val)
        x = np.array(x)
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        r2 = r2_score(y, y_pred)

        raw = f"Coefficients (degree {degree}): {coeffs}\nR²: {r2:.4f}"
        self.summary = self._gemini_insight("polynomial regression", raw)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, color="steelblue")
        plt.plot(np.sort(x), poly(np.sort(x)), color="red", label=f"Degree {degree} fit")
        plt.title("Polynomial Regression")
        plt.legend()
        plt.show()

        print("\n=== Polynomial Regression Summary ===")
        print(self.summary)
        return raw

    def gam(self, spline_df: int = 10):
        x = self._convert_months_to_numbers(self.x_val)
        if not all(isinstance(val, (int, float)) for val in x):
            print("[Warning] GAM requires numeric x values. Could not convert all months.")
            return "GAM not applicable for categorical X values."
        x = np.array(x).reshape(-1, 1)
        y = np.array(self.y_val)
        try:
            bs = BSplines(x, df=[spline_df], degree=[3])
            gam_model = GLMGam(y, sm.add_constant(x), smoother=bs).fit()
            y_pred = gam_model.predict(sm.add_constant(x), exog_smooth=x)

            raw = (
                f"Intercept: {gam_model.params[0]:.4f}\n"
                f"R² (pseudo): {gam_model.pseudo_rsquared():.4f}\n"
                f"Deviance: {gam_model.deviance:.2f}"
            )
            self.insights = self._gemini_insight("GAM", raw)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.x_val, y=self.y_val, color="steelblue")
            plt.plot(self.x_val, y_pred, color="red", label="GAM fit")
            plt.title("Generalized Additive Model (GAM)")
            plt.legend()
            plt.show()

            print("\n=== GAM Insights ===")
            print(self.insights)
            return raw
        except Exception as e:
            print(f"[Warning] GAM analysis failed: {e}")
            return f"GAM analysis failed: {e}"

    def mars_approx(self, knots: Optional[List[float]] = None):
        x = self._convert_months_to_numbers(self.x_val)
        if not all(isinstance(val, (int, float)) for val in x):
            print("[Warning] MARS approximation requires numeric x values. Could not convert all months.")
            return "MARS approximation not applicable for categorical X values."
        x = np.sort(np.array(x))
        y = np.array(self.y_val)[np.argsort(x)]
        if knots is None:
            knots = np.percentile(x, [25, 50, 75])
        knots = sorted([k for k in knots if x.min() <= k <= x.max()])
        segments = []
        prev = x.min()
        for knot in knots:
            mask = (x >= prev) & (x <= knot)
            if mask.sum() > 1:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                segments.append((slope, intercept, prev, knot))
            prev = knot
        mask = x > prev
        if mask.sum() > 1:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            segments.append((slope, intercept, prev, x.max()))
        y_pred = np.zeros_like(x)
        for slope, intercept, start, end in segments:
            seg_mask = (x >= start) & (x <= end)
            y_pred[seg_mask] = slope * x[seg_mask] + intercept
        r2 = r2_score(y, y_pred)
        raw = f"Knots: {knots}\nSegments: {segments}\nR²: {r2:.4f}"
        self.summary = self._gemini_insight("MARS approximation", raw)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.x_val, y=self.y_val, color="steelblue")
        plt.plot(x, y_pred, color="red", label="Piecewise-linear fit")
        plt.title("Simple MARS-style Approximation")
        plt.legend()
        plt.show()
        print("\n=== MARS Approximation Summary ===")
        print(self.summary)
        return raw

    def _convert_months_to_numbers(self, x_list):
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        converted = [month_map.get(str(x), x) for x in x_list]
        if all(isinstance(x, int) for x in converted):
            return converted
        return x_list

# --- Example usage ---
if __name__ == "__main__":
    print("Choose input mode:")
    print("1. CSV file\n2. JSON file\n3. Natural language prompt (Gemini)")
    print("Enter 1, 2, or 3: [Defaulting to 3]")
    mode = input().strip()
    if mode not in ["1", "2", "3"]:
        mode = "3"

    data = None
    if mode == "1":
        file_path = input("Enter CSV file path: ").strip()
        df = pd.read_csv(file_path)
        x_col = input(f"Enter column name for x values (options: {list(df.columns)}): ").strip()
        y_col = input(f"Enter column name for y values (options: {list(df.columns)}): ").strip()
        data = GraphData(
            x_val=df[x_col].tolist(),
            y_val=df[y_col].tolist(),
            x_label=x_col,
            y_label=y_col,
            graph_type="scatter"
        )
    elif mode == "2":
        file_path = input("Enter JSON file path: ").strip()
        with open(file_path, "r") as f:
            data_dict = json.load(f)
        data = GraphData(
            x_val=data_dict["x_val"],
            y_val=data_dict["y_val"],
            x_label=data_dict["x_label"],
            y_label=data_dict["y_label"],
            graph_type=data_dict.get("graph_type", "scatter")
        )
    elif mode == "3":
        user_prompt = input("Enter your data request (e.g. 'Monthly sales for 2024'): ").strip()
        gemini_data_prompt = (
            f"Given the following user request, generate a Python dictionary with keys: 'x_val', 'y_val', 'x_label', 'y_label', 'graph_type'. "
            f"Each of 'x_val' and 'y_val' should be a list of at least 8 values. "
            f"Respond ONLY with a valid Python dictionary.\nRequest: {user_prompt}"
        )
        response = gemini_model.generate_content(gemini_data_prompt)
        try:
            resp_text = response.text.strip()
            if resp_text.startswith("```"):
                resp_text = resp_text.split("\n", 1)[-1] if "\n" in resp_text else resp_text
                resp_text = resp_text.strip('`')
            if resp_text.endswith("```"):
                resp_text = resp_text.rsplit("````", 1)[0]
            resp_text = resp_text.strip()
            data_dict = eval(resp_text, {"__builtins__": {}})
            data = GraphData(
                x_val=data_dict["x_val"],
                y_val=data_dict["y_val"],
                x_label=data_dict["x_label"],
                y_label=data_dict["y_label"],
                graph_type=data_dict.get("graph_type", "scatter")
            )
            print("[Debug] Data object created from Gemini prompt:", data)
        except Exception as e:
            print(f"Gemini could not generate valid data: {e}\nRaw response: {response.text}")
            exit(1)
    else:
        print("Invalid input mode.")
        exit(1)

    if data is not None:
        options = [
            ("Descriptive statistics", lambda: data.descriptive_stats()),
            ("Inferential statistics", lambda: data.inferential_stats()),
            ("Time-series analysis", lambda: data.time_series_analysis()),
            ("Clustering/classification", lambda: data.clustering_classification()),
            ("Custom visualisation", None),
            ("Polynomial regression", lambda: data.polynomial_regression()),
            ("GAM", lambda: data.gam()),
            ("MARS approximation", lambda: data.mars_approx()),
            ("Exit", None)
        ]

        viz_types = ["scatter", "histogram", "bar", "box", "heatmap"]
        selected_viz = "scatter"
        while True:
            print("\nSelect an option:")
            for idx, (label, _) in enumerate(options):
                print(f"  [{idx+1}] {label}")
            choice = input("Enter the number of the analysis/visualisation to run (or Exit): ").strip()
            try:
                choice_idx = int(choice) - 1
                if choice_idx < 0 or choice_idx >= len(options):
                    print("Invalid choice. Try again.")
                    continue
            except ValueError:
                print("Invalid input. Try again.")
                continue
            label, func = options[choice_idx]
            if label == "Exit":
                print("Exiting application.")
                break
            elif label == "Custom visualisation":
                print("Select graph type for visualisation:")
                for v_idx, vtype in enumerate(viz_types):
                    print(f"  [{v_idx+1}] {vtype}")
                v_choice = input("Enter the number of the graph type: ").strip()
                try:
                    v_choice_idx = int(v_choice) - 1
                    if v_choice_idx < 0 or v_choice_idx >= len(viz_types):
                        print("Invalid graph type. Defaulting to scatter.")
                        selected_viz = "scatter"
                    else:
                        selected_viz = viz_types[v_choice_idx]
                except ValueError:
                    print("Invalid input. Defaulting to scatter.")
                    selected_viz = "scatter"
                print(f"[User-selected: {selected_viz}]")
                data.visualise(viz_type=selected_viz)
            elif func is not None:
                print(f"[Running: {label}]")
                func()
    else:
        print("[Warning] No data object was created. Please check your input and try again.")
    # Remove old menu simulation and stray except block. Only keep new menu loop above.
