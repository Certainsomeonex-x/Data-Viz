import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import sys
import os

from Datavizmain import GraphData, gemini_model

class DataVizApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Data-Viz Gemini UI")
        self.geometry("1080x800")
        self.minsize(900, 650)
        self.configure(fg_color="#181a20")
        self.data = None
        self.create_widgets()

    def create_widgets(self):
        # Analysis selection (move above output for accessibility)
        analysis_frame = ctk.CTkFrame(self, fg_color="#23232f", corner_radius=18)
        analysis_frame.pack(fill="x", padx=40, pady=(8, 2))
        analysis_label = ctk.CTkLabel(analysis_frame, text="Analysis Type", font=("Inter", 15, "bold"), text_color="#f5f6fa")
        analysis_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.analysis_var = tk.StringVar(value="Descriptive statistics")
        self.analysis_options = [
            "Descriptive statistics",
            "Inferential statistics",
            "Time-series analysis",
            "Clustering/classification",
            "Custom visualisation",
            "Polynomial regression",
            "GAM",
            "MARS approximation"
        ]
        self.analysis_menu = ctk.CTkComboBox(analysis_frame, variable=self.analysis_var, values=self.analysis_options, font=("Inter", 13), dropdown_font=("Inter", 12), corner_radius=10)
        self.analysis_menu.pack(fill="x", padx=10, pady=(0, 4))
        # Custom viz type label and ComboBox
        viz_label = ctk.CTkLabel(analysis_frame, text="Graph Type (for Custom Visualisation)", font=("Inter", 13), text_color="#f5f6fa")
        viz_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.viz_type_var = ctk.StringVar(value="scatter")
        self.viz_types = ["scatter", "histogram", "bar", "box", "heatmap"]
        self.viz_menu = ctk.CTkOptionMenu(analysis_frame, variable=self.viz_type_var, values=self.viz_types, font=("Inter", 13), command=self.on_viz_type_change)
        self.viz_menu.pack(fill="x", padx=10, pady=(0, 4))
        self.viz_menu.set("scatter")
        self.viz_menu.configure(state="normal")
        self.analysis_menu.bind("<<ComboboxSelected>>", self.on_analysis_change)
        # Prompt entry (chatbox)
        prompt_frame = ctk.CTkFrame(self, fg_color="#23232f", corner_radius=18)
        prompt_frame.pack(fill="x", padx=40, pady=(8, 2))
        prompt_label = ctk.CTkLabel(prompt_frame, text="Enter your data prompt:", font=("Inter", 13), text_color="#f5f6fa")
        prompt_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.prompt_entry = ctk.CTkTextbox(prompt_frame, height=40, font=("Inter", 13), corner_radius=10)
        self.prompt_entry.pack(fill="x", padx=10, pady=(0, 4))
        # Output frame
        output_frame = ctk.CTkFrame(self, fg_color="#23232f", corner_radius=18)
        output_frame.pack(fill="both", expand=True, padx=40, pady=(8, 2))
        output_label = ctk.CTkLabel(output_frame, text="Output:", font=("Inter", 13), text_color="#f5f6fa")
        output_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.output_text = ctk.CTkTextbox(output_frame, height=120, font=("Inter", 12), corner_radius=10)
        self.output_text.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self.output_text.configure(state="disabled")
        # Plot frame
        self.plot_frame = ctk.CTkFrame(self, fg_color="#23232f", corner_radius=18)
        self.plot_frame.pack(fill="both", expand=True, padx=40, pady=(8, 2))
        # Run button
        run_btn = ctk.CTkButton(self, text="Run Analysis", font=("Inter", 14, "bold"), corner_radius=12, command=self.run_analysis_thread)
        run_btn.pack(pady=(8, 12))

    def on_viz_type_change(self, event=None):
        # Only update plot if Custom visualisation is selected AND data exists
        if self.analysis_var.get() == "Custom visualisation" and self.data is not None:
            self.show_plot()

    def on_analysis_change(self, event=None):
        # Enable graph type ComboBox only for Custom visualisation
        if self.analysis_var.get() == "Custom visualisation":
            self.viz_menu.configure(state="normal")
        else:
            self.viz_menu.configure(state="normal")
        # Don't clear or plot when changing analysis type - wait for Run Analysis button

    def run_analysis_thread(self):
        import threading
        threading.Thread(target=self._run_analysis_bg, daemon=True).start()

    def _run_analysis_bg(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            self.after(0, lambda: self.set_output("⚠️ **Please enter a data prompt.**"))
            return
        self.after(0, self.clear_output)
        self.after(0, lambda: self.set_output("⏳ _Generating data from Gemini..._\n"))
        try:
            gemini_data_prompt = (
                f"Given the following user request, generate a Python dictionary with keys: 'x_val', 'y_val', 'x_label', 'y_label', 'graph_type'. "
                f"Each of 'x_val' and 'y_val' should be a list of at least 8 values. "
                f"Respond ONLY with a valid Python dictionary.\nRequest: {prompt}"
            )
            response = gemini_model.generate_content(gemini_data_prompt)
            resp_text = response.text.strip()
            if resp_text.startswith("```"):
                resp_text = resp_text.split("\n", 1)[-1] if "\n" in resp_text else resp_text
                resp_text = resp_text.strip('`')
            if resp_text.endswith("````"):
                resp_text = resp_text.rsplit("````", 1)[0]
            resp_text = resp_text.strip()
            data_dict = eval(resp_text, {"__builtins__": {}})
            self.data = GraphData(
                x_val=data_dict["x_val"],
                y_val=data_dict["y_val"],
                x_label=data_dict["x_label"],
                y_label=data_dict["y_label"],
                graph_type=data_dict.get("graph_type", "scatter")
            )
            # Log the generated data for debugging
            self.after(0, lambda: self.set_output(f"✅ **Data generated:** {len(self.data.x_val)} data points\n**X ({self.data.x_label}):** {self.data.x_val[:3]}...\n**Y ({self.data.y_label}):** {self.data.y_val[:3]}...\n\n⏳ _Running analysis..._\n"))
        except Exception as e:
            self.after(0, lambda: self.set_output(f"❌ **Gemini could not generate valid data:** {e}"))
            return
        # Schedule analysis and plotting on the main thread
        self.after(0, self._run_analysis_main)

    def _run_analysis_main(self):
        analysis = self.analysis_var.get()
        try:
            if analysis == "Descriptive statistics":
                self.data.descriptive_stats()
            elif analysis == "Inferential statistics":
                self.data.inferential_stats()
            elif analysis == "Time-series analysis":
                self.data.time_series_analysis()
            elif analysis == "Clustering/classification":
                self.data.clustering_classification()
            elif analysis == "Custom visualisation":
                self.data.visualise(viz_type=self.viz_type_var.get())
            elif analysis == "Polynomial regression":
                self.data.polynomial_regression()
            elif analysis == "GAM":
                gam_result = self.data.gam()
                # If GAM returns an error string, show it with x values
                if isinstance(gam_result, str) and ("not applicable" in gam_result or "failed" in gam_result):
                    x_vals = self.data.x_val
                    self.set_output(f"❌ **GAM analysis error:** {gam_result}\n\n**X values used:** {x_vals}")
                    return
            elif analysis == "MARS approximation":
                self.data.mars_approx()
            else:
                self.set_output("❌ **Unknown analysis type.**")
                return
            summary = getattr(self.data, 'summary', '') or getattr(self.data, 'insights', '') or ''
        except Exception as e:
            # For GAM, show x values if error
            if analysis == "GAM":
                x_vals = self.data.x_val if hasattr(self.data, 'x_val') else None
                self.set_output(f"❌ **GAM analysis failed:** {e}\n\n**X values used:** {x_vals}")
            else:
                self.set_output(f"❌ **Analysis failed:** {e}")
            return
        if summary:
            self.set_output(f"### Result\n{summary}")
        else:
            self.set_output("✅ _Analysis complete, but no summary available._")
        self.show_plot()



    def clear_output(self):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state="disabled")
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

    def set_output(self, markdown_text):
        # Simple markdown to text rendering for clarity (headers, bold, italics, code)
        import re
        text = markdown_text
        text = re.sub(r"^### (.*)$", r"\1\n" + "="*40, text, flags=re.MULTILINE)  # h3
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # italics
        text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("end", text.strip() + "\n")
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def show_plot(self):
        # Don't plot if no data exists
        if self.data is None or not hasattr(self.data, 'y_val') or not hasattr(self.data, 'x_val'):
            return
        
        # Get plot_frame size and set a larger minimum
        self.plot_frame.update_idletasks()
        width = max(self.plot_frame.winfo_width(), 700)
        height = max(self.plot_frame.winfo_height(), 500)
        dpi = 100
        plt.close('all')  # Clear previous figures
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
        # Re-plot for all analysis types
        if hasattr(self.data, 'y_val') and hasattr(self.data, 'x_val'):
            analysis = self.analysis_var.get()
            import pandas as pd
            import numpy as np
            import seaborn as sns
            # Try to parse x_val as dates only if they look like dates
            used_x_as_time = False
            x_for_plot = self.data.x_val
            
            # Check if x_label suggests dates or if values look like dates
            date_keywords = ['date', 'time', 'year', 'month', 'day', 'period']
            x_label_lower = self.data.x_label.lower() if self.data.x_label else ""
            
            if any(keyword in x_label_lower for keyword in date_keywords) or analysis == "Time-series analysis":
                try:
                    parsed_dates = pd.to_datetime(self.data.x_val, errors='raise')
                    if len(parsed_dates) == len(self.data.y_val):
                        x_for_plot = parsed_dates
                        used_x_as_time = True
                except Exception:
                    # If parsing fails, keep original x_val
                    pass
            # Time-series analysis
            if analysis == "Time-series analysis":
                ts = pd.Series(self.data.y_val, index=x_for_plot, name=self.data.y_label)
                ma = ts.rolling(window=7).mean()
                ax.plot(ts, label="Original", linewidth=2)
                ax.plot(ma, label="Moving Avg", linewidth=2, linestyle='--')
                ax.set_xlabel(self.data.x_label if self.data.x_label else "Time")
                ax.set_ylabel(self.data.y_label if self.data.y_label else "Value")
                ax.set_title(f"Time-Series Analysis: {self.data.x_label} vs {self.data.y_label}")
                ax.legend()
                # Format date axis if dates
                if used_x_as_time:
                    import matplotlib.dates as mdates
                    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
                    fig.autofmt_xdate()
            # Descriptive statistics
            elif analysis == "Descriptive statistics":
                ax.hist(self.data.y_val, bins=10, color="#0099cc", edgecolor="black")
                ax.set_xlabel(self.data.y_label if self.data.y_label else "Value")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Histogram of {self.data.y_label}")
            # Inferential statistics
            elif analysis == "Inferential statistics":
                if used_x_as_time:
                    ax.scatter(x_for_plot, self.data.y_val, color="#00bfae", s=80, edgecolor="black")
                else:
                    # Use original x_val for non-date data
                    ax.scatter(range(len(self.data.x_val)), self.data.y_val, color="#00bfae", s=80, edgecolor="black")
                    ax.set_xticks(range(len(self.data.x_val)))
                    ax.set_xticklabels(self.data.x_val, rotation=45, ha='right')
                ax.set_xlabel(self.data.x_label if self.data.x_label else "X")
                ax.set_ylabel(self.data.y_label if self.data.y_label else "Y")
                ax.set_title(f"Scatter Plot: {self.data.x_label} vs {self.data.y_label}")
            # Clustering/classification
            elif analysis == "Clustering/classification":
                # Convert x_val to numeric for clustering
                try:
                    x_numeric = pd.to_numeric(self.data.x_val, errors='coerce')
                    if x_numeric.isna().any():
                        raise ValueError("Non-numeric x values")
                    x = np.array(x_numeric)
                except Exception:
                    # Label encode categorical x_val
                    unique_vals = list(dict.fromkeys(self.data.x_val))
                    label_map = {val: idx for idx, val in enumerate(unique_vals)}
                    x = np.array([label_map[val] for val in self.data.x_val])
                
                y = np.array(self.data.y_val)
                from sklearn.cluster import KMeans
                data = np.column_stack((x, y))
                kmeans = KMeans(n_clusters=3, random_state=42).fit(data)
                labels = kmeans.labels_
                
                if used_x_as_time:
                    sns.scatterplot(x=x_for_plot, y=y, hue=labels, palette="viridis", ax=ax, legend='full')
                else:
                    # Plot with indices and show original labels
                    positions = range(len(self.data.x_val))
                    scatter = ax.scatter(positions, y, c=labels, cmap='viridis', s=80, edgecolor="black")
                    ax.set_xticks(positions)
                    ax.set_xticklabels(self.data.x_val, rotation=45, ha='right')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                
                ax.set_xlabel(self.data.x_label if self.data.x_label else "X")
                ax.set_ylabel(self.data.y_label if self.data.y_label else "Y")
                ax.set_title(f"K-Means Clustering: {self.data.x_label} vs {self.data.y_label}")
            # Custom visualisation
            elif analysis == "Custom visualisation":
                viz_type = self.viz_type_var.get()
                if viz_type == "scatter":
                    if used_x_as_time:
                        sns.scatterplot(x=x_for_plot, y=self.data.y_val, color="#00bfae", s=80, edgecolor="black", ax=ax)
                    else:
                        positions = range(len(self.data.x_val))
                        ax.scatter(positions, self.data.y_val, color="#00bfae", s=80, edgecolor="black")
                        ax.set_xticks(positions)
                        ax.set_xticklabels(self.data.x_val, rotation=45, ha='right')
                    ax.set_xlabel(self.data.x_label if self.data.x_label else "X")
                    ax.set_ylabel(self.data.y_label if self.data.y_label else "Y")
                    ax.set_title(f"Scatter Plot: {self.data.x_label} vs {self.data.y_label}")
                elif viz_type == "histogram":
                    ax.hist(self.data.y_val, bins=10, color="#0099cc", edgecolor="black")
                    ax.set_xlabel(self.data.y_label if self.data.y_label else "Value")
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Histogram of {self.data.y_label}")
                elif viz_type == "bar":
                    x_positions = range(len(self.data.x_val))
                    ax.bar(x_positions, self.data.y_val, color="#0099cc")
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(self.data.x_val, rotation=45, ha='right')
                    ax.set_xlabel(self.data.x_label if self.data.x_label else "X")
                    ax.set_ylabel(self.data.y_label if self.data.y_label else "Y")
                    ax.set_title(f"Bar Chart: {self.data.x_label} vs {self.data.y_label}")
                elif viz_type == "box":
                    ax.boxplot(self.data.y_val)
                    ax.set_ylabel(self.data.y_label if self.data.y_label else "Value")
                    ax.set_title(f"Box Plot: {self.data.y_label}")
                elif viz_type == "heatmap":
                    try:
                        # Try to create numeric heatmap
                        df_corr = pd.DataFrame({self.data.x_label: pd.to_numeric(self.data.x_val, errors='coerce'), 
                                                 self.data.y_label: self.data.y_val})
                        corr = df_corr.corr()
                        sns.heatmap(corr, annot=True, cmap="mako", linewidths=1, linecolor="white", ax=ax)
                        ax.set_title(f"Correlation Heatmap: {self.data.x_label} vs {self.data.y_label}")
                    except Exception:
                        ax.text(0.5, 0.5, "Heatmap requires numeric data", ha='center', va='center')
                        ax.set_title("Heatmap - Invalid Data")
            # Polynomial regression
            elif analysis == "Polynomial regression":
                y = np.array(self.data.y_val)
                # Try to convert x_val to numeric, fallback to label encoding
                try:
                    x_numeric = pd.to_numeric(self.data.x_val, errors='coerce')
                    if x_numeric.isna().any():
                        raise ValueError("Non-numeric x values")
                    x_numeric = np.array(x_numeric)
                    is_categorical = False
                except Exception:
                    unique_vals = list(dict.fromkeys(self.data.x_val))
                    label_map = {val: idx for idx, val in enumerate(unique_vals)}
                    x_numeric = np.array([label_map[val] for val in self.data.x_val])
                    is_categorical = True
                
                try:
                    coeffs = np.polyfit(x_numeric, y, 2)
                    poly = np.poly1d(coeffs)
                    
                    if is_categorical or not used_x_as_time:
                        # Plot with indices and show original labels
                        positions = range(len(self.data.x_val))
                        ax.scatter(positions, y, color="steelblue", s=80, edgecolor="black")
                        x_curve = np.linspace(0, len(self.data.x_val)-1, 200)
                        y_curve = poly(x_curve)
                        ax.plot(x_curve, y_curve, color="red", linewidth=2, label="Degree 2 fit")
                        ax.set_xticks(positions)
                        ax.set_xticklabels(self.data.x_val, rotation=45, ha='right')
                    else:
                        ax.scatter(x_for_plot, y, color="steelblue", s=80, edgecolor="black")
                        x_curve = np.linspace(np.min(x_numeric), np.max(x_numeric), 200)
                        ax.plot(x_curve, poly(x_curve), color="red", linewidth=2, label="Degree 2 fit")
                    
                    ax.set_xlabel(self.data.x_label if self.data.x_label else "X")
                    ax.set_ylabel(self.data.y_label if self.data.y_label else "Y")
                    ax.set_title("Polynomial Regression")
                    ax.legend()
                except Exception as e:
                    self.set_output(f"❌ Polynomial regression plot failed: {e}\nX values: {self.data.x_val}")
                    return
            # GAM
            elif analysis == "GAM":
                y = np.array(self.data.y_val)
                from statsmodels.gam.api import GLMGam, BSplines
                import statsmodels.api as sm
                # Try to convert x_val to numeric, fallback to label encoding
                try:
                    x_numeric = pd.to_numeric(self.data.x_val, errors='coerce')
                    if x_numeric.isna().any():
                        raise ValueError("Non-numeric x values")
                    x_gam = np.array(x_numeric).reshape(-1, 1)
                    is_categorical = False
                except Exception:
                    unique_vals = list(dict.fromkeys(self.data.x_val))
                    label_map = {val: idx for idx, val in enumerate(unique_vals)}
                    x_gam = np.array([label_map[val] for val in self.data.x_val]).reshape(-1, 1)
                    is_categorical = True
                
                try:
                    bs = BSplines(x_gam, df=[10], degree=[3])
                    gam_model = GLMGam(y, sm.add_constant(x_gam), smoother=bs).fit()
                    y_pred = gam_model.predict(sm.add_constant(x_gam), exog_smooth=x_gam)
                    
                    if is_categorical or not used_x_as_time:
                        # Plot with indices and show original labels
                        positions = range(len(self.data.x_val))
                        ax.scatter(positions, y, color="steelblue", s=80, edgecolor="black")
                        # Sort for smooth curve
                        sorted_indices = np.argsort(x_gam.flatten())
                        ax.plot(sorted_indices, y_pred[sorted_indices], color="red", linewidth=2, label="GAM fit")
                        ax.set_xticks(positions)
                        ax.set_xticklabels(self.data.x_val, rotation=45, ha='right')
                    else:
                        ax.scatter(x_for_plot, y, color="steelblue", s=80, edgecolor="black")
                        ax.plot(x_for_plot, y_pred, color="red", linewidth=2, label="GAM fit")
                    
                    ax.set_xlabel(self.data.x_label if self.data.x_label else "X")
                    ax.set_ylabel(self.data.y_label if self.data.y_label else "Y")
                    ax.set_title("Generalized Additive Model (GAM)")
                    ax.legend()
                except Exception as e:
                    self.set_output(f"❌ GAM plot failed: {e}\nX values: {self.data.x_val}")
                    return
            # MARS approximation
            elif analysis == "MARS approximation":
                y = np.array(self.data.y_val)
                # Try to convert x_val to numeric, fallback to label encoding
                try:
                    x_numeric = pd.to_numeric(self.data.x_val, errors='coerce')
                    if x_numeric.isna().any():
                        raise ValueError("Non-numeric x values")
                    x_mars = np.array(x_numeric)
                    is_categorical = False
                except Exception:
                    unique_vals = list(dict.fromkeys(self.data.x_val))
                    label_map = {val: idx for idx, val in enumerate(unique_vals)}
                    x_mars = np.array([label_map[val] for val in self.data.x_val])
                    is_categorical = True
                
                try:
                    knots = np.percentile(x_mars, [25, 50, 75])
                    segments = []
                    prev = x_mars.min()
                    for knot in knots:
                        mask = (x_mars >= prev) & (x_mars <= knot)
                        if mask.sum() > 1:
                            slope, intercept = np.polyfit(x_mars[mask], y[mask], 1)
                            segments.append((slope, intercept, prev, knot))
                        prev = knot
                    mask = x_mars > prev
                    if mask.sum() > 1:
                        slope, intercept = np.polyfit(x_mars[mask], y[mask], 1)
                        segments.append((slope, intercept, prev, x_mars.max()))
                    
                    y_pred = np.zeros_like(x_mars)
                    for slope, intercept, start, end in segments:
                        seg_mask = (x_mars >= start) & (x_mars <= end)
                        y_pred[seg_mask] = slope * x_mars[seg_mask] + intercept
                    
                    if is_categorical or not used_x_as_time:
                        # Plot with indices and show original labels
                        positions = range(len(self.data.x_val))
                        ax.scatter(positions, y, color="steelblue", s=80, edgecolor="black")
                        ax.plot(positions, y_pred, color="red", linewidth=2, label="Piecewise-linear fit")
                        ax.set_xticks(positions)
                        ax.set_xticklabels(self.data.x_val, rotation=45, ha='right')
                    else:
                        ax.scatter(x_for_plot, y, color="steelblue", s=80, edgecolor="black")
                        ax.plot(x_for_plot, y_pred, color="red", linewidth=2, label="Piecewise-linear fit")
                    
                    ax.set_xlabel(self.data.x_label if self.data.x_label else "X")
                    ax.set_ylabel(self.data.y_label if self.data.y_label else "Y")
                    ax.set_title("Simple MARS-style Approximation")
                    ax.legend()
                except Exception as e:
                    self.set_output(f"❌ MARS plot failed: {e}\nX values: {self.data.x_val}")
                    return
            # Note: Each analysis type now handles its own axis labels and formatting
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = DataVizApp()
    app.mainloop()
