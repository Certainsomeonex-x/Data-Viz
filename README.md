<div align="center">

# 📊 Data-Viz: AI-Powered Data Visualization Studio

<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
<img src="https://img.shields.io/badge/AI-Gemini%202.5%20Flash-purple.svg" alt="AI Powered">
<img src="https://img.shields.io/badge/UI-CustomTkinter-orange.svg" alt="UI Framework">
<img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">

### *Transform Natural Language into Stunning Data Visualizations* ✨

[🚀 Features](#-features) • [📥 Installation](#-installation) • [🎯 Usage](#-usage) • [🧠 How It Works](#-how-it-works) • [👥 Contributors](#-contributors)

---

</div>

## 🌟 What is Data-Viz?

**Data-Viz** is an advanced Python-based data analysis and visualization tool that revolutionizes how you interact with data. Simply describe your data in plain English, and watch as AI generates comprehensive statistical analyses and beautiful visualizations!

No more wrestling with CSV files or struggling with complex data formatting—just tell the AI what you want to analyze, and it handles the rest.

### 🤖 Powered by Google Gemini 2.5 Flash

**Data-Viz leverages the cutting-edge Google Gemini 2.5 Flash API** for intelligent natural language understanding and structured data generation. This powerful AI model:

- 🚀 **Lightning-Fast Response**: Optimized for speed without compromising quality
- 🧠 **Smart Data Understanding**: Interprets complex data requests in natural language
- 📊 **Structured Output**: Generates properly formatted datasets with labels
- 💡 **Context-Aware**: Understands statistical context and data relationships
- 🎯 **High Accuracy**: Produces realistic and meaningful data distributions

The Gemini 2.5 Flash model is specifically chosen for its balance of speed, cost-efficiency, and intelligence—perfect for real-time data generation and analysis workflows.

### 💡 The Big Idea

We believe data analysis should be accessible to everyone, not just data scientists. Data-Viz bridges the gap between human language and data science by:

- 🤖 **AI-Driven Data Generation**: Using Google's **Gemini 2.5 Flash API** to understand your prompts
- 📊 **Intelligent Visualization**: Automatically selecting the best chart type for your analysis
- 🎨 **Beautiful UI**: Modern CustomTkinter interface that's both powerful and intuitive
- 🔬 **Advanced Analytics**: From basic statistics to ARIMA forecasting and GAM modeling

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 📈 Analysis Types

- **Descriptive Statistics** - Understand your data distribution
- **Inferential Statistics** - Make data-driven predictions
- **Time-Series Analysis** - ARIMA forecasting with moving averages
- **Clustering/Classification** - KMeans clustering with visual separation
- **Polynomial Regression** - Curve fitting up to degree 2
- **GAM (Generalized Additive Models)** - Flexible non-linear modeling
- **MARS Approximation** - Piecewise linear regression
- **Custom Visualizations** - Scatter, Bar, Histogram, Box, Heatmap

</td>
<td width="50%">

### 🎯 Key Capabilities

- ✅ Natural language prompt input
- ✅ AI-powered data generation
- ✅ Automatic data type detection (numeric, categorical, dates)
- ✅ Smart axis labeling (never lose your original data labels!)
- ✅ Real-time plot rendering
- ✅ Categorical & temporal data support
- ✅ Error handling & validation
- ✅ Beautiful matplotlib visualizations
- ✅ Threading for responsive UI
- ✅ Dark mode UI with modern aesthetics

</td>
</tr>
</table>

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** (3.9 or 3.10 recommended)
- **pip** package manager
- **Git** (for cloning the repository)
- **Google Gemini API Key** ([Get one free here](https://makersuite.google.com/app/apikey))

### Step 1: Clone the Repository

```bash
git clone https://github.com/Certainsomeonex-x/Data-Viz.git
cd Data-Viz
```

### Step 2: Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages include:**
- `customtkinter` - Modern UI framework
- `matplotlib` - Plotting library
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning
- `statsmodels` - Statistical modeling
- `google-generativeai` - Gemini AI SDK
- `pydantic` - Data validation

### Step 4: Configure API Key

Create a `.env` file in the project root:

```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual Gemini API key.

### Step 5: Launch the Application! 🎉

```bash
python dataviz_ui_customtk.py
```

---

## 🎯 Usage

### Quick Start Guide

1. **Launch the App** - Run `python dataviz_ui_customtk.py`

2. **Select Analysis Type** - Choose from the dropdown:
   - Descriptive statistics
   - Inferential statistics
   - Time-series analysis
   - Clustering/classification
   - Custom visualisation
   - Polynomial regression
   - GAM
   - MARS approximation

3. **Enter Your Prompt** - Describe your data in plain English:
   ```
   "Create a dataset of monthly sales from January to December 
   with values ranging from 1000 to 5000"
   ```

4. **Click "Run Analysis"** - Watch as the AI:
   - Generates your data
   - Performs the selected analysis
   - Creates a beautiful visualization

5. **View Results** - See your plot and any statistical output!

### 💡 Example Prompts

<details>
<summary><b>📊 Time-Series Analysis</b></summary>

```
"Generate monthly temperature data for 2024 starting from 15°C in January,
peaking at 35°C in July, and returning to 18°C in December"
```

**Result**: ARIMA forecast with moving average overlay, proper date formatting on x-axis.

</details>

<details>
<summary><b>🎯 Clustering Analysis</b></summary>

```
"Create a dataset with 3 distinct customer groups based on age (20-60) 
and spending (100-1000), with 50 samples"
```

**Result**: KMeans clustering with 3 color-coded clusters and proper labels.

</details>

<details>
<summary><b>📈 Polynomial Regression</b></summary>

```
"Generate data showing the relationship between study hours (1-10) 
and exam scores (40-95) for 30 students"
```

**Result**: Scatter plot with degree-2 polynomial fit curve.

</details>

<details>
<summary><b>🔍 Categorical Data</b></summary>

```
"Compare average salaries across departments: Engineering $85k, 
Marketing $65k, Sales $70k, HR $60k"
```

**Result**: Bar chart with proper categorical labels (not encoded numbers!).

</details>

### 🚫 Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `API Key not found` | Missing `.env` file | Create `.env` with `GEMINI_API_KEY=...` |
| `Module not found` | Missing dependencies | Run `pip install -r requirements.txt` |
| `Plot not showing` | Data generation failed | Check your prompt clarity and API connectivity |
| `Labels showing numbers` | *Fixed in latest version!* | Update to latest commit |
| `UI not responsive` | Background task running | Wait for "Run Analysis" to complete |

---

## 🧠 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (CustomTkinter)            │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Analysis │  │   Prompt  │  │   Run    │  │   Plot    │  │
│  │ Selector │  │  TextBox  │  │  Button  │  │   Frame   │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Background Thread Manager   │
         │  (Non-blocking execution)    │
         └──────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
┌────────────────┐         ┌────────────────────┐
│  Gemini 2.5    │         │  Analysis Engine   │
│  Flash API     │────────▶│  (Datavizmain.py)  │
│  (AI Data Gen) │         │                    │
└────────────────┘         └─────────┬──────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         ┌──────────────────┐            ┌──────────────────┐
         │  Data Processing │            │  Visualization   │
         │  - Type detection│            │  - matplotlib    │
         │  - Validation    │            │  - Smart labels  │
         │  - Encoding      │            │  - Date parsing  │
         └──────────────────┘            └──────────────────┘
```

### The Magic Behind the Scenes 🎩✨

1. **Prompt Engineering** - Your natural language prompt is sent to **Google Gemini 2.5 Flash API**
2. **AI Generation** - Gemini 2.5 Flash interprets your intent and generates structured data (x_values, y_values, labels)
3. **Data Validation** - Pydantic models ensure data integrity and type safety
4. **Type Detection** - Automatic detection of numeric, categorical, or temporal data
5. **Analysis Execution** - Selected statistical method runs on the validated data
6. **Smart Labeling** - Original labels preserved (not replaced with encoded numbers!)
7. **Plot Rendering** - matplotlib creates beautiful, properly-labeled visualizations

> **🔥 Why Gemini 2.5 Flash?**
> 
> We chose Gemini 2.5 Flash for its exceptional balance of:
> - ⚡ **Speed**: Near-instant responses for real-time data generation
> - 🎯 **Accuracy**: High-quality structured output with minimal hallucinations
> - 💰 **Cost-Efficiency**: Free tier available with generous quotas
> - 🧠 **Intelligence**: Advanced natural language understanding
> - 🔄 **Reliability**: Stable API with high uptime

### 🔧 Technical Process Flow

#### For Categorical Data:
```python
Original: ["Red", "Blue", "Green", "Red", "Blue"]
         ↓
Label Encoding: [0, 1, 2, 0, 1]  # For computation
         ↓
Plotting: Uses indices [0, 1, 2, 3, 4]
         ↓
Display: set_xticklabels(["Red", "Blue", "Green", "Red", "Blue"])
         ↓
Result: You see your original labels! ✅
```

#### For Date Data:
```python
Input: ["2024-01-01", "2024-02-01", "2024-03-01"]
         ↓
Parsing: pd.to_datetime() → datetime objects
         ↓
Plotting: Direct plotting with date objects
         ↓
Formatting: ConciseDateFormatter for clean labels
         ↓
Result: Beautiful "Jan", "Feb", "Mar" labels! ✅
```

---

## 📚 Project Facts

### 🎓 Educational Value

- **Learn by Doing**: See how AI interprets data requests
- **Statistical Literacy**: Understand when to use each analysis type
- **Practical Application**: Real-world data visualization techniques
- **Code Quality**: Clean, well-documented Python code to learn from

### 🔬 Mathematical Concepts Implemented

- **ARIMA**: AutoRegressive Integrated Moving Average for time-series forecasting
- **GAM**: Generalized Additive Models with B-Splines for flexible curve fitting
- **MARS**: Multivariate Adaptive Regression Splines for piecewise approximation
- **KMeans**: Unsupervised clustering with configurable k-values
- **Polynomial Regression**: Least-squares fitting with numpy polyfit

### 📊 Supported Data Types

| Type | Examples | Handling |
|------|----------|----------|
| **Numeric** | `1, 2, 3, 4.5` | Direct plotting |
| **Categorical** | `"A", "B", "C"` | Label encoding + smart labels |
| **Dates** | `"2024-01-01"` | pd.to_datetime + date formatting |
| **Mixed** | Auto-detected | Intelligent type inference |

### 🛡️ Error Prevention Features

- ✅ API key validation on startup
- ✅ Data validation with Pydantic models
- ✅ Try-except blocks for all analysis types
- ✅ User-friendly error messages
- ✅ Thread safety for UI operations
- ✅ Graceful degradation on failures

---

## 👥 Contributors

This project is the result of collaborative effort and inspiration from multiple brilliant minds:

<table>
<tr>
<td align="center" width="33%">
<img src="https://github.com/Certainsomeonex-x.png" width="100px;" alt="Anirudha Patnaik"/><br />
<b>Anirudha Patnaik</b><br />
<a href="https://github.com/Certainsomeonex-x">@Certainsomeonex-x</a><br />
<sub>Project Creator & Lead Developer</sub><br />
<sub>🎨 UI Design • 🔧 Core Architecture</sub>
</td>
<td align="center" width="33%">
<img src="https://github.com/soumy0dev.png" width="100px;" alt="Soumyadip Das"/><br />
<b>Soumyadip Das</b><br />
<a href="https://github.com/soumy0dev">@soumy0dev</a><br />
<sub>Core Contributor</sub><br />
<sub>📊 Statistical Methods • 🧮 Mathematical Logic</sub>
</td>
<td align="center" width="33%">
<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="100px;" alt="GitHub Copilot"/><br />
<b>GitHub Copilot</b><br />
<sub>AI Assistant</sub><br />
<sub>🤖 UI Implementation • 🐛 Bug Fixes</sub><br />
<sub>✨ Code Optimization</sub>
</td>
</tr>
</table>

### 🌟 Special Acknowledgments

- **Google Gemini Team** - For the groundbreaking **Gemini 2.5 Flash API** that powers the intelligent natural language data generation. This project wouldn't be possible without their cutting-edge AI technology and generous free tier access.
- **Soumyadip Das** - For providing the foundational mathematical concepts and statistical analysis algorithms that power the core analysis engine
- **Anirudha Patnaik** - For the project vision, Gemini AI integration, and overall architecture
- **GitHub Copilot** - For assistance in implementing the CustomTkinter UI, fixing axis labeling issues, and providing code suggestions throughout development
- **Open Source Community** - For the amazing libraries (matplotlib, pandas, scikit-learn, statsmodels) that make this project possible

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What this means:
✅ You can use this project commercially  
✅ You can modify the code  
✅ You can distribute your changes  
✅ You can use it privately  
❗ You must include the original license  
❗ No warranty or liability

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Ideas for Contributions:
- 🎨 Additional plot types (violin plots, 3D plots, etc.)
- 🔧 More analysis methods (SVM, Neural Networks, etc.)
- 🌍 Multi-language support
- 📱 Export functionality (save plots as images)
- 🎯 Preset prompt templates
- 📊 Data import from CSV/Excel
- 🔐 API key management UI

---

## 🐛 Known Issues & Roadmap

### Current Limitations:
- Requires internet connection for Gemini API
- Limited to matplotlib plotting (no interactive plots yet)
- No data persistence (can't save/load sessions)
- Single plot view (no plot comparison)

### Future Plans:
- [ ] Interactive plots with Plotly
- [ ] Data export functionality
- [ ] Session save/load
- [ ] Multi-plot comparison view
- [ ] Built-in prompt templates
- [ ] CSV/Excel file import
- [ ] Offline mode with sample datasets
- [ ] More statistical tests (t-tests, ANOVA, etc.)

---

## 💬 Support & Contact

Having issues? Want to share feedback?

- 📧 Open an [Issue](https://github.com/Certainsomeonex-x/Data-Viz/issues)
- 💬 Start a [Discussion](https://github.com/Certainsomeonex-x/Data-Viz/discussions)
- ⭐ Star this repo if you find it useful!

---

<div align="center">

### 🌟 If you found this project helpful, please consider giving it a star! ⭐

**Made with ❤️ by the Data-Viz Team**

[⬆ Back to Top](#-data-viz-ai-powered-data-visualization-studio)

</div>
