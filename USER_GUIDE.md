# User Guide for Data-Viz Application

## Table of Contents
1. [Getting Started](#getting-started)
2. [Interactive Mode](#interactive-mode)
3. [Programmatic Usage](#programmatic-usage)
4. [Examples](#examples)
5. [Tips and Best Practices](#tips-and-best-practices)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- pip (Python package manager)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Certainsomeonex-x/Data-Viz.git
   cd Data-Viz
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key**
   
   Create a `.env` file in the project directory:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Verify installation**
   ```bash
   python test_data_viz_app.py
   ```

## Interactive Mode

The interactive mode provides a user-friendly interface for creating and modifying visualizations.

### Starting Interactive Mode

```bash
python data_viz_app.py
```

### Main Menu Options

When you run the app, you'll see:
```
================================================================================
DATA VISUALIZATION APP WITH GEMINI AI
================================================================================

Options:
1. Create new visualization
2. Request changes/inference on current visualization
3. Exit

Enter your choice (1-3):
```

### Option 1: Create New Visualization

1. Select option `1`
2. Enter your problem statement (be specific and descriptive)
3. Optionally provide data in JSON format or press Enter to skip
4. Wait for Gemini to process your request
5. Review the summary, insights, and recommendations
6. Choose whether to generate and display the graph
7. Optionally save the graph to a file

**Example Flow:**
```
Enter your choice (1-3): 1

Enter your problem statement: Show monthly sales trends for the last 6 months

Enter data (JSON/CSV) or press Enter to skip: 
{"months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"], 
 "sales": [1000, 1200, 1100, 1500, 1800, 2000]}

Processing with Gemini AI...
✓ Configuration generated successfully!

[Summary displays here]

Generate and display graph? (y/n): y
Enter save path (or press Enter to skip saving): monthly_sales.png

✓ Graph saved to monthly_sales.png
```

### Option 2: Request Changes/Inference

After creating a visualization, you can:
- Modify the visualization style
- Add or remove data series
- Request additional analysis
- Ask questions about the data

**Example Flow:**
```
Enter your choice (1-3): 2

Describe the changes or inference you need: 
Add a trend line and predict next month's sales

Processing change request...
✓ Changes processed successfully!

[Updated summary displays here]

Regenerate graph with changes? (y/n): y
```

### Option 3: Exit

Select option `3` to exit the application.

## Programmatic Usage

For integration into other Python scripts or applications:

### Basic Usage

```python
from data_viz_app import DataVizApp

# Initialize the app
app = DataVizApp()

# Create a visualization
problem = "Analyze customer satisfaction across 5 categories"
config = app.process_prompt(problem)

# Display the summary
print(app.display_summary(config))

# Generate and save the graph
fig = app.generate_graph(config, save_path='satisfaction.png')
```

### With Custom Data

```python
from data_viz_app import DataVizApp
import json

# Initialize
app = DataVizApp()

# Prepare your data
data = json.dumps({
    "categories": ["Service", "Quality", "Price", "Support", "Delivery"],
    "ratings": [4.5, 4.8, 3.9, 4.2, 4.6]
})

# Create visualization
problem = "Show customer satisfaction ratings by category"
config = app.process_prompt(problem, data)

# Generate graph
fig = app.generate_graph(config)

# Display (for Jupyter notebooks or scripts with GUI)
import matplotlib.pyplot as plt
plt.show()
```

### Requesting Changes

```python
# After creating initial visualization
updated_config = app.request_changes(
    "Add comparison with industry average of 4.0"
)

# Generate updated graph
fig = app.generate_graph(updated_config)
```

## Examples

### Example 1: Sales Analysis

**Problem Statement:**
> "Show quarterly sales performance for 2024 across three product lines"

**Expected Output:**
- Multi-series bar chart
- Summary of sales trends
- Insights about best/worst performing products
- Recommendations for improvement

### Example 2: Scientific Data

**Problem Statement:**
> "Visualize the correlation between temperature and chemical reaction rate"

**Sample Data:**
```json
{
    "temperature": [20, 30, 40, 50, 60, 70],
    "rate": [0.5, 1.2, 2.8, 5.1, 8.3, 12.1]
}
```

**Expected Output:**
- Scatter plot with trend line
- Analysis of correlation strength
- Insights about optimal temperature range

### Example 3: Survey Results

**Problem Statement:**
> "Display employee engagement survey results across departments"

**Expected Output:**
- Grouped bar chart
- Department comparison analysis
- Key insights about engagement levels
- Recommendations for HR

### Example 4: Time Series

**Problem Statement:**
> "Show website traffic trends over the last 30 days with peak identification"

**Expected Output:**
- Line graph with markers
- Identification of peak traffic days
- Analysis of traffic patterns
- Recommendations for content scheduling

## Tips and Best Practices

### Writing Effective Problem Statements

1. **Be Specific**
   - ❌ "Show sales"
   - ✅ "Show monthly sales trends for Product A vs Product B over Q3 2024"

2. **Include Context**
   - ❌ "Graph this data"
   - ✅ "Compare customer retention rates across marketing channels to identify the most effective channel"

3. **Specify Desired Insights**
   - ❌ "Make a chart"
   - ✅ "Analyze conversion rates by traffic source and identify opportunities for improvement"

### Providing Data

1. **JSON Format** (Recommended)
   ```json
   {
       "labels": ["A", "B", "C"],
       "values": [10, 20, 15]
   }
   ```

2. **Multiple Series**
   ```json
   {
       "months": ["Jan", "Feb", "Mar"],
       "product_a": [100, 120, 110],
       "product_b": [80, 95, 100]
   }
   ```

3. **Let Gemini Generate Data**
   - When exploring concepts or creating examples
   - Simply skip data entry and describe what you need

### Graph Type Selection

Gemini automatically selects the appropriate graph type based on:
- Type of data (categorical, continuous, time series)
- Number of variables
- Desired analysis

Common selections:
- **Bar Chart**: Categorical comparisons
- **Line Graph**: Trends over time
- **Scatter Plot**: Correlations between variables
- **Pie Chart**: Composition/proportions
- **Histogram**: Distributions

### Optimizing API Usage

1. **Batch Similar Requests**
   - Create initial visualization
   - Make multiple change requests on the same base

2. **Clear Problem Statements**
   - Reduces need for follow-up clarifications
   - Gets better results on first try

3. **Reuse Configurations**
   - Save the config dictionary
   - Regenerate graphs without API calls

## Troubleshooting

### API Key Issues

**Problem:** "Gemini API key not found"

**Solution:**
1. Verify `.env` file exists in project root
2. Check file contains: `GEMINI_API_KEY=your_key_here`
3. Ensure no extra spaces or quotes around the key
4. Restart your Python session

### Module Not Found Errors

**Problem:** `ModuleNotFoundError: No module named 'google.generativeai'`

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Display Issues

**Problem:** Graph doesn't display

**Solution:**
1. Check matplotlib backend:
   ```python
   import matplotlib
   print(matplotlib.get_backend())
   ```

2. Try different backend (add to script):
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg', 'WXAgg'
   ```

3. For headless environments, save to file instead of displaying

### Rate Limiting

**Problem:** "Rate limit exceeded" or "Quota exceeded"

**Solution:**
1. Wait a few minutes before retrying
2. Check your API quota in Google AI Studio
3. Implement delays between requests:
   ```python
   import time
   time.sleep(2)  # 2 second delay
   ```

### Poor Quality Results

**Problem:** Generated graphs don't match expectations

**Solution:**
1. Provide more specific problem statements
2. Include sample data for better context
3. Use the change request feature to refine
4. Specify the desired graph type in your prompt

### Connection Errors

**Problem:** "Connection timeout" or "Network error"

**Solution:**
1. Check internet connection
2. Verify firewall settings
3. Try again - temporary network issues are common
4. Check Google AI service status

## Advanced Features

### Custom Configuration

You can manually create or modify configurations:

```python
custom_config = {
    "graph_type": "bar",
    "title": "Custom Visualization",
    "x_label": "Categories",
    "y_label": "Values",
    "data": {
        "x_values": ["A", "B", "C"],
        "y_values": [10, 20, 15]
    },
    "summary": "Custom summary text",
    "insights": ["Custom insight"],
    "recommendations": ["Custom recommendation"]
}

fig = app.generate_graph(custom_config)
```

### Batch Processing

Process multiple visualizations:

```python
problems = [
    "Show sales trends",
    "Display customer demographics",
    "Analyze product performance"
]

for i, problem in enumerate(problems):
    config = app.process_prompt(problem)
    app.generate_graph(config, save_path=f'graph_{i}.png')
```

### Integration with Pandas

```python
import pandas as pd
import json

# Load data from CSV
df = pd.read_csv('data.csv')

# Convert to JSON for Gemini
data = df.to_json(orient='split')

# Create visualization
config = app.process_prompt("Analyze this data", data)
```

## Next Steps

- Explore the `examples.py` file for more usage patterns
- Check the API documentation for advanced features
- Join the community discussions
- Contribute improvements to the project

## Getting Help

- **Issues**: Open an issue on GitHub
- **Questions**: Check existing issues or start a discussion
- **Updates**: Watch the repository for new features
