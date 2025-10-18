# Data-Viz

An intelligent data visualization application powered by Google's Gemini API. This application processes user problem statements and data, automatically generates appropriate visualizations, and provides comprehensive summaries and insights.

## Features

- 🤖 **AI-Powered Analysis**: Uses Google Gemini API to understand problem statements and generate relevant visualizations
- 📊 **Multiple Chart Types**: Supports bar charts, line graphs, scatter plots, pie charts, histograms, and more
- 📈 **Automatic Data Processing**: Generates sample data when needed or processes user-provided data
- 💡 **Intelligent Insights**: Provides summaries, key insights, and recommendations based on the visualization
- 🔄 **Interactive Mode**: Request changes and additional inference on existing visualizations
- 💾 **Export Capability**: Save visualizations as high-quality images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Certainsomeonex-x/Data-Viz.git
cd Data-Viz
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
   - Copy `.env.example` to `.env`
   - Add your Gemini API key to the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

### Interactive Mode

Run the application in interactive mode:
```bash
python data_viz_app.py
```

Follow the prompts to:
1. Create new visualizations by providing a problem statement and optional data
2. Request changes or additional analysis on existing visualizations
3. Save visualizations as image files

### Programmatic Usage

```python
from data_viz_app import DataVizApp

# Initialize the app
app = DataVizApp()

# Create a visualization
problem = "Show the trend of monthly sales for the last 6 months"
config = app.process_prompt(problem)

# Display the summary
print(app.display_summary(config))

# Generate and display the graph
fig = app.generate_graph(config, save_path='sales_trend.png')
plt.show()

# Request changes
updated_config = app.request_changes("Add a comparison with the previous year")
fig = app.generate_graph(updated_config)
plt.show()
```

## Example Use Cases

### 1. Sales Analysis
```
Problem: "Analyze quarterly sales performance across different product categories"
```
The app will generate appropriate visualizations with insights about trends and performance.

### 2. Scientific Data
```
Problem: "Visualize the relationship between temperature and reaction rate"
Data: {"temperature": [20, 30, 40, 50, 60], "rate": [0.5, 1.2, 2.8, 5.1, 8.3]}
```

### 3. Survey Results
```
Problem: "Show customer satisfaction survey results for 5 different service categories"
```

### 4. Financial Trends
```
Problem: "Compare monthly expenses vs income over the last year"
```

## Configuration Format

The Gemini API generates configurations in the following format:

```json
{
    "graph_type": "line",
    "title": "Monthly Sales Trend",
    "x_label": "Month",
    "y_label": "Sales ($)",
    "data": {
        "x_values": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "y_values": [1000, 1500, 1300, 1800, 2100, 2400],
        "labels": []
    },
    "summary": "The graph shows a general upward trend in sales...",
    "insights": [
        "Sales increased by 140% from January to June",
        "Notable spike in May suggests successful marketing campaign"
    ],
    "recommendations": [
        "Investigate factors behind May's success",
        "Prepare for potential seasonal variations"
    ]
}
```

## Supported Graph Types

- **bar**: Bar charts for categorical comparisons
- **line**: Line graphs for trends over time
- **scatter**: Scatter plots for correlation analysis
- **pie**: Pie charts for composition analysis
- **histogram**: Histograms for distribution analysis

## Requirements

- Python 3.8+
- Google Gemini API key
- See `requirements.txt` for package dependencies

## Project Structure

```
Data-Viz/
├── data_viz_app.py      # Main application file
├── requirements.txt     # Python dependencies
├── .env.example        # Example environment configuration
├── README.md           # This file
└── LICENSE            # License file
```

## API Key Setup

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See the LICENSE file for details.

## Troubleshooting

### API Key Issues
- Ensure your `.env` file is in the project root directory
- Verify your API key is valid and has the correct permissions
- Check that the `.env` file contains `GEMINI_API_KEY=your_key_here`

### Visualization Issues
- Ensure all required packages are installed: `pip install -r requirements.txt`
- Check that matplotlib backend is properly configured for your system
- For display issues, try different matplotlib backends

### Data Format Issues
- Provide data in JSON format for best results
- Ensure data structure matches the problem statement
- Let Gemini generate sample data if unsure about format

## Support

For issues and questions, please open an issue on the GitHub repository.
