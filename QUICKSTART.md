# Quick Start Guide

Get up and running with Data-Viz in 5 minutes!

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/Certainsomeonex-x/Data-Viz.git
cd Data-Viz

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Get Your API Key (1 minute)

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

## Step 3: Configure (30 seconds)

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your key
# GEMINI_API_KEY=your_key_here
```

Or on Linux/Mac:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Step 4: Run Your First Visualization (1 minute)

### Option A: Interactive Mode

```bash
python data_viz_app.py
```

Then:
1. Select option `1` (Create new visualization)
2. Enter: `Show sales trends for the last 6 months`
3. Press Enter to skip data (Gemini will generate sample data)
4. Review the results and generate the graph!

### Option B: Run Examples

```bash
python examples.py
```

This will generate 5 example visualizations automatically.

### Option C: Quick Python Script

Create a file `quick_test.py`:

```python
from data_viz_app import DataVizApp
import matplotlib.pyplot as plt

# Initialize
app = DataVizApp()

# Create visualization
problem = "Show monthly website traffic trends"
config = app.process_prompt(problem)

# Display summary
print(app.display_summary(config))

# Generate and show graph
app.generate_graph(config)
plt.show()
```

Run it:
```bash
python quick_test.py
```

## What's Next?

- Read the [User Guide](USER_GUIDE.md) for detailed usage
- Check [Examples](examples.py) for more use cases
- Review [Security Best Practices](SECURITY.md)
- Explore different graph types and problem statements

## Common First-Time Issues

### Issue: "API key not found"
**Fix:** Make sure `.env` file exists and contains `GEMINI_API_KEY=your_key`

### Issue: "Module not found"
**Fix:** Run `pip install -r requirements.txt`

### Issue: Graph doesn't display
**Fix:** Save to file instead:
```python
app.generate_graph(config, save_path='output.png')
```

## Sample Problem Statements to Try

1. **Sales Analysis**
   - "Compare quarterly sales across product categories"
   - "Show year-over-year revenue growth"

2. **Data Analysis**
   - "Visualize the correlation between price and demand"
   - "Display distribution of customer ages"

3. **Trends**
   - "Show daily active users over the past month"
   - "Analyze temperature trends throughout the year"

4. **Comparisons**
   - "Compare marketing channel effectiveness"
   - "Show performance metrics across teams"

## Getting Help

- **Issues?** Check [Troubleshooting](USER_GUIDE.md#troubleshooting)
- **Questions?** Open an issue on GitHub
- **Ideas?** We welcome contributions!

---

**🎉 Congratulations!** You're now ready to create AI-powered data visualizations.
