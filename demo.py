#!/usr/bin/env python3
"""
Demo script for Data-Viz application

This script demonstrates the core capabilities without requiring a real API key.
It uses mock responses to show how the application works.
"""

import json
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
from data_viz_app import DataVizApp


def create_mock_response(problem_statement):
    """Create a mock Gemini response based on the problem."""
    
    # Determine appropriate visualization based on problem
    if "sales" in problem_statement.lower() or "trend" in problem_statement.lower():
        return {
            "graph_type": "line",
            "title": "Monthly Sales Trends",
            "x_label": "Month",
            "y_label": "Sales ($)",
            "data": {
                "x_values": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                "y_values": [45000, 52000, 48000, 61000, 73000, 82000],
                "labels": []
            },
            "summary": "The visualization shows a clear upward trend in monthly sales over the first half of the year. Starting from $45,000 in January, sales grew by 82% to reach $82,000 in June. There's a notable acceleration in growth from April onwards, with the steepest increase between May and June.",
            "insights": [
                "Sales increased by 82% from January to June, indicating strong business growth",
                "The growth rate accelerated in Q2, with a 34% increase from April to June",
                "May showed the highest month-over-month growth at 19.7%",
                "Only February to March showed a slight decline (-7.7%), which quickly recovered"
            ],
            "recommendations": [
                "Investigate factors contributing to the May-June surge to replicate this success",
                "Analyze what caused the March dip and implement preventive measures",
                "Consider scaling operations to support continued growth trajectory",
                "Prepare for seasonal variations by building a larger historical dataset"
            ]
        }
    
    elif "category" in problem_statement.lower() or "comparison" in problem_statement.lower():
        return {
            "graph_type": "bar",
            "title": "Product Category Performance",
            "x_label": "Category",
            "y_label": "Revenue ($)",
            "data": {
                "x_values": ["Electronics", "Clothing", "Food", "Books", "Toys"],
                "y_values": [125000, 98000, 156000, 45000, 67000],
                "labels": []
            },
            "summary": "This bar chart compares revenue across five product categories. Food leads with $156,000 in revenue, followed by Electronics at $125,000. Books show the lowest revenue at $45,000, representing an opportunity for growth or strategic review.",
            "insights": [
                "Food category generates 32% of total revenue, making it the top performer",
                "Combined Electronics and Food account for 57% of total revenue",
                "Books underperform with only 9% of total revenue",
                "There's a $111,000 gap between the highest and lowest performing categories"
            ],
            "recommendations": [
                "Invest more in Food and Electronics marketing given their strong performance",
                "Analyze the Books category to determine if it should be expanded or phased out",
                "Consider cross-selling opportunities between high and low performers",
                "Study successful strategies from Food category for application to others"
            ]
        }
    
    elif "scatter" in problem_statement.lower() or "correlation" in problem_statement.lower():
        return {
            "graph_type": "scatter",
            "title": "Price vs. Sales Volume Correlation",
            "x_label": "Price ($)",
            "y_label": "Units Sold",
            "data": {
                "x_values": [10, 15, 20, 25, 30, 35, 40],
                "y_values": [850, 720, 620, 480, 380, 280, 210],
                "labels": []
            },
            "summary": "The scatter plot reveals a strong negative correlation between price and sales volume. As price increases from $10 to $40, units sold decrease from 850 to 210, demonstrating typical price elasticity of demand.",
            "insights": [
                "Clear negative correlation: higher prices result in lower sales volume",
                "A $10 increase in price roughly corresponds to 200-unit decrease in sales",
                "The relationship appears roughly linear, suggesting consistent price sensitivity",
                "The $10 price point maximizes volume, while higher prices may maximize revenue"
            ],
            "recommendations": [
                "Perform revenue optimization to find the ideal price point (Price × Volume)",
                "Consider market segmentation with different price tiers",
                "Test price points between $20-$25 where volume-revenue balance seems optimal",
                "Implement dynamic pricing based on demand and inventory levels"
            ]
        }
    
    else:
        # Default generic response
        return {
            "graph_type": "line",
            "title": "Data Visualization",
            "x_label": "X Axis",
            "y_label": "Y Axis",
            "data": {
                "x_values": [1, 2, 3, 4, 5, 6],
                "y_values": [10, 25, 20, 35, 45, 40],
                "labels": []
            },
            "summary": "This visualization presents the data in a clear and interpretable format, showing trends and patterns over the measured period.",
            "insights": [
                "The data shows overall upward trend with some fluctuation",
                "Peak value occurs at position 5",
                "Slight decrease observed in the final measurement"
            ],
            "recommendations": [
                "Continue monitoring for emerging patterns",
                "Consider collecting additional data points for better trend analysis",
                "Investigate causes of fluctuations for process improvement"
            ]
        }


def demo_example_1():
    """Demo 1: Sales Trend Analysis."""
    print("\n" + "="*80)
    print("DEMO 1: SALES TREND ANALYSIS")
    print("="*80)
    
    with patch('data_viz_app.genai.configure'), \
         patch('data_viz_app.genai.GenerativeModel') as mock_model:
        
        # Setup mock
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        problem = "Show monthly sales trends for the first half of 2024"
        mock_response = create_mock_response(problem)
        
        mock_instance.generate_content.return_value.text = json.dumps(mock_response)
        
        # Run demo
        app = DataVizApp(api_key="demo_key")
        config = app.process_prompt(problem)
        
        print(f"\nProblem: {problem}\n")
        print(app.display_summary(config))
        
        # Generate graph
        fig = app.generate_graph(config)
        plt.savefig('/tmp/demo1_sales_trend.png', dpi=150, bbox_inches='tight')
        print("✓ Graph saved to /tmp/demo1_sales_trend.png")
        plt.close()


def demo_example_2():
    """Demo 2: Category Comparison."""
    print("\n" + "="*80)
    print("DEMO 2: PRODUCT CATEGORY COMPARISON")
    print("="*80)
    
    with patch('data_viz_app.genai.configure'), \
         patch('data_viz_app.genai.GenerativeModel') as mock_model:
        
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        problem = "Compare revenue across product categories"
        mock_response = create_mock_response(problem)
        
        mock_instance.generate_content.return_value.text = json.dumps(mock_response)
        
        app = DataVizApp(api_key="demo_key")
        config = app.process_prompt(problem)
        
        print(f"\nProblem: {problem}\n")
        print(app.display_summary(config))
        
        fig = app.generate_graph(config)
        plt.savefig('/tmp/demo2_category_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Graph saved to /tmp/demo2_category_comparison.png")
        plt.close()


def demo_example_3():
    """Demo 3: Scatter Plot Analysis."""
    print("\n" + "="*80)
    print("DEMO 3: PRICE-VOLUME CORRELATION ANALYSIS")
    print("="*80)
    
    with patch('data_viz_app.genai.configure'), \
         patch('data_viz_app.genai.GenerativeModel') as mock_model:
        
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        problem = "Show the correlation between price and sales volume"
        mock_response = create_mock_response(problem)
        
        mock_instance.generate_content.return_value.text = json.dumps(mock_response)
        
        app = DataVizApp(api_key="demo_key")
        config = app.process_prompt(problem)
        
        print(f"\nProblem: {problem}\n")
        print(app.display_summary(config))
        
        fig = app.generate_graph(config)
        plt.savefig('/tmp/demo3_price_volume.png', dpi=150, bbox_inches='tight')
        print("✓ Graph saved to /tmp/demo3_price_volume.png")
        plt.close()


def demo_change_request():
    """Demo 4: Making Changes to Visualization."""
    print("\n" + "="*80)
    print("DEMO 4: REQUESTING CHANGES TO VISUALIZATION")
    print("="*80)
    
    with patch('data_viz_app.genai.configure'), \
         patch('data_viz_app.genai.GenerativeModel') as mock_model:
        
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        # Initial visualization
        problem = "Show monthly sales trends"
        initial_response = create_mock_response(problem)
        
        mock_instance.generate_content.return_value.text = json.dumps(initial_response)
        
        app = DataVizApp(api_key="demo_key")
        config = app.process_prompt(problem)
        
        print(f"\nInitial Problem: {problem}")
        print("\n[Initial visualization created]")
        
        # Request changes
        change_request = "Add comparison with previous year and highlight growth rate"
        
        # Mock updated response
        updated_response = {
            "graph_type": "line",
            "title": "Monthly Sales Trends - Year-over-Year Comparison",
            "x_label": "Month",
            "y_label": "Sales ($)",
            "data": {
                "x_values": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                "y_values": [
                    [45000, 52000, 48000, 61000, 73000, 82000],  # 2024
                    [38000, 42000, 45000, 48000, 52000, 58000]   # 2023
                ],
                "labels": ["2024", "2023"]
            },
            "summary": "The updated visualization now compares 2024 sales with 2023, revealing significant year-over-year growth. In January 2024, sales were $45,000 compared to $38,000 in January 2023, representing 18% growth. By June, the gap widened with 2024 reaching $82,000 versus $58,000 in 2023, showing 41% growth.",
            "insights": [
                "Year-over-year growth accelerated from 18% in January to 41% in June",
                "2024 consistently outperformed 2023 across all months",
                "The growth gap widened significantly in Q2 2024",
                "Average monthly growth rate across the period was approximately 27%"
            ],
            "recommendations": [
                "Document successful strategies from 2024 for future replication",
                "Allocate resources to maintain and accelerate this growth trajectory",
                "Set ambitious but achievable targets for H2 2024 based on this momentum",
                "Share best practices across teams to maximize organizational learning"
            ]
        }
        
        mock_instance.generate_content.return_value.text = json.dumps(updated_response)
        
        updated_config = app.request_changes(change_request)
        
        print(f"\nChange Request: {change_request}\n")
        print(app.display_summary(updated_config))
        
        fig = app.generate_graph(updated_config)
        plt.savefig('/tmp/demo4_yoy_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Updated graph saved to /tmp/demo4_yoy_comparison.png")
        plt.close()


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("DATA-VIZ APPLICATION DEMO")
    print("="*80)
    print("\nThis demo showcases the application's capabilities using mock data.")
    print("For real usage, you'll need a Gemini API key (see QUICKSTART.md)")
    print("\nGenerating visualizations...")
    
    try:
        demo_example_1()
        demo_example_2()
        demo_example_3()
        demo_change_request()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated visualizations:")
        print("  • /tmp/demo1_sales_trend.png")
        print("  • /tmp/demo2_category_comparison.png")
        print("  • /tmp/demo3_price_volume.png")
        print("  • /tmp/demo4_yoy_comparison.png")
        print("\nNext steps:")
        print("  1. View the generated images")
        print("  2. Get your Gemini API key from https://makersuite.google.com/app/apikey")
        print("  3. Follow QUICKSTART.md to try with real data")
        print("  4. Explore examples.py for more use cases")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
