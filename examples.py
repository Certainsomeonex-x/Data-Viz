"""
Example usage of the Data Visualization App

This script demonstrates how to use the DataVizApp programmatically.
"""

from data_viz_app import DataVizApp
import matplotlib.pyplot as plt


def example_sales_analysis():
    """Example 1: Sales trend analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Sales Trend Analysis")
    print("="*80)
    
    app = DataVizApp()
    
    problem = "Show the trend of monthly sales for the last 6 months with actual sales data"
    data = """{
        "months": ["January", "February", "March", "April", "May", "June"],
        "sales": [45000, 52000, 48000, 61000, 73000, 82000]
    }"""
    
    print(f"\nProblem: {problem}")
    print(f"Data: {data}")
    
    # Process with Gemini
    config = app.process_prompt(problem, data)
    
    # Display summary
    print(app.display_summary(config))
    
    # Generate graph
    fig = app.generate_graph(config, save_path='example_sales_trend.png')
    plt.savefig('example_sales_trend.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graph saved as 'example_sales_trend.png'")
    plt.close()
    
    return app, config


def example_product_comparison():
    """Example 2: Product category comparison."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Product Category Comparison")
    print("="*80)
    
    app = DataVizApp()
    
    problem = "Compare sales performance across different product categories for Q1"
    data = """{
        "categories": ["Electronics", "Clothing", "Food", "Books", "Toys"],
        "sales": [125000, 98000, 156000, 45000, 67000]
    }"""
    
    print(f"\nProblem: {problem}")
    print(f"Data: {data}")
    
    config = app.process_prompt(problem, data)
    print(app.display_summary(config))
    
    fig = app.generate_graph(config, save_path='example_category_comparison.png')
    plt.savefig('example_category_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graph saved as 'example_category_comparison.png'")
    plt.close()
    
    return app, config


def example_with_changes():
    """Example 3: Creating a visualization and then requesting changes."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Visualization with Modifications")
    print("="*80)
    
    app = DataVizApp()
    
    # Initial visualization
    problem = "Show website traffic trends over the past week"
    
    print(f"\nInitial Problem: {problem}")
    config = app.process_prompt(problem)
    print(app.display_summary(config))
    
    fig = app.generate_graph(config)
    plt.savefig('example_traffic_initial.png', dpi=300, bbox_inches='tight')
    print("\n✓ Initial graph saved as 'example_traffic_initial.png'")
    plt.close()
    
    # Request changes
    change_request = "Add comparison with the previous week and highlight the peak day"
    print(f"\n\nChange Request: {change_request}")
    
    updated_config = app.request_changes(change_request)
    print(app.display_summary(updated_config))
    
    fig = app.generate_graph(updated_config)
    plt.savefig('example_traffic_updated.png', dpi=300, bbox_inches='tight')
    print("\n✓ Updated graph saved as 'example_traffic_updated.png'")
    plt.close()
    
    return app, updated_config


def example_no_data():
    """Example 4: Let Gemini generate appropriate sample data."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Auto-generated Data")
    print("="*80)
    
    app = DataVizApp()
    
    problem = "Visualize the distribution of student grades in a class of 50 students"
    
    print(f"\nProblem: {problem}")
    print("(No data provided - Gemini will generate sample data)")
    
    config = app.process_prompt(problem)
    print(app.display_summary(config))
    
    fig = app.generate_graph(config)
    plt.savefig('example_grades_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graph saved as 'example_grades_distribution.png'")
    plt.close()
    
    return app, config


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("DATA VISUALIZATION APP - EXAMPLE USAGE")
    print("="*80)
    
    try:
        # Example 1: Sales Analysis
        example_sales_analysis()
        
        # Example 2: Product Comparison
        example_product_comparison()
        
        # Example 3: With Changes
        example_with_changes()
        
        # Example 4: Auto-generated Data
        example_no_data()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  - example_sales_trend.png")
        print("  - example_category_comparison.png")
        print("  - example_traffic_initial.png")
        print("  - example_traffic_updated.png")
        print("  - example_grades_distribution.png")
        
    except ValueError as e:
        print(f"\n✗ Configuration Error: {str(e)}")
        print("\nMake sure you have set up your .env file with GEMINI_API_KEY")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    main()
