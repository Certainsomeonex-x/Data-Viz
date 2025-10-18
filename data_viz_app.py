"""
Data Visualization Application with Gemini API Integration

This application processes user problem statements and data using Google's Gemini API,
generates appropriate visualizations, and provides summaries and insights.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dotenv import load_dotenv


class DataVizApp:
    """Main application class for data visualization with Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DataVizApp.
        
        Args:
            api_key: Google Gemini API key. If not provided, loads from .env file.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in .env file or pass it as a parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.last_config = None
        self.last_figure = None
        
    def process_prompt(self, problem_statement: str, data: Optional[str] = None) -> Dict[str, Any]:
        """
        Process the problem statement using Gemini API.
        
        Args:
            problem_statement: User's problem statement
            data: Optional data in JSON or CSV format
            
        Returns:
            Dictionary containing graph configuration and analysis
        """
        # Construct prompt for Gemini
        prompt = self._construct_prompt(problem_statement, data)
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_gemini_response(response.text)
            return result
        except Exception as e:
            raise Exception(f"Error processing with Gemini API: {str(e)}")
    
    def _construct_prompt(self, problem_statement: str, data: Optional[str] = None) -> str:
        """Construct a detailed prompt for Gemini API."""
        prompt = f"""Given the following problem statement and data, generate a comprehensive visualization configuration.

Problem Statement: {problem_statement}

{f"Data: {data}" if data else "No specific data provided - please generate sample data relevant to the problem."}

Please provide a response in the following JSON format:
{{
    "graph_type": "type of graph (bar, line, scatter, pie, histogram, etc.)",
    "title": "descriptive title for the graph",
    "x_label": "x-axis label",
    "y_label": "y-axis label",
    "data": {{
        "x_values": [list of x values],
        "y_values": [list of y values or multiple series],
        "labels": [optional labels for categories/legend]
    }},
    "summary": "detailed summary explaining what the graph shows and how it relates to the problem statement",
    "insights": ["key insight 1", "key insight 2", "..."],
    "recommendations": ["recommendation 1", "recommendation 2", "..."]
}}

Ensure the data is meaningful and relevant to the problem statement. If multiple data series are needed, structure them appropriately.
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the Gemini API response to extract JSON configuration."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        
        if json_match:
            try:
                config = json.loads(json_match.group())
                return config
            except json.JSONDecodeError:
                pass
        
        # If JSON parsing fails, create a structured response from the text
        return {
            "graph_type": "line",
            "title": "Data Visualization",
            "x_label": "X Axis",
            "y_label": "Y Axis",
            "data": {
                "x_values": list(range(10)),
                "y_values": list(range(10))
            },
            "summary": response_text,
            "insights": ["Unable to parse structured response"],
            "recommendations": ["Please refine the problem statement"]
        }
    
    def generate_graph(self, config: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate a graph based on the configuration.
        
        Args:
            config: Graph configuration dictionary
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        self.last_config = config
        
        graph_type = config.get('graph_type', 'line').lower()
        title = config.get('title', 'Data Visualization')
        x_label = config.get('x_label', 'X Axis')
        y_label = config.get('y_label', 'Y Axis')
        data = config.get('data', {})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_values = data.get('x_values', [])
        y_values = data.get('y_values', [])
        labels = data.get('labels', [])
        
        # Handle empty data
        if not x_values:
            x_values = [0]
        if not y_values:
            y_values = [0]
        
        # Generate graph based on type
        if graph_type == 'bar':
            if y_values and isinstance(y_values[0], list):
                # Multiple series
                x_pos = np.arange(len(x_values))
                width = 0.8 / len(y_values)
                for i, series in enumerate(y_values):
                    offset = width * i - (width * len(y_values) / 2) + width / 2
                    label = labels[i] if i < len(labels) else f'Series {i+1}'
                    ax.bar(x_pos + offset, series, width, label=label)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_values)
                ax.legend()
            else:
                ax.bar(x_values, y_values)
                
        elif graph_type == 'line':
            if y_values and isinstance(y_values[0], list):
                # Multiple series
                for i, series in enumerate(y_values):
                    label = labels[i] if i < len(labels) else f'Series {i+1}'
                    ax.plot(x_values, series, marker='o', label=label)
                ax.legend()
            else:
                ax.plot(x_values, y_values, marker='o')
                
        elif graph_type == 'scatter':
            if y_values and isinstance(y_values[0], list):
                for i, series in enumerate(y_values):
                    label = labels[i] if i < len(labels) else f'Series {i+1}'
                    ax.scatter(x_values, series, label=label, alpha=0.6)
                ax.legend()
            else:
                ax.scatter(x_values, y_values, alpha=0.6)
                
        elif graph_type == 'pie':
            ax.pie(y_values, labels=labels if labels else x_values, autopct='%1.1f%%')
            
        elif graph_type == 'histogram':
            ax.hist(y_values, bins=len(x_values) if x_values else 10)
            
        else:
            # Default to line graph
            ax.plot(x_values, y_values, marker='o')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        if graph_type != 'pie':
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.last_figure = fig
        return fig
    
    def display_summary(self, config: Dict[str, Any]) -> str:
        """
        Display a formatted summary of the visualization.
        
        Args:
            config: Graph configuration dictionary
            
        Returns:
            Formatted summary string
        """
        summary_text = "\n" + "="*80 + "\n"
        summary_text += "VISUALIZATION SUMMARY\n"
        summary_text += "="*80 + "\n\n"
        
        summary_text += f"Graph Type: {config.get('graph_type', 'N/A')}\n"
        summary_text += f"Title: {config.get('title', 'N/A')}\n\n"
        
        summary_text += "ANALYSIS:\n"
        summary_text += "-" * 80 + "\n"
        summary_text += f"{config.get('summary', 'No summary available')}\n\n"
        
        if 'insights' in config and config['insights']:
            summary_text += "KEY INSIGHTS:\n"
            summary_text += "-" * 80 + "\n"
            for i, insight in enumerate(config['insights'], 1):
                summary_text += f"{i}. {insight}\n"
            summary_text += "\n"
        
        if 'recommendations' in config and config['recommendations']:
            summary_text += "RECOMMENDATIONS:\n"
            summary_text += "-" * 80 + "\n"
            for i, rec in enumerate(config['recommendations'], 1):
                summary_text += f"{i}. {rec}\n"
            summary_text += "\n"
        
        summary_text += "="*80 + "\n"
        
        return summary_text
    
    def request_changes(self, change_request: str) -> Dict[str, Any]:
        """
        Request changes or inference based on the current visualization.
        
        Args:
            change_request: Description of desired changes or questions
            
        Returns:
            Updated configuration dictionary
        """
        if not self.last_config:
            raise ValueError("No previous visualization exists. Please create a visualization first.")
        
        prompt = f"""Based on the following existing visualization configuration and the user's change request, 
provide an updated configuration or additional inference.

Current Configuration:
{json.dumps(self.last_config, indent=2)}

User's Request: {change_request}

Please provide the updated configuration in the same JSON format as before, or if the request is for analysis/inference, 
provide insights in the summary and insights fields.
"""
        
        try:
            response = self.model.generate_content(prompt)
            updated_config = self._parse_gemini_response(response.text)
            self.last_config = updated_config
            return updated_config
        except Exception as e:
            raise Exception(f"Error processing change request: {str(e)}")
    
    def run_interactive(self):
        """Run the application in interactive mode."""
        print("\n" + "="*80)
        print("DATA VISUALIZATION APP WITH GEMINI AI")
        print("="*80 + "\n")
        
        while True:
            print("\nOptions:")
            print("1. Create new visualization")
            print("2. Request changes/inference on current visualization")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                problem_statement = input("\nEnter your problem statement: ").strip()
                if not problem_statement:
                    print("Problem statement cannot be empty.")
                    continue
                
                data_input = input("\nEnter data (JSON/CSV) or press Enter to skip: ").strip()
                data = data_input if data_input else None
                
                print("\nProcessing with Gemini AI...")
                try:
                    config = self.process_prompt(problem_statement, data)
                    print("\n✓ Configuration generated successfully!")
                    
                    print(self.display_summary(config))
                    
                    save_choice = input("\nGenerate and display graph? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        save_path = input("Enter save path (or press Enter to skip saving): ").strip()
                        save_path = save_path if save_path else None
                        
                        fig = self.generate_graph(config, save_path)
                        plt.show()
                        
                        if save_path:
                            print(f"\n✓ Graph saved to {save_path}")
                    
                except Exception as e:
                    print(f"\n✗ Error: {str(e)}")
            
            elif choice == '2':
                if not self.last_config:
                    print("\n✗ No visualization exists. Please create one first (option 1).")
                    continue
                
                change_request = input("\nDescribe the changes or inference you need: ").strip()
                if not change_request:
                    print("Request cannot be empty.")
                    continue
                
                print("\nProcessing change request...")
                try:
                    updated_config = self.request_changes(change_request)
                    print("\n✓ Changes processed successfully!")
                    
                    print(self.display_summary(updated_config))
                    
                    regenerate = input("\nRegenerate graph with changes? (y/n): ").strip().lower()
                    if regenerate == 'y':
                        save_path = input("Enter save path (or press Enter to skip saving): ").strip()
                        save_path = save_path if save_path else None
                        
                        fig = self.generate_graph(updated_config, save_path)
                        plt.show()
                        
                        if save_path:
                            print(f"\n✓ Graph saved to {save_path}")
                
                except Exception as e:
                    print(f"\n✗ Error: {str(e)}")
            
            elif choice == '3':
                print("\nThank you for using Data Visualization App!")
                break
            
            else:
                print("\n✗ Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main entry point for the application."""
    try:
        app = DataVizApp()
        app.run_interactive()
    except ValueError as e:
        print(f"\nConfiguration Error: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Created a .env file with your GEMINI_API_KEY")
        print("2. Or set the GEMINI_API_KEY environment variable")
        print("\nExample .env file:")
        print("GEMINI_API_KEY=your_api_key_here")
    except Exception as e:
        print(f"\nUnexpected Error: {str(e)}")


if __name__ == "__main__":
    main()
