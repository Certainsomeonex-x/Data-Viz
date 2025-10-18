"""
Unit tests for the Data Visualization App

This test suite validates the core functionality without requiring API calls.
"""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
from data_viz_app import DataVizApp


class TestDataVizApp(unittest.TestCase):
    """Test cases for DataVizApp class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key_12345"
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_initialization(self, mock_model, mock_configure):
        """Test app initialization with API key."""
        app = DataVizApp(api_key=self.api_key)
        
        mock_configure.assert_called_once_with(api_key=self.api_key)
        mock_model.assert_called_once_with('gemini-pro')
        self.assertIsNotNone(app.model)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_initialization_without_api_key(self, mock_model, mock_configure):
        """Test app initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError) as context:
                DataVizApp()
            self.assertIn("API key not found", str(context.exception))
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_construct_prompt(self, mock_model, mock_configure):
        """Test prompt construction."""
        app = DataVizApp(api_key=self.api_key)
        
        problem = "Test problem statement"
        data = '{"test": "data"}'
        
        prompt = app._construct_prompt(problem, data)
        
        self.assertIn(problem, prompt)
        self.assertIn(data, prompt)
        self.assertIn("JSON format", prompt)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_parse_gemini_response_valid_json(self, mock_model, mock_configure):
        """Test parsing valid JSON response."""
        app = DataVizApp(api_key=self.api_key)
        
        response_text = '''
        Here is the configuration:
        {
            "graph_type": "bar",
            "title": "Test Graph",
            "x_label": "X",
            "y_label": "Y",
            "data": {
                "x_values": [1, 2, 3],
                "y_values": [10, 20, 30]
            },
            "summary": "Test summary",
            "insights": ["Insight 1"],
            "recommendations": ["Rec 1"]
        }
        '''
        
        config = app._parse_gemini_response(response_text)
        
        self.assertEqual(config['graph_type'], 'bar')
        self.assertEqual(config['title'], 'Test Graph')
        self.assertIn('data', config)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_generate_line_graph(self, mock_model, mock_configure):
        """Test line graph generation."""
        app = DataVizApp(api_key=self.api_key)
        
        config = {
            "graph_type": "line",
            "title": "Line Graph Test",
            "x_label": "Time",
            "y_label": "Value",
            "data": {
                "x_values": [1, 2, 3, 4, 5],
                "y_values": [10, 20, 15, 25, 30]
            }
        }
        
        fig = app.generate_graph(config)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_generate_bar_graph(self, mock_model, mock_configure):
        """Test bar graph generation."""
        app = DataVizApp(api_key=self.api_key)
        
        config = {
            "graph_type": "bar",
            "title": "Bar Graph Test",
            "x_label": "Category",
            "y_label": "Count",
            "data": {
                "x_values": ["A", "B", "C"],
                "y_values": [10, 20, 15]
            }
        }
        
        fig = app.generate_graph(config)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_generate_scatter_graph(self, mock_model, mock_configure):
        """Test scatter graph generation."""
        app = DataVizApp(api_key=self.api_key)
        
        config = {
            "graph_type": "scatter",
            "title": "Scatter Plot Test",
            "x_label": "X",
            "y_label": "Y",
            "data": {
                "x_values": [1, 2, 3, 4, 5],
                "y_values": [2, 4, 3, 5, 6]
            }
        }
        
        fig = app.generate_graph(config)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_generate_pie_chart(self, mock_model, mock_configure):
        """Test pie chart generation."""
        app = DataVizApp(api_key=self.api_key)
        
        config = {
            "graph_type": "pie",
            "title": "Pie Chart Test",
            "data": {
                "x_values": ["A", "B", "C"],
                "y_values": [30, 40, 30],
                "labels": ["A", "B", "C"]
            }
        }
        
        fig = app.generate_graph(config)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_display_summary(self, mock_model, mock_configure):
        """Test summary display formatting."""
        app = DataVizApp(api_key=self.api_key)
        
        config = {
            "graph_type": "line",
            "title": "Test Graph",
            "summary": "This is a test summary.",
            "insights": ["Insight 1", "Insight 2"],
            "recommendations": ["Recommendation 1"]
        }
        
        summary = app.display_summary(config)
        
        self.assertIn("VISUALIZATION SUMMARY", summary)
        self.assertIn("Test Graph", summary)
        self.assertIn("test summary", summary)
        self.assertIn("Insight 1", summary)
        self.assertIn("Recommendation 1", summary)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_multiple_series_line_graph(self, mock_model, mock_configure):
        """Test multi-series line graph generation."""
        app = DataVizApp(api_key=self.api_key)
        
        config = {
            "graph_type": "line",
            "title": "Multi-Series Line Graph",
            "x_label": "Time",
            "y_label": "Value",
            "data": {
                "x_values": [1, 2, 3, 4],
                "y_values": [[10, 20, 15, 25], [5, 15, 10, 20]],
                "labels": ["Series A", "Series B"]
            }
        }
        
        fig = app.generate_graph(config)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_save_graph(self, mock_model, mock_configure):
        """Test graph saving functionality."""
        import tempfile
        import os
        
        app = DataVizApp(api_key=self.api_key)
        
        config = {
            "graph_type": "line",
            "title": "Save Test",
            "x_label": "X",
            "y_label": "Y",
            "data": {
                "x_values": [1, 2, 3],
                "y_values": [1, 2, 3]
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fig = app.generate_graph(config, save_path=tmp_path)
            
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
            
            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation and edge cases."""
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_empty_config(self, mock_model, mock_configure):
        """Test handling of empty configuration."""
        app = DataVizApp(api_key="test_key")
        
        config = {}
        
        # Should not raise an error, uses defaults
        fig = app.generate_graph(config)
        self.assertIsNotNone(fig)
        plt.close(fig)
    
    @patch('data_viz_app.genai.configure')
    @patch('data_viz_app.genai.GenerativeModel')
    def test_minimal_config(self, mock_model, mock_configure):
        """Test minimal valid configuration."""
        app = DataVizApp(api_key="test_key")
        
        config = {
            "data": {
                "x_values": [1, 2, 3],
                "y_values": [1, 2, 3]
            }
        }
        
        fig = app.generate_graph(config)
        self.assertIsNotNone(fig)
        plt.close(fig)


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataVizApp))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit(run_tests())
