
from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from pydantic import BaseModel
from typing import List
import seaborn as sns
import pandas as pd
import matplotlib as plt
import numpy as np

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY not found in environment variables.')
client = genai.Client("GEMINI_API_KEY")

#defining the basic model for our graph
class GraphData(BaseModel):
    x_val : List[float]
    y_val : List[float]
    x_label : str
    y_label : str
    graph_type : str
    summary : str
    insights : str

    @classmethod
    def from_csv(cls, file_path: str, x_col: str, y_col: str, x_label: str = '', y_label: str = '', graph_type: str = '', summary: str = '', insights: str = ''):
        df = pd.read_csv(file_path)
        return cls(
            x_val=df[x_col].tolist(),
            y_val=df[y_col].tolist(),
            x_label=x_label or x_col,
            y_label=y_label or y_col,
            graph_type=graph_type,
            summary=summary,
            insights=insights
        )

    @classmethod
    def from_json(cls, file_path: str, x_key: str, y_key: str, x_label: str = '', y_label: str = '', graph_type: str = '', summary: str = '', insights: str = ''):
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(
            x_val=data[x_key],
            y_val=data[y_key],
            x_label=x_label or x_key,
            y_label=y_label or y_key,
            graph_type=graph_type,
            summary=summary,
            insights=insights
        )
