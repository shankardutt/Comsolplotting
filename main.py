import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any, Set
from collections import OrderedDict
import io

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.pattern_columns = []
        self.numeric_columns = []
        
    def load_data(self, file) -> None:
        """Load and preprocess the data file."""
        content = file.getvalue().decode('utf-8')
        
        # Skip comment lines and get headers
        lines = [line for line in content.split('\n') if line.strip() and not line.startswith('%')]
        headers = [h.strip() for h in content.split('\n') if h.startswith('%')][-1].replace('%', '').split('  ')
        headers = [h.strip() for h in headers if h.strip()]
        
        # Parse data lines
        data = []
        for line in lines:
            values = [val.strip() for val in line.split() if val.strip()]
            if len(values) == len(headers):
                data.append(values)
        
        # Create DataFrame
        self.df = pd.DataFrame(data, columns=headers)
        
        # Convert numeric columns
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
                if col not in self.numeric_columns:
                    self.numeric_columns.append(col)
            except:
                continue
                
    def identify_patterns(self, threshold_ratio: float = 0.5) -> Dict[str, List[Any]]:
        """
        Identify columns with repeating patterns.
        threshold_ratio: maximum unique values / total values ratio to consider as pattern
        """
        patterns = OrderedDict()
        for col in self.df.columns:
            unique_values = sorted(self.df[col].unique())
            if len(unique_values) < len(self.df) * threshold_ratio:
                patterns[col] = unique_values
        return patterns
    
    def get_filtered_data(self, pattern_filters: Dict[str, Any]) -> pd.DataFrame:
        """Get data filtered by multiple pattern values."""
        filtered_df = self.df.copy()
        for col, value in pattern_filters.items():
            if value is not None:
                filtered_df = filtered_df[filtered_df[col] == value]
        return filtered_df

class StreamlitApp:
    def __init__(self):
        self.analyzer = DataAnalyzer()
        
    def run(self):
        st.title("Comsol Multi-Pattern Data Analysis")
        st.header("Shankar Dutt")
        st.subheader("shankar.dutt@anu.edu.au")
        st.write("Upload your data file to analyze patterns and create visualizations.")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
        
        if uploaded_file is not None:
            # Load data
            self.analyzer.load_data(uploaded_file)
            
            # Identify patterns
            patterns = self.analyzer.identify_patterns()
            
            if patterns:
                st.header("Pattern Selection")
                
                # Create pattern filters
                pattern_filters = {}
                cols = st.columns(min(3, len(patterns)))  # Up to 3 columns
                
                for idx, (col_name, unique_values) in enumerate(patterns.items()):
                    with cols[idx % 3]:
                        pattern_filters[col_name] = st.selectbox(
                            f"Select {col_name}:",
                            options=[None] + list(unique_values),
                            help=f"Filter by {col_name}"
                        )
                
                # Remove None values from filters
                active_filters = {k: v for k, v in pattern_filters.items() if v is not None}
                
                if active_filters:
                    # Get filtered data
                    filtered_data = self.analyzer.get_filtered_data(active_filters)
                    
                    # Create tabs for different visualizations
                    tab1, tab2 = st.tabs(["2D Plot", "3D Plot"])
                    
                    with tab1:
                        self.create_2d_plot(filtered_data, active_filters)
                    
                    with tab2:
                        self.create_3d_plot(filtered_data, active_filters)
                else:
                    st.info("Select at least one pattern filter to visualize data")
            else:
                st.warning("No repeating patterns found in the data.")
    
    def create_2d_plot(self, data: pd.DataFrame, filters: Dict[str, Any]):
        """Create 2D plot with selected axes."""
        st.subheader("2D Visualization")
        
        # Get available columns (excluding pattern columns)
        available_cols = [col for col in data.columns if col not in filters]
        
        # Select axes
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis:", options=available_cols, key="2d_x")
        with col2:
            y_col = st.selectbox("Select Y-axis:", options=available_cols, key="2d_y")
        
        if x_col and y_col:
            # Create plot
            fig = px.scatter(data, x=x_col, y=y_col,
                           title=f"{y_col} vs {x_col}")
            
            # Add filter information to title
            filter_text = " | ".join([f"{k}={v}" for k, v in filters.items()])
            fig.update_layout(title=f"{y_col} vs {x_col}<br><sup>Filters: {filter_text}</sup>")
            
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig)
    
    def create_3d_plot(self, data: pd.DataFrame, filters: Dict[str, Any]):
        """Create 3D plot with selected axes."""
        st.subheader("3D Visualization")
        
        # Get available columns (excluding pattern columns)
        available_cols = [col for col in data.columns if col not in filters]
        
        # Select axes
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Select X-axis:", options=available_cols, key="3d_x")
        with col2:
            y_col = st.selectbox("Select Y-axis:", options=available_cols, key="3d_y")
        with col3:
            z_col = st.selectbox("Select Z-axis:", options=available_cols, key="3d_z")
        
        if x_col and y_col and z_col:
            # Create 3D plot
            fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col,
                              title=f"3D Plot: {z_col} vs {y_col} vs {x_col}")
            
            # Add filter information to title
            filter_text = " | ".join([f"{k}={v}" for k, v in filters.items()])
            fig.update_layout(
                title=f"3D Plot: {z_col} vs {y_col} vs {x_col}<br><sup>Filters: {filter_text}</sup>",
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col
                )
            )
            
            st.plotly_chart(fig)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()