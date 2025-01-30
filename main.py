import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any
import io

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.pattern_columns = []
        self.value_columns = []
        
    def load_data(self, file) -> None:
        """Load and preprocess the data file."""
        # Read the content of the uploaded file
        content = file.getvalue().decode('utf-8')
        
        # Skip comment lines starting with %
        lines = [line for line in content.split('\n') if line.strip() and not line.startswith('%')]
        
        # Get headers from the last comment line
        headers = [h.strip() for h in content.split('\n') if h.startswith('%')][-1].replace('%', '').split('  ')
        headers = [h.strip() for h in headers if h.strip()]
        
        # Parse the data lines
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
            except:
                continue
    
    def identify_patterns(self) -> Dict[str, List[Any]]:
        """Identify columns with repeating patterns."""
        patterns = {}
        for col in self.df.columns:
            unique_values = self.df[col].unique()
            if len(unique_values) < len(self.df) / 2:  # If column has repeated values
                patterns[col] = sorted(unique_values)
        return patterns
    
    def get_filtered_data(self, pattern_col: str, pattern_value: Any) -> pd.DataFrame:
        """Get data filtered by pattern value."""
        return self.df[self.df[pattern_col] == pattern_value]

class StreamlitApp:
    def __init__(self):
        self.analyzer = DataAnalyzer()
        
    def run(self):
        st.title("Comsol Data Analysis and Visualization")
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
                st.header("Detected Patterns")
                
                # Pattern selection
                pattern_col = st.selectbox(
                    "Select pattern column:",
                    options=list(patterns.keys())
                )
                
                if pattern_col:
                    # Create tabs for different visualizations
                    tab1, tab2 = st.tabs(["Single Pattern Plot", "Compare Patterns"])
                    
                    with tab1:
                        self.single_pattern_plot(pattern_col, patterns)
                    
                    with tab2:
                        self.compare_patterns_plot(pattern_col, patterns)
            else:
                st.warning("No repeating patterns found in the data.")
    
    def single_pattern_plot(self, pattern_col: str, patterns: Dict[str, List[Any]]):
        """Create single pattern plot."""
        st.subheader("Single Pattern Visualization")
        
        # Select pattern value
        pattern_value = st.selectbox(
            f"Select value for {pattern_col}:",
            options=patterns[pattern_col]
        )
        
        # Get filtered data
        filtered_data = self.analyzer.get_filtered_data(pattern_col, pattern_value)
        
        # Select columns for x and y axes
        remaining_cols = [col for col in filtered_data.columns if col != pattern_col]
        x_col = st.selectbox("Select X-axis:", options=remaining_cols, key="single_x")
        y_col = st.selectbox("Select Y-axis:", options=remaining_cols, key="single_y")
        
        # Create plot
        if x_col and y_col:
            fig = px.scatter(filtered_data, x=x_col, y=y_col, 
                           title=f"{y_col} vs {x_col} for {pattern_col}={pattern_value}")
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig)
    
    def compare_patterns_plot(self, pattern_col: str, patterns: Dict[str, List[Any]]):
        """Create comparison plot for different pattern values."""
        st.subheader("Pattern Comparison Visualization")
        
        # Select columns for x and y axes
        remaining_cols = [col for col in self.analyzer.df.columns if col != pattern_col]
        x_col = st.selectbox("Select X-axis:", options=remaining_cols, key="compare_x")
        y_col = st.selectbox("Select Y-axis:", options=remaining_cols, key="compare_y")
        
        if x_col and y_col:
            fig = go.Figure()
            
            # Add traces for each pattern value
            for value in patterns[pattern_col]:
                filtered_data = self.analyzer.get_filtered_data(pattern_col, value)
                fig.add_trace(go.Scatter(
                    x=filtered_data[x_col],
                    y=filtered_data[y_col],
                    mode='lines+markers',
                    name=f"{pattern_col}={value}"
                ))
            
            fig.update_layout(
                title=f"Comparison of {y_col} vs {x_col} for different {pattern_col} values",
                xaxis_title=x_col,
                yaxis_title=y_col
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
