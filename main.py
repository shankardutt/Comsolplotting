import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any, Set
from collections import OrderedDict
import io
import base64

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

    def format_data_for_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format the data for display and export."""
        display_df = df.copy()
        for col in display_df.select_dtypes(include=[np.float64]).columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.6e}" if abs(x) < 0.001 else f"{x:.6f}")
        return display_df

class StreamlitApp:
    def __init__(self):
        self.analyzer = DataAnalyzer()
        
    def get_download_link(self, df: pd.DataFrame, filename: str) -> str:
        """Generate a download link for the dataframe."""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        return href
        
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
                
                # Create comparison mode toggle
                compare_mode = st.checkbox("Enable Comparison Mode", 
                                        help="Compare multiple pattern combinations in the same plot")
                
                if compare_mode:
                    max_comparisons = 5
                    num_comparisons = st.number_input("Number of comparisons", 
                                                    min_value=2, max_value=max_comparisons, value=2)
                    comparison_filters = []
                    
                    # Create multiple pattern filter sets
                    for i in range(num_comparisons):
                        st.subheader(f"Dataset {i+1}")
                        pattern_filters = {}
                        cols = st.columns(min(3, len(patterns)))
                        
                        for idx, (col_name, unique_values) in enumerate(patterns.items()):
                            with cols[idx % 3]:
                                pattern_filters[col_name] = st.selectbox(
                                    f"Select {col_name}:",
                                    options=[None] + list(unique_values),
                                    key=f"comp_{i}_{col_name}"
                                )
                        
                        # Remove None values from filters
                        active_filters = {k: v for k, v in pattern_filters.items() if v is not None}
                        if active_filters:
                            comparison_filters.append(active_filters)
                    
                    if comparison_filters:
                        # Create tabs for different visualizations
                        tab1, tab2, tab3 = st.tabs(["2D Comparison", "3D Comparison", "Data View"])
                        
                        with tab1:
                            self.create_2d_comparison(comparison_filters)
                        
                        with tab2:
                            self.create_3d_comparison(comparison_filters)
                            
                        with tab3:
                            self.show_data_comparison(comparison_filters)
                
                else:
                    # Single dataset mode
                    pattern_filters = {}
                    cols = st.columns(min(3, len(patterns)))
                    
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
                        # Create tabs for different visualizations
                        tab1, tab2, tab3 = st.tabs(["2D Plot", "3D Plot", "Data View"])
                        
                        with tab1:
                            self.create_2d_plot(active_filters)
                        
                        with tab2:
                            self.create_3d_plot(active_filters)
                            
                        with tab3:
                            self.show_data_view(active_filters)
                    else:
                        st.info("Select at least one pattern filter to visualize data")
            else:
                st.warning("No repeating patterns found in the data.")
    
    def create_2d_plot(self, filters: Dict[str, Any]):
        """Create 2D plot with selected axes."""
        st.subheader("2D Visualization")
        
        filtered_data = self.analyzer.get_filtered_data(filters)
        available_cols = [col for col in filtered_data.columns if col not in filters]
        
        # Select axes
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis:", options=available_cols, key="2d_x")
        with col2:
            y_col = st.selectbox("Select Y-axis:", options=available_cols, key="2d_y")
        
        if x_col and y_col:
            fig = px.scatter(filtered_data, x=x_col, y=y_col)
            
            filter_text = " | ".join([f"{k}={v}" for k, v in filters.items()])
            fig.update_layout(title=f"{y_col} vs {x_col}<br><sup>Filters: {filter_text}</sup>")
            
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig)
    
    def create_2d_comparison(self, comparison_filters: List[Dict[str, Any]]):
        """Create 2D comparison plot."""
        st.subheader("2D Comparison")
        
        # Get common available columns
        available_cols = set(self.analyzer.df.columns)
        for filters in comparison_filters:
            available_cols = available_cols - set(filters.keys())
        available_cols = sorted(list(available_cols))
        
        # Select axes
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis:", options=available_cols, key="comp_2d_x")
        with col2:
            y_col = st.selectbox("Select Y-axis:", options=available_cols, key="comp_2d_y")
        
        if x_col and y_col:
            fig = go.Figure()
            
            for idx, filters in enumerate(comparison_filters):
                filtered_data = self.analyzer.get_filtered_data(filters)
                filter_text = " | ".join([f"{k}={v}" for k, v in filters.items()])
                
                fig.add_trace(go.Scatter(
                    x=filtered_data[x_col],
                    y=filtered_data[y_col],
                    mode='lines+markers',
                    name=f"Dataset {idx+1}: {filter_text}"
                ))
            
            fig.update_layout(
                title="2D Comparison Plot",
                xaxis_title=x_col,
                yaxis_title=y_col
            )
            
            st.plotly_chart(fig)
    
    def create_3d_plot(self, filters: Dict[str, Any]):
        """Create 3D plot with selected axes."""
        st.subheader("3D Visualization")
        
        filtered_data = self.analyzer.get_filtered_data(filters)
        available_cols = [col for col in filtered_data.columns if col not in filters]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Select X-axis:", options=available_cols, key="3d_x")
        with col2:
            y_col = st.selectbox("Select Y-axis:", options=available_cols, key="3d_y")
        with col3:
            z_col = st.selectbox("Select Z-axis:", options=available_cols, key="3d_z")
        
        if x_col and y_col and z_col:
            fig = px.scatter_3d(filtered_data, x=x_col, y=y_col, z=z_col)
            
            filter_text = " | ".join([f"{k}={v}" for k, v in filters.items()])
            fig.update_layout(
                title=f"3D Plot<br><sup>Filters: {filter_text}</sup>",
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col
                )
            )
            
            st.plotly_chart(fig)
    
    def create_3d_comparison(self, comparison_filters: List[Dict[str, Any]]):
        """Create 3D comparison plot."""
        st.subheader("3D Comparison")
        
        # Get common available columns
        available_cols = set(self.analyzer.df.columns)
        for filters in comparison_filters:
            available_cols = available_cols - set(filters.keys())
        available_cols = sorted(list(available_cols))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Select X-axis:", options=available_cols, key="comp_3d_x")
        with col2:
            y_col = st.selectbox("Select Y-axis:", options=available_cols, key="comp_3d_y")
        with col3:
            z_col = st.selectbox("Select Z-axis:", options=available_cols, key="comp_3d_z")
        
        if x_col and y_col and z_col:
            fig = go.Figure()
            
            for idx, filters in enumerate(comparison_filters):
                filtered_data = self.analyzer.get_filtered_data(filters)
                filter_text = " | ".join([f"{k}={v}" for k, v in filters.items()])
                
                fig.add_trace(go.Scatter3d(
                    x=filtered_data[x_col],
                    y=filtered_data[y_col],
                    z=filtered_data[z_col],
                    mode='lines+markers',
                    name=f"Dataset {idx+1}: {filter_text}"
                ))
            
            fig.update_layout(
                title="3D Comparison Plot",
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col
                )
            )
            
            st.plotly_chart(fig)
    
    def show_data_view(self, filters: Dict[str, Any]):
        """Show data view with download option."""
        st.subheader("Data View")
        
        filtered_data = self.analyzer.get_filtered_data(filters)
        display_data = self.analyzer.format_data_for_display(filtered_data)
        
        # Show data table
        st.dataframe(display_data)
        
        # Create download button
        filter_text = "_".join([f"{k}{v}" for k, v in filters.items()])
        filename = f"filtered_data_{filter_text}.csv"
        st.markdown(self.get_download_link(display_data, filename), unsafe_allow_html=True)
    
    def show_data_comparison(self, comparison_filters: List[Dict[str, Any]]):
        """Show comparison data view with download options."""
        st.subheader("Data Comparison View")
        
        for idx, filters in enumerate(comparison_filters):
            st.subheader(f"Dataset {idx+1}")
            filtered_data = self.analyzer.get_filtered_data(filters)
            display_data = self.analyzer.format_data_for_display(filtered_data)
            
            # Show data table
            st.dataframe(display_data)
            
            # Create download button
            filter_text = "_".join([f"{k}{v}" for k, v in filters.items()])
            filename = f"dataset_{idx+1}_{filter_text}.csv"
            st.markdown(self.get_download_link(display_data, filename), unsafe_allow_html=True)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()