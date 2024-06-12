import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from io import BytesIO

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Mecklenburg_Deck_Prospects.csv")

df = load_data()

# Check for the correct column name
if 'predicted_prob' not in df.columns:
    st.error("The column 'predicted_prob' does not exist in the dataset.")
    st.stop()

df['Deck Need Probability'] = df['predicted_prob']

# Sidebar for user inputs
st.sidebar.header('Filter Options')

# Numeric filter inputs
min_x = st.sidebar.number_input('Min X Coord', value=float(df['X Coord'].min()))
max_x = st.sidebar.number_input('Max X Coord', value=float(df['X Coord'].max()))
min_y = st.sidebar.number_input('Min Y Coord', value=float(df['Y Coord'].min()))
max_y = st.sidebar.number_input('Max Y Coord', value=float(df['Y Coord'].max()))

# Adjust cell size
cell_size = st.sidebar.number_input('Cell Size', value=5000, step=1000)

# Filter data based on user inputs
filtered_df = df[(df['X Coord'] >= min_x) & (df['X Coord'] <= max_x) &
                 (df['Y Coord'] >= min_y) & (df['Y Coord'] <= max_y)]

# Assign bins to X and Y coordinates
x_bins = np.arange(filtered_df['X Coord'].min(), filtered_df['X Coord'].max() + cell_size, cell_size)
y_bins = np.arange(filtered_df['Y Coord'].min(), filtered_df['Y Coord'].max() + cell_size, cell_size)

filtered_df['X Bin'] = pd.cut(filtered_df['X Coord'], bins=x_bins, labels=x_bins[:-1], include_lowest=True)
filtered_df['Y Bin'] = pd.cut(filtered_df['Y Coord'], bins=y_bins, labels=y_bins[:-1], include_lowest=True)

# Calculate the sum of Deck Area Flag and count the number of rows in each bin
bin_stats = filtered_df.groupby(['X Bin', 'Y Bin']).agg(
    Deck_Need_Probability_Sum=('Deck Need Probability', 'sum'),
    Count=('Deck Need Probability', 'size')
).reset_index()

# Pivot the table for heatmap plotting
#heatmap_data = bin_stats.pivot('Y Bin', 'X Bin', 'Deck_Need_Probability_Sum')

heatmap_data = bin_stats.pivot(index='Y Bin', columns='X Bin', values='Deck_Need_Probability_Sum')


# Create the Plotly heatmap figure
fig = px.density_heatmap(
    filtered_df, x='X Coord', y='Y Coord', z='Deck Need Probability', nbinsx=len(x_bins)-1, nbinsy=len(y_bins)-1,
    color_continuous_scale='Viridis', histfunc='sum'
)

fig.update_layout(clickmode='event+select')

# Use session state to manage selected bin
if 'selected_bin' not in st.session_state:
    st.session_state['selected_bin'] = None

# Display the Plotly figure
st.plotly_chart(fig)

# Placeholder for table of selected points
selected_points = st.empty()

# Function to get filtered rows
def get_filtered_rows(x_bin, y_bin):
    return filtered_df[(filtered_df['X Bin'] == x_bin) & (filtered_df['Y Bin'] == y_bin)]

# Simulate cell selection using a button for demonstration purposes
if st.button('Show Rows for Example Bin'):
    # Example bins for demonstration purposes
    example_x_bin = x_bins[1]  # Replace with actual selected X bin
    example_y_bin = y_bins[1]  # Replace with actual selected Y bin
    st.session_state['selected_bin'] = (example_x_bin, example_y_bin)

# Display filtered rows based on selected bin
if st.session_state['selected_bin']:
    x_bin, y_bin = st.session_state['selected_bin']
    filtered_rows = get_filtered_rows(x_bin, y_bin)
    selected_points.dataframe(filtered_rows)

    # Download button
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    download_data = to_excel(filtered_rows)
    st.download_button(
        label="Download filtered data as Excel",
        data=download_data,
        file_name='filtered_data.xlsx'
    )
