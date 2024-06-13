import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

x_range = st.sidebar.slider(
    'X Coord Range',
    min_value=1384768.00,
    max_value=1534680.00,
    value=(1384768.00, 1534680.00),
    step=1000.00
)

y_range = st.sidebar.slider(
    'Y Coord Range',
    min_value=464747.08,
    max_value=645088.00,
    value=(464747.08, 645088.00),
    step=1000.00
)

deck_area = st.sidebar.slider(
    'Deck Area Between',
    min_value=0,
    max_value=1000,
    value=(0, 500),
    step=1
)

cell_size = st.sidebar.number_input('Cell Size', value=5000, step=1000)

filtered_df = df[(df['X Coord'] >= x_range[0]) & (df['X Coord'] <= x_range[1]) &
                 (df['Y Coord'] >= y_range[0]) & (df['Y Coord'] <= y_range[1]) &
                 (df['Deck Area'] >= deck_area[0]) & (df['Deck Area'] <= deck_area[1])].copy()

x_bins = np.arange(filtered_df['X Coord'].min(), filtered_df['X Coord'].max() + cell_size, cell_size)
y_bins = np.arange(filtered_df['Y Coord'].min(), filtered_df['Y Coord'].max() + cell_size, cell_size)

filtered_df['X Bin'] = pd.cut(filtered_df['X Coord'], bins=x_bins, labels=x_bins[:-1], include_lowest=True)
filtered_df['Y Bin'] = pd.cut(filtered_df['Y Coord'], bins=y_bins, labels=y_bins[:-1], include_lowest=True)

bin_stats = filtered_df.groupby(['X Bin', 'Y Bin'], observed=True).agg(
    Deck_Need_Probability_Sum=('Deck Need Probability', 'sum'),
    Count=('Deck Need Probability', 'size')
).reset_index()

# Pivot the table for heatmap plotting and sort the index for proper ordering
heatmap_data = bin_stats.pivot(index='Y Bin', columns='X Bin', values='Deck_Need_Probability_Sum').sort_index(ascending=False)

# Plot using Matplotlib
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Sum of Deck Need Probability'})
plt.title('Heatmap of Deck Need Probability')
plt.xlabel('X Coord')
plt.ylabel('Y Coord')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Display the heatmap
st.pyplot(plt)

# Placeholder for table of selected points
selected_points = st.empty()

# Function to get filtered rows
def get_filtered_rows(x_bin, y_bin):
    return filtered_df[(filtered_df['X Bin'] == x_bin) & (filtered_df['Y Bin'] == y_bin)]

if st.button('Show Rows for Targeted Area'):
    example_x_bin = x_bins[1]
    example_y_bin = y_bins[1]
    st.session_state['selected_bin'] = (example_x_bin, example_y_bin)

if st.session_state.get('selected_bin'):
    x_bin, y_bin = st.session_state['selected_bin']
    filtered_rows = get_filtered_rows(x_bin, y_bin)
    selected_points.dataframe(filtered_rows)

