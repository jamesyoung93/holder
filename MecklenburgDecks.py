import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from matplotlib.colors import LinearSegmentedColormap

# Custom color map similar to the provided image
colors = ["#440154", "#30678D", "#35B779", "#FDE724"]
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'custom_cmap'

# Create the custom color map
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

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

# Reproject coordinates to latitude and longitude for sliders
geometry = [Point(xy) for xy in zip(df['X Coord'], df['Y Coord'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:2264")  # Assuming EPSG:2264 (NAD83 / North Carolina)
gdf = gdf.to_crs(epsg=4326)

# Update df with new coordinates
df['longitude'] = gdf.geometry.x
df['latitude'] = gdf.geometry.y

x_range = st.sidebar.slider(
    'Longitude Range',
    min_value=float(df['longitude'].min()),
    max_value=float(df['longitude'].max()),
    value=(float(df['longitude'].min()), float(df['longitude'].max())),
    step=0.001
)

y_range = st.sidebar.slider(
    'Latitude Range',
    min_value=float(df['latitude'].min()),
    max_value=float(df['latitude'].max()),
    value=(float(df['latitude'].min()), float(df['latitude'].max())),
    step=0.001
)

deck_area = st.sidebar.slider(
    'Deck Area Between',
    min_value=0,
    max_value=1000,
    value=(0, 500),
    step=1
)

#cell_size = st.sidebar.number_input('Cell Size', value=5000, step=100)
cell_size = st.sidebar.number_input('Cell Size', value=0.01, step=0.01, min_value = 0.01)

filtered_df = df[(df['longitude'] >= x_range[0]) & (df['longitude'] <= x_range[1]) &
                 (df['latitude'] >= y_range[0]) & (df['latitude'] <= y_range[1]) &
                 (df['Deck Area'] >= deck_area[0]) & (df['Deck Area'] <= deck_area[1])].copy()

#x_bins = np.arange(filtered_df['longitude'].min(), filtered_df['longitude'].max() + (cell_size / 100000), (cell_size / 100000))
#y_bins = np.arange(filtered_df['latitude'].min(), filtered_df['latitude'].max() + (cell_size / 100000), (cell_size / 100000))

x_bins = np.arange(filtered_df['longitude'].min(), filtered_df['longitude'].max() + cell_size, cell_size)
y_bins = np.arange(filtered_df['latitude'].min(), filtered_df['latitude'].max() + cell_size, cell_size)

filtered_df['X Bin'] = pd.cut(filtered_df['longitude'], bins=x_bins, labels=x_bins[:-1], include_lowest=True)
filtered_df['Y Bin'] = pd.cut(filtered_df['latitude'], bins=y_bins, labels=y_bins[:-1], include_lowest=True)

bin_stats = filtered_df.groupby(['X Bin', 'Y Bin'], observed=True).agg(
    Deck_Need_Probability_Sum=('Deck Need Probability', 'sum'),
    Count=('Deck Need Probability', 'size')
).reset_index()

# Drop NaN values
bin_stats = bin_stats.dropna(subset=['X Bin', 'Y Bin'])

# Convert bins to float
bin_stats['X Bin'] = bin_stats['X Bin'].astype(float)
bin_stats['Y Bin'] = bin_stats['Y Bin'].astype(float)

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(bin_stats['X Bin'], bin_stats['Y Bin'])]
gdf = gpd.GeoDataFrame(bin_stats, geometry=geometry, crs="EPSG:4326")  # Now using WGS84

# Create a map plot with contextily basemap
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Convert the heatmap to the same CRS as the basemap
xmin, ymin, xmax, ymax = gdf.total_bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Add basemap
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Add the scatter plot with custom color map
scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['Deck_Need_Probability_Sum'], cmap=cm, alpha=0.6, s=50, edgecolor='k', zorder=2)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Sum of Deck Need Probability')

# Plot adjustments
plt.title('Deck Need Probability with Basemap', fontsize=18)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)

# Display the plot in Streamlit
st.pyplot(fig)

# Placeholder for table of selected points
selected_points = st.empty()

# Create Streamlit inputs for min and max longitude and latitude
def get_filtered_rows(min_longitude, max_longitude, min_latitude, max_latitude):
    return filtered_df[(filtered_df['longitude'] >= min_longitude) &
                       (filtered_df['longitude'] <= max_longitude) &
                       (filtered_df['latitude'] >= min_latitude) &
                       (filtered_df['latitude'] <= max_latitude)]

# Define the cell size
cell_size = 0.01

# Create Streamlit inputs for min and max longitude and latitude
#min_longitude = st.number_input('Min Longitude', value=float(filtered_df['longitude'].min()), step=cell_size)
#max_longitude = st.number_input('Max Longitude', value=float(filtered_df['longitude'].max()), step=cell_size)
#min_latitude = st.number_input('Min Latitude', value=float(filtered_df['latitude'].min()), step=cell_size)
#max_latitude = st.number_input('Max Latitude', value=float(filtered_df['latitude'].max()), step=cell_size)

# Button to filter rows for the targeted area
if st.button('Show Rows for Targeted Area'):
    filtered_rows = get_filtered_rows(x_range[0], x_range[1], y_range[0],  y_range[1])
    st.session_state['filtered_rows'] = filtered_rows

# Display filtered rows if they exist in session state
if st.session_state.get('filtered_rows') is not None:
    st.dataframe(st.session_state['filtered_rows'])
