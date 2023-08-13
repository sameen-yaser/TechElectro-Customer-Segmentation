import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the processed data
data = pd.read_csv('TechElectro_Customer_Data.csv')

# Create a Streamlit app
st.title('Customer Segmentation Dashboard')
st.write('Explore customer segments and preferences')

# Sidebar
st.sidebar.title('Options')
show_data = st.sidebar.checkbox('Show Raw Data')
show_clusters = st.sidebar.checkbox('Show Customer Segments')

# Show raw data if checkbox is selected
if show_data:
    st.subheader('Raw Data')
    st.write(data)

# Show customer segments if checkbox is selected
if show_clusters:
    st.subheader('Customer Segments')
    # Select features for clustering
    X = data[['Age', 'AnnualIncome (USD)', 'TotalPurchases']]
    
    # Apply K-means clustering
    n_clusters = 3  # You can use the optimal number of clusters found earlier
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['Cluster'] = kmeans.fit_predict(X)
    
    # Visualize customer segments
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='AnnualIncome (USD)', y='TotalPurchases', hue='Cluster', palette='viridis', ax=ax)
    st.pyplot(fig)

# Run the Streamlit app
if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Suppress warning
    st.write('Navigate to the sidebar to explore options.')
