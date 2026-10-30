
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Genetic Data Compression",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Data Compression and Pattern Visualization")

model = joblib.load("genetic_pca_lda_model.joblib")
variance = pd.read_csv("pca_explained_variance.csv")
pca = pd.read_csv("pca_patient_coordinates.csv")
lda = pd.read_csv("lda_patient_coordinates.csv")

st.success("Model loaded successfully!")

col1, col2 = st.columns(2)

col1.metric("Total Records", len(pca))
col2.metric(
    "PCA Components for 95% Variance",
    model["components_for_95_percent_variance"]
)

st.header("PCA Explained Variance")

fig1 = px.line(
    variance,
    x="Component",
    y="Cumulative Variance",
    markers=True
)
fig1.add_hline(y=0.95, line_dash="dash")
st.plotly_chart(fig1, use_container_width=True)

st.header("PCA Visualization")

fig2 = px.scatter(
    pca,
    x="PC1",
    y="PC2",
    color="Genetic Disorder"
)
st.plotly_chart(fig2, use_container_width=True)

st.header("LDA Visualization")

fig3 = px.scatter(
    lda,
    x="LD1",
    y="LD2",
    color="Genetic Disorder"
)
st.plotly_chart(fig3, use_container_width=True)

st.header("Dataset Preview")
st.dataframe(pca.head(20))

st.info("Educational project only. Not a medical diagnostic tool.")
