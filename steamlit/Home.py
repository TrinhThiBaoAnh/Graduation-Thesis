import streamlit as st
# Use a Markdown block to center-align content
col1, col2, col3 = st.columns(3)

with col2:
    st.image("download.png")
st.markdown(
    "<h1 style='text-align: center; color: #FF5733; font-size: 30px;'>"
    "RESEARCH AND DEVELOP A UNET-BASED NETWORK ARCHITECTURE FOR LESION SEGMENTATION IN GASTROINTESTINAL ENDOSCOPY IMAGES"
    "</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center; font-size: 20px;'>"
    "Research Supervisor: PhD. Vu Thi Ly"
    "</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center; font-size: 20px;'>"
    "Author: Trinh T. Bao Anh"
    "</h1>",
    unsafe_allow_html=True
)