def load_css(file_path='styles.css'):
    with open(file_path, 'r') as f:
        css = f.read()
    import streamlit as st
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)