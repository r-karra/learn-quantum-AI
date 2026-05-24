import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(page_title="My Streamlit App", page_icon="🚀")

    st.title("Welcome to Streamlit! 🚀")
    st.write("This is a simple app created to help you get started.")

    # Sidebar for inputs
    st.sidebar.header("Configuration")
    name = st.sidebar.text_input("Enter your name", "Guest")
    
    # Main content
    st.header(f"Hello, {name}!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Table")
        df = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'D'],
            'Values': np.random.randint(10, 100, 4)
        })
        st.dataframe(df)

    with col2:
        st.subheader("Interactive Chart")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c']
        )
        st.line_chart(chart_data)

    if st.button("Celebrate!"):
        st.balloons()
        st.success("Yay! You ran your first Streamlit app.")

if __name__ == "__main__":
    main()
