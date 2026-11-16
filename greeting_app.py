import streamlit as st

st.set_page_config(page_title="Greetings & Calculator app", page_icon="✨")

with st.sidebar:
    st.header("Settings")
    name = st.text_input("Your name", value="Student")

st.title("Greetings & Calculator App")
st.write(f"Welcome to the Greetings & Calculator app, {name}!")

tab1, tab2 = st.tabs(["Greetings", "Calculator"])

with tab1:
    st.subheader("A Friendly Greeting")
    mood = st.selectbox("How are you feeling?", ["Happy", "Neutral", "Sad", "Tired", "Other"])
    if mood == "Happy":
        st.success("Great to hear you are in a happy mood!")
    elif mood == "Neutral":
        st.warning("Ok, no worries, hope you have a better day tomorrow!")
    elif mood == "Sad":
        st.error("That sucks. Maybe do some exercise to feel better!")
    else:
        st.info("Get some rest!")

with tab2:
    st.subheader("Simple Calculator")
    col1, col2 = st.columns(2)
    with col1:
        num1 = st.number_input("First number", value=0.0)
    with col2:
        num2 = st.number_input("Second number", value=0.0)
    
    operation = st.radio("Operation", ["+", "-", "x", "÷"], horizontal=True)
    if st.button("Calculate"):
        if operation == "+":
            result = num1 + num2
        elif operation == "-":
            result = num1 - num2
        elif operation == "x":
            result = num1 * num2
        else:
            result = num1 / num2 if num2 != 0 else "undefined (division by zero)"
        st.metric("Result", result)