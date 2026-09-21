import streamlit as st

from rag import get_suggestion

st.set_page_config(page_title="What Can I Cook?", page_icon="🍳")

st.title("🍳 What Can I Cook?")
st.caption("Tell me what ingredients you have, and I'll suggest recipes you can make.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! What ingredients do you have at home? (e.g. paneer, tomato, onion, rice)",
            "recipes": None,
        }
    ]


def render_recipes(recipes):
    with st.expander("See retrieved candidate recipes"):
        for r in recipes:
            st.markdown(f"**{r['name']}** ({r['cuisine']})")
            st.write(f"Ingredients: {', '.join(r['ingredients'])}")
            st.write(r["instructions"])
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["recipes"]:
            render_recipes(message["recipes"])

if prompt := st.chat_input("e.g. paneer, tomato, onion, rice"):
    history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": prompt, "recipes": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, recipes = get_suggestion(prompt, history=history)
            except Exception as e:
                answer, recipes = f"Something went wrong: {e}", None
        st.markdown(answer)
        if recipes:
            render_recipes(recipes)

    st.session_state.messages.append({"role": "assistant", "content": answer, "recipes": recipes})
