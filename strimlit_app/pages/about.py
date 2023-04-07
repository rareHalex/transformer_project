import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="About",
    page_icon="👋",
)


def inference_sweets():
    HtmlFile = open("about.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    print(source_code)
    components.html(source_code)
    st.markdown("![Alt Text](https://media.tenor.com/JTWkLQyjzWYAAAAC/peach-peach-cat.gif)")
    st.write('The work was performed by Kalinin Vladislav')


inference_sweets()