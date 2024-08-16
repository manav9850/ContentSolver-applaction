import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_pipeline()

st.title("Question Answering with Hugging Face")

st.write("Ask a question based on the provided context.")


context = st.text_area("Context", height=200)


question = st.text_input("Question")

if st.button("Get Answer"):
    if context and question:
        # Use the pipeline to get the answer
        result = qa_pipeline(question=question, context=context)

        # Display the answer
        st.write(f"**Answer:** {result['answer']}")
        st.write(f"**Score:** {result['score']:.4f}")
    else:
        st.write("Please provide both context and a question.")

st.markdown("Enter a context above and ask a question about it. The model will extract the answer from the context.")
