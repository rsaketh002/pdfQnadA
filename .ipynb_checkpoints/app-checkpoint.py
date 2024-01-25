import streamlit as st
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, TFAutoModelForQuestionAnswering
import PyPDF2

# Function to extract text from PDF
def read_pdf(file):
    with open(file, "rb") as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        text = ""
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
    return text

# Function for question answering using the specified model
def answer_question(context, question, model_name="deepset/bert-large-uncased-whole-word-masking-squad2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
    
    inputs = tokenizer(question, context, return_tensors="tf")
    outputs = model(inputs)
    
    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = (tf.argmax(outputs.end_logits, axis=1) + 1).numpy()[0]
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# Streamlit app
def main():
    st.title("PDF Question Answering App")
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read PDF
        pdf_text = read_pdf(uploaded_file)

        # Display extracted text
        st.subheader("Extracted Text from PDF")
        st.text(pdf_text)

        # Question input
        question = st.text_input("Ask a question:")

        if st.button("Get Answer"):
            # Answer the question
            answer = answer_question(pdf_text, question)
            
            # Display the answer
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
