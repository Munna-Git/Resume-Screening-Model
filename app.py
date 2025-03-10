# -*- python3.10 -*-

# Updated app.py (Improved)
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import PyPDF2  # For PDF parsing


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load models
le = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))

def cleanResume(txt):
    # Same cleaning function as the ML model
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)  
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Resume Screening App")
    st.markdown("Upload your resume (PDF or TXT)")

    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docs"])
    
    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = uploaded_file.read().decode('utf-8')
            
            cleaned_resume = cleanResume(resume_text)
            
            # Prediction
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]
            category = le.inverse_transform([prediction_id])[0]
            
            # Display
            st.subheader("Predicted Category:")
            st.success(category)
            st.markdown("---")
            st.subheader("Cleaned Resume Text:")
            st.text(cleaned_resume[:500] + "...")  # Show sample text
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == '__main__':
    main()