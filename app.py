import streamlit as st
import pickle
import re
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Load models (FIXED loading code)
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
    
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)  
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def get_top_skills(resume_text, tfidf_vectorizer, n=5):
    # Transform cleaned resume
    transformed_text = tfidf_vectorizer.transform([resume_text])
    
    # Get feature names and scores
    feature_array = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_scores = transformed_text.toarray().flatten()
    
    # Sort scores in descending order
    sorted_indices = tfidf_scores.argsort()[::-1]
    top_skills = feature_array[sorted_indices][:n]
    
    return list(top_skills)

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])  # Fixed variable name
    
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        
        cleaned_resume = cleanResume(resume_text)  # Fixed variable name
        input_features = tfidf.transform([cleaned_resume])
        
        # Prediction
        prediction_id = clf.predict(input_features)[0]
        
        # Get top skills
        top_skills = get_top_skills(cleaned_resume, tfidf)
        
        # Display results
        category_mapping = {
            6: "Data Science", 12: "HR", 0: "Advocate", 1: "Arts", 24: "Web Designing",
            16: "Mechanical Engineer", 22: "Sales", 14: "Health and fitness", 5: "Civil Engineer",
            15: "Java Developer", 4: "Business Analyst", 21: "SAP Developer", 2: "Automation Testing",
            11: "Electrical Engineering", 18: "Operations Manager", 20: "Python Developer",
            8: "DevOps Engineer", 17: "Network Security Engineer", 19: "PMO", 7: "Database",
            13: "Hadoop", 10: "ETL Developer", 9: "DotNet Developer", 3: "Blockchain", 23: "Testing"
        }
        
        category_name = category_mapping.get(prediction_id, "Unknown")
        st.subheader("**Analysis Results**")
        st.success(f"Predicted Category: {category_name}")
        st.info(f"**Top 5 Relevant Skills:**\n{', '.join(top_skills)}")
        st.write("\n**Why this category?**")
        st.write("The model identified these key skills from your resume that are commonly associated with this job category. This simple analysis is based on word importance in your document.")

if __name__ == '__main__':
    main()