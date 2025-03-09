import streamlit as st
import pickle
import re
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Loading models (FIXED the tfidf loading)
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    # Your existing cleaning function
    return cleanText

def get_top_skills(resume_text, n=5):
    feature_array = np.array(tfidf.get_feature_names_out())
    tfidf_scores = tfidf.transform([resume_text]).toarray().flatten()
    top_indices = tfidf_scores.argsort()[-n:][::-1]
    return feature_array[top_indices]

def main():
    st.title("📄 Resume Screening App")
    st.markdown("Upload your resume to see its predicted job category")
    
    uploaded_file = st.file_uploader('Choose a file', type=['txt', 'pdf'])
    
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        
        # Cleaning and prediction
        cleaned_resume = cleanResume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        
        # Get prediction and probability
        prediction_id = clf.predict(input_features)[0]
        proba = clf.predict_proba(input_features).max()
        
        # Get top skills
        top_skills = get_top_skills(cleaned_resume)
        
        # Display results
        st.success("Analysis Complete!")
        
        # Category mapping
        category_mapping = {
            6: "Data Science",
            # ... (your existing mapping)
        }
        
        category_name = category_mapping.get(prediction_id, "Unknown")
        
        # Main results card
        with st.container():
            st.subheader("Results")
            st.markdown(f"""
            **Predicted Category:** {category_name}  
            **Confidence Level:** {proba:.1%}  
            
            **Top Relevant Skills Detected:**  
            {', '.join(top_skills)}
            """)
            
            # Show advanced details in expanders
            with st.expander("View Processed Text"):
                st.write(cleaned_resume)
                
            with st.expander("How This Works"):
                st.markdown("""
                This app uses machine learning to analyze resume content:
                1. Cleans text (removes URLs, special characters)
                2. Converts text to numerical features (TF-IDF)
                3. Predicts category using a trained KNN model
                """)

if __name__ == '__main__':
    main()