from flask import Flask, request, render_template
import re
import pickle
from pdfminer.high_level import extract_text

app = Flask(__name__)

# Load your pre-trained files
le = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))

# Clean resume text (from your code)
def clean_resume(txt):
    clean_text = re.sub(r'http\S+\s', ' ', txt)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+\s', ' ', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)  
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text) 
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

# Skills mapping (YOU MUST COMPLETE THIS!)
skills_mapping = {
    "Data Science": ["Python / R", "Machine Learning & AI", "Data Visualization", "SQL & NoSQL Databases", "Statistical Analysis"],
    "HR": ["Recruitment & Talent Acquisition", "Employee Relations", "HR Software", "Performance Management", "Labor Laws & Compliance"],
    "Advocate": ["Legal Research & Writing", "Contract & Negotiation", "Courtroom Advocacy", "Case Analysis & Strategy", "Legal Regulations"],
    "Arts": ["Creativity & Design Thinking", "Drawing / Painting / Sculpture", "Digital Art & Graphic Design", "Art History & Theory", "Portfolio Development"],
    "Web Designing": ["HTML, CSS, JavaScript", "UI/UX Design", "Responsive Web Design", "Graphic Design Tools", "WordPress / CMS"],
    "Mechanical Engineer": ["CAD Software", "Thermodynamics", "Material Science", "Manufacturing Processes", "Problem-Solving"],
    "Sales": ["CRM", "Negotiation & Persuasion", "Market Research", "Communication", "Lead Generation"],
    "Health and Fitness": ["Nutrition & Diet Planning", "Personal Training", "Anatomy & Physiology", "Coaching & Motivation", "CPR & First Aid"],
    "Civil Engineer": ["Structural Analysis", "AutoCAD & Civil 3D", "Project Management", "Geotechnical Skills", "Building Codes"],
    "Java Developer": ["Core Java", "Spring Boot & Hibernate", "RESTful APIs", "Database Management", "Multithreading"],
    "Business Analyst": ["Requirement Gathering", "Data Analysis", "Process Mapping", "Stakeholder Communication", "BI Tools"],
    "SAP Developer": ["SAP ABAP", "SAP Fiori", "SAP Modules", "Integration", "Debugging"],
    "Automation Testing": ["Selenium", "TestNG / JUnit", "CI/CD", "API Testing", "Scripting Languages"],
    "Electrical Engineering": ["Circuit Design", "Embedded Systems", "Power Systems", "MATLAB", "PLC & SCADA"],
    "Operations Manager": ["Supply Chain", "Process Optimization", "Budgeting", "Team Management", "Risk Management"],
    "Python Developer": ["Python", "Django / Flask", "Data Structures", "RESTful APIs", "Databases"],
    "DevOps Engineer": ["CI/CD", "Cloud Platforms", "Docker & Kubernetes", "Infrastructure as Code", "Monitoring & Logging"],
    "Network Security Engineer": ["Network Protocols", "Cybersecurity", "VPNs & IDS", "SIEM Tools", "Risk Assessment"],
    "PMO": ["Project Planning", "Agile & Scrum", "Risk Management", "Budgeting", "Stakeholder Communication"],
    "Database": ["SQL & NoSQL", "Performance Tuning", "Backup & Recovery", "Data Security", "Cloud Databases"],
    "Hadoop": ["Hadoop Ecosystem", "Apache Spark", "Big Data Processing", "NoSQL Databases", "Python / Scala"],
    "ETL Developer": ["ETL Tools", "Data Warehousing", "SQL & Scripting", "Performance Tuning", "Cloud ETL"],
    "DotNet Developer": ["C# & .NET Core", "ASP.NET MVC", "Entity Framework", "Frontend Technologies", "Azure Services"],
    "Blockchain": ["Smart Contracts", "Blockchain Frameworks", "Cryptography", "Consensus Mechanisms", "Web3 & dApps"],
    "Testing": ["Manual Testing", "Automation Testing", "Performance Testing", "API Testing", "Bug Tracking"]
}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Extract text from PDF or TXT
            if file.filename.endswith('.pdf'):
                text = extract_text(file)
            else:
                text = file.read().decode('utf-8')
            
            # Clean and predict
            cleaned_text = clean_resume(text)
            features = tfidf.transform([cleaned_text])
            pred_id = clf.predict(features)[0]
            category = le.inverse_transform([pred_id])[0]
            
            # Get skills (default to empty list if category missing)
            skills = skills_mapping.get(category, [])
            return render_template('index.html', 
                                   category=category, 
                                   skills=skills[:5])  # Show top 5 skills
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

print("helo")