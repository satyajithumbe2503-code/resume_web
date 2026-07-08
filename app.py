import re
import time
import PyPDF2
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fpdf import FPDF
import base64


# --- 1. INITIAL APP & PAGE CONFIG ---
st.set_page_config(
    page_title="Resume Job Match Scorer & Builder",
    page_icon="https://files.softicons.com/download/toolbar-icons/mono-general-icons-2-by-custom-icon-design/ico/document.ico",
    layout="wide"  
)

# --- 2. DOWNLOAD NLTK RESOURCES ---
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt_tab')
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

download_nltk_resources()

# --- 3. HELPER FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += " " + extracted
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    
    if not resume_processed or not job_processed:
        return 0.0, resume_processed, job_processed
        
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed

def create_pdf(name, title, email, phone, linkedin, summary, experience, education, skills):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=name.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
    pdf.cell(200, 10, txt=f"{email} | {phone} | {linkedin}".encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
    
    pdf.ln(10)
    sections = [("Summary", summary), ("Experience", experience), ("Education", education), ("Skills", skills)]
    for head, content in sections:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=head, ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=content.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_roll_no" not in st.session_state:
    st.session_state.user_roll_no = ""
if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = False

# --- 5. LOGIN GATE ---
if not st.session_state.logged_in:
    st.title("✨ Welcome — Please Enter Your Details")
    with st.form("login_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        roll_no = st.text_input("Roll No")
        submitted = st.form_submit_button("Enter")

    if submitted:
        if not name.strip() or not email.strip() or not roll_no.strip():
            st.warning("Please fill in Name, Email, and Roll No before continuing.")
        else:
            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.session_state.user_email = email
            st.session_state.user_roll_no = roll_no
            st.rerun()
    st.stop()

# --- 6. GLOBAL CUSTOM STYLING ---
if st.session_state.dark_theme:
    st.html("""
        <style>
        .stApp { background-color: #0E1117; }
        .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, h1, h2, h3, .stHeader { color: #FAFAFA !important; }
        .stMetric { background-color: #262730; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); border: 1px solid #444; }
        .rec-box { background-color: #262730; padding: 15px; border-radius: 8px; border: 1px solid #f0c674; border-left: 5px solid #f59e0b; margin-top: 10px; color: #FAFAFA; }
        [data-testid="stFileUploaderDropzone"] { background-color: #1E222B; border: 2px dashed #9E7BB5; border-radius: 10px; }
        .stTextArea textarea, .stTextInput input { background-color: #1E222B; border: 2px solid #9E7BB5; border-radius: 10px; color: #FAFAFA !important; }
        div[data-testid="stSidebar"] { background-color: #262730 !important; }
        
        .resume-paper { background: #1E222B; padding: 35px; border: 1px solid #30363D; border-radius: 4px; font-family: 'Times New Roman', Times, serif; color: #E6EDF2; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .resume-section-head { font-family: 'Arial', sans-serif; font-size: 1.15em; font-weight: bold; color: #58a6ff; border-bottom: 1px solid #30363D; padding-bottom: 2px; margin-top: 20px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
        .resume-paper { 
        background: #FFFFFF; 
        padding: 40px; 
        border: 5px solid #6a4c93; 
        border-radius: 15px;      
        font-family: 'Times New Roman', Times, serif; 
        color: #111111; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
        </style>
        """)
else:
    st.html("""
        <style>
        .stApp { background-color: #ffe4ec; }
        .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, h1, h2, h3 { color: #4a1d35 !important; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .rec-box { background-color: #fff8e1; padding: 15px; border-radius: 8px; border: 1px solid #f0c674; border-left: 5px solid #f59e0b; margin-top: 10px; }
        [data-testid="stFileUploaderDropzone"] { background-color: #f3faf0; border: 2px dashed #6a4c93; border-radius: 10px; }
        .stTextArea textarea, .stTextInput input { background-color: #f3faf0; border: 2px solid #6a4c93; border-radius: 10px; }
        
        .resume-paper { background: #FFFFFF; padding: 40px; border: 1px solid #D3D3D3; border-radius: 4px; font-family: 'Times New Roman', Times, serif; color: #111111; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
        .resume-section-head { font-family: 'Arial', sans-serif; font-size: 1.15em; font-weight: bold; color: #111111; border-bottom: 1.5px solid #111111; padding-bottom: 2px; margin-top: 22px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
        </style>
        """)

# --- 7. SIDEBAR MENU (LEFT HAND NAVIGATOR) ---
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Select Option:", ["🔍 Resume Analyzer", "📝 Build Resume", "⚙️ Settings / Profile"])

with st.sidebar:
    st.divider()
    if st.button("⬅️ Log Out / Back to Login"):
        st.session_state.logged_in = False
        st.rerun()
        
    uploaded_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        st.session_state.photo = uploaded_file
        
    if page == "📝 Build Resume":
        st.subheader("✍️ Resume Input Form")
        
        res_name = st.text_input("Full Name", value=st.session_state.user_name)
        res_title = st.text_input("Professional Title (e.g., Software Engineer)")
        res_email = st.text_input("Email Address", value=st.session_state.user_email)
        res_phone = st.text_input("Phone Number")
        res_linkedin = st.text_input("LinkedIn Profile")
        res_summary = st.text_area("Professional Summary")
        res_experience = st.text_area("Work Experience (Company, Role, Description, Years)")
        res_education = st.text_area("Education (Degree, College, Year)")
        res_skills = st.text_area("Skills (List your core skills)")
        
        st.divider()
        st.subheader("🎨 Formatting & Custom Theme")
        selected_template = st.selectbox("Layout Design:", ["Classic Minimalist", "Modern Executive"])
        selected_font = st.selectbox("Font Style:", ["Times New Roman", "Arial", "Georgia", "Courier New", "Verdana"])
        
        custom_font_size = st.number_input("Font Size (px):", min_value=10, max_value=24, value=15, step=1)
        font_size_css = f"{custom_font_size}px"
        
        user_theme_color = st.color_picker("Choose Layout Theme Color:", "#1e3d59")
        section_head_color = st.color_picker("Choose Section Heading Color:", "#6a4c93") 
        
    # **येथे 'About' आणि 'How it works' नेहमी डाव्या बाजूच्या नॅव्हिगेटरमध्ये (Sidebar) दिसेल**
    st.subheader("🔍 Resume Analyzer Info")
    image_url = "https://www.guiseppegetto.com/wp-content/uploads/2026/04/ChatGPT-Image-Apr-14-2026-10_26_00-AM.jpg" 
    try:
        st.image(image_url, caption="Optimize Your Application Strategy", use_container_width=True)
    except:
        pass
        
    st.header("ℹ️ About")
    st.info("💡 This tool helps you measure how your resume matches a job description and identify important keywords.")
    
    st.header("⚙️ How it works")
    st.markdown("""
    1. **Text Extraction:** Parses plain text smoothly directly from your uploaded PDF file.
    2. **Clean Text:** Removes grammatical standard English stopwords and symbols.
    3. **TF-IDF Analysis:** Evaluates exact technical keyword counts & high value phrase weights.
    4. **Cosine Match:** Mathematical formula score generation maps profile with targeted role.
    """)

# --- 8. PAGE ROUTING CONTROLS ---

# PAGE 1: SETTINGS
if page == "⚙️ Settings / Profile":
    st.title("⚙️ Settings & Profile")
    st.subheader("⚙️ App Settings")
    dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_theme)
    if dark_mode != st.session_state.dark_theme:
        st.session_state.dark_theme = dark_mode
        st.rerun()

    st.divider()
    st.subheader("👤 Profile Details")
    st.write(f"**Name:** {st.session_state.user_name}")
    st.write(f"**Email:** {st.session_state.user_email}")
    st.write(f"**Roll No:** {st.session_state.user_roll_no}")

# PAGE 2: RESUME BUILDER
elif page == "📝 Build Resume":
    st.title("📝 Resume Builder")
    st.write("Fill your details in the Left Sidebar Form; your resume will be rendered below in real-time:")
    

    # फोटो प्रोसेस करा
    photo_html = ""
    if 'photo' in st.session_state and st.session_state.photo is not None:
        bytes_data = st.session_state.photo.getvalue()
        base64_photo = base64.b64encode(bytes_data).decode()
        photo_html = f'<img src="data:image/png;base64,{base64_photo}" style="width:120px; height:120px; border-radius:10px; object-fit:cover; border: 3px solid #ccc;">'
 
 
    header_content = f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
        <div style="flex: 2;">
            <h1 style="margin: 0; color: {user_theme_color};">{res_name if res_name else 'YOUR NAME'}</h1>
            <p style="font-style: italic; margin: 5px 0; font-size: 1.2em;">{res_title if res_title else 'Professional Title'}</p>
            <p style="margin: 0; font-size: 0.95em;">📧 {res_email} | 📞 {res_phone if res_phone else 'Phone'} | 🔗 {res_linkedin if res_linkedin else 'LinkedIn'}</p>
        </div>
        <div style="flex: 0 0 130px; text-align: right;">
            {photo_html}
        </div>
    </div>
    """

    # थीमप्रमाणे रेंडर करा
    if selected_template == "Modern Executive":
       st.html(f"""
    <div class="modern-paper" style="background: #FFFFFF; padding: 35px; border: 3px solid {user_theme_color}; border-radius: 10px; font-family: '{selected_font}';">
        <div class="modern-header" style="background: {user_theme_color}; color: #ffffff; padding: 25px; margin: -35px -35px 25px -35px; border-radius: 4px 4px 0 0;">
            {header_content.replace(f'color: {user_theme_color};', 'color: #ffffff;')}
        </div>
        
        <div style="font-size: 1.1em; font-weight: bold; color: {section_head_color}; border-bottom: 2px solid {section_head_color}; padding-bottom: 3px; margin-top: 20px; margin-bottom: 8px; text-transform: uppercase;">
            Professional Summary
        </div>
        <p style='white-space: pre-line;'>{res_summary if res_summary else '...'}</p>
        
        <div style="font-size: 1.1em; font-weight: bold; color: {section_head_color}; border-bottom: 2px solid {section_head_color}; padding-bottom: 3px; margin-top: 20px; margin-bottom: 8px; text-transform: uppercase;">
            Work Experience
        <p style='white-space: pre-line;'>{res_experience if res_experience else '...'}</p>
        </div>
        <div style="font-size: 1.1em; font-weight: bold; color: {section_head_color}; border-bottom: 2px solid {section_head_color}; padding-bottom: 3px; margin-top: 20px; margin-bottom: 8px; text-transform: uppercase;">
           Education
        <p style='white-space: pre-line;'>{res_experience if res_experience else '...'}</p>
        </div>
        <div style="font-size: 1.1em; font-weight: bold; color: {section_head_color}; border-bottom: 2px solid {section_head_color}; padding-bottom: 3px; margin-top: 20px; margin-bottom: 8px; text-transform: uppercase;">
           Skills
        <p style='white-space: pre-line;'>{res_experience if res_experience else '...'}</p>
        </div>
        """)
    else:
      st.html(f"""
    <div class="resume-paper" style="background: #FFFFFF; padding: 40px; border: 1px solid #D3D3D3; border-radius: 4px; font-family: '{selected_font}', serif; color: #111111;">
        {header_content}
        
        <div class="section-head" style="font-family: 'Arial', sans-serif; font-size: 1.15em; font-weight: bold; color: {section_head_color}; border-bottom: 1.5px solid {section_head_color}; padding-bottom: 2px; margin-top: 22px; margin-bottom: 10px; text-transform: uppercase;">
            Professional Summary
        </div>
        <p style='white-space: pre-line;'>{res_summary if res_summary else '...'}</p>
        
        <div class="section-head" style="font-family: 'Arial', sans-serif; font-size: 1.15em; font-weight: bold; color: {section_head_color}; border-bottom: 1.5px solid {section_head_color}; padding-bottom: 2px; margin-top: 22px; margin-bottom: 10px; text-transform: uppercase;">
            Work Experience
        </div>
        <p style='white-space: pre-line;'>{res_experience if res_experience else '...'}</p>

        <div class="section-head" style="font-family: 'Arial', sans-serif; font-size: 1.15em; font-weight: bold; color: {section_head_color}; border-bottom: 1.5px solid {section_head_color}; padding-bottom: 2px; margin-top: 22px; margin-bottom: 10px; text-transform: uppercase;">
            Education
        </div>
        <p style='white-space: pre-line;'>{res_education if res_education else '...'}</p>

        <div class="section-head" style="font-family: 'Arial', sans-serif; font-size: 1.15em; font-weight: bold; color: {section_head_color}; border-bottom: 1.5px solid {section_head_color}; padding-bottom: 2px; margin-top: 22px; margin-bottom: 10px; text-transform: uppercase;">
            Skills
        </div>
        <p style='white-space: pre-line;'>{res_skills if res_skills else '...'}</p>
    </div>
    """)
            
    resume_download_text = f"""{res_name if res_name else 'YOUR NAME'}
    {res_title if res_title else 'Professional Title'}
📧  {res_email} | 📞 {res_phone if res_phone else 'Phone'} | 🔗 {res_linkedin if res_linkedin else 'LinkedIn'}

    
==================================================
PROFESSIONAL SUMMARY
==================================================
{res_summary if res_summary else ''}

==================================================
WORK EXPERIENCE
==================================================
{res_experience if res_experience else ''}

==================================================
EDUCATION
==================================================
{res_education if res_education else ''}

==================================================
SKILLS
==================================================
{res_skills if res_skills else ''}
"""
    
    
    pdf_data = create_pdf(res_name, res_title, res_email, res_phone, res_linkedin, res_summary, res_experience, res_education, res_skills)
    st.download_button(
        label="📥 Download Resume as PDF",
        data=pdf_data,
        file_name=f"Resume_{res_name.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )


# PAGE 3: RESUME ANALYZER
elif page == "🔍 Resume Analyzer":
    st.title("🔍 Resume Analyzer")
    st.write("Optimize Your Application Strategy using **TF-IDF + Cosine Similarity**.")
    
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
    job_description = st.text_area("Paste the job description", height=200)
    
    if st.button("Analyze Match", icon=":material/analytics:"):
        if not uploaded_file:
            st.warning("Please upload your resume.")
        elif not job_description.strip():
            st.warning("Please paste the job description.")
        else:
            with st.spinner("Analyzing your resume..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if not resume_text.strip():
                    st.error("Could not extract text from PDF. Please check the file.")
                else:
                    similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)
                    st.success("Analysis Complete! 🎉")
                    
                    st.subheader("🎯 Results")
                    
                    if similarity_score >= 75:
                        progress_color = "green"
                        status_text = "🎯 Excellent Match! Your resume strongly aligns."
                        st.balloons()
                    elif similarity_score >= 40:
                        progress_color = "orange"
                        status_text = "👍 Good Match. Your resume aligns fairly well, but could use more keywords."
                    else:
                        progress_color = "red"
                        status_text = "⚠️ Low Match. Consider tailoring your resume heavily to match the description."

                    st.metric("Match Score", f"{similarity_score}%")
                    st.progress(int(similarity_score) / 100)
                    st.markdown(f"<p style='font-weight:bold; color:{progress_color};'>{status_text}</p>", unsafe_allow_html=True)
                    
                    # Chart setup
                    fig, ax = plt.subplots(figsize=(6, 1.3))
                    bg_color = "#262730" if st.session_state.dark_theme else "#b0e6f4"
                    text_color = "#FAFAFA" if st.session_state.dark_theme else "#020C3F"
                    
                    fig.patch.set_facecolor(bg_color)
                    fig.patch.set_edgecolor(text_color)
                    fig.patch.set_linewidth(2)
                    ax.set_facecolor("#ffffff" if not st.session_state.dark_theme else "#1E222B")
                    
                    colors = ["#c41212ac", "#ed630dae", "#17db5bb3"]
                    if similarity_score < 40:
                        color_index = 0
                    elif similarity_score < 75:
                        color_index = 1
                    else:
                        color_index = 2
                    
                    ax.barh([0], [similarity_score], color=colors[color_index], height=0.4)
                    ax.set_title("Resume Job Match Summary", fontsize=12, color=text_color)
                    ax.set_xlabel("Match percentage", fontsize=9, color=text_color)
                    ax.set_xlim(0, 100)
                    ax.set_xticks([0, 20, 40, 60, 80, 100])
                    ax.tick_params(colors=text_color)
                    ax.set_yticks([])
                    st.pyplot(fig)
                    
                    # Keywords Analysis
                    job_words_list = [w for w in job_processed.split() if len(w) > 2]
                    resume_words_list = [w for w in resume_processed.split() if len(w) > 2]
                    
                    job_word_counts = Counter(job_words_list)
                    resume_word_counts = Counter(resume_words_list)
                    top_job_keywords = [word for word, count in job_word_counts.most_common(7)]
                    
                    if top_job_keywords:
                        st.subheader("📊 Keyword Frequency Analysis")
                        chart_data = {
                            'Keywords': top_job_keywords,
                            'Job Description': [job_word_counts[word] for word in top_job_keywords],
                            'Your Resume': [resume_word_counts[word] for word in top_job_keywords]
                        }
                        df_chart = pd.DataFrame(chart_data).set_index('Keywords')
                        st.bar_chart(df_chart)
                    
                    missing_keywords = [word for word in top_job_keywords if resume_word_counts[word] == 0]
                    
                    st.subheader("🔑 Key Recommendations")
                    if missing_keywords:
                        keyword_color = "#64B5F6" if st.session_state.dark_theme else "#1565c0"
                        st.markdown(
                            f"""<div class='rec-box'>
                            <strong>💡 Missing Keywords Found:</strong><br>
                            Consider adding relevant instances of these key terms from the job post to boost your visibility:
                            <p style='margin-top:8px; font-weight:600; color:{keyword_color};'> {", ".join(missing_keywords)}</p>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.success("Awesome! You have targeted all the central core keywords of this job description in your resume.")
                    
                    if similarity_score < 75:
                        st.info("💡 **Pro-Tip to boost score:** ATS systems love powerful operational verbs, Consider rephrasing your experience section using verbs like: *Spearheaded, Formulated, Executed, Streamlined, Orchestrated, or Maximized*.")
