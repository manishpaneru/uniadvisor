import streamlit as st
import groq
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import json
import traceback

# Load environment variables
load_dotenv()

# Configure Groq client with minimal settings
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = groq.Groq(api_key=api_key)

def get_course_information(university, course):
    """
    Get detailed course information using OpenAI API
    """
    system_message = """You are a comprehensive course information bot. Return a detailed JSON object.
    Focus on providing extensive, well-researched information about the university course.
    Include specific details about curriculum, learning outcomes, and career opportunities."""
    
    prompt = f"""Generate a detailed JSON object about the {course} at {university}. 
    Provide comprehensive information including curriculum details, career paths, and student experiences.
    Return ONLY a JSON object with this exact structure:
    {{
        "course_overview": "Detailed 3-4 paragraph overview including: course structure, 
        learning objectives, unique features, accreditation, and industry partnerships",
        "tuition": {{
            "per_semester": "amount in AUD with breakdown",
            "full_course": "total amount in AUD with additional costs"
        }},
        "faculty_reviews": [
            "detailed review 1 with specific feedback about teaching quality",
            "detailed review 2 with specific feedback about support",
            "detailed review 3 with specific feedback about expertise",
            "detailed review 4 with specific feedback about industry connection"
        ],
        "alumni_reviews": [
            "detailed review 1 with career outcome",
            "detailed review 2 with specific skills gained",
            "detailed review 3 with industry placement",
            "detailed review 4 with international opportunities"
        ],
        "career_prospects": [
            "detailed career path 1 with average time to position",
            "detailed career path 2 with industry demand",
            "detailed career path 3 with specific companies",
            "detailed career path 4 with advancement opportunities",
            "detailed career path 5 with global opportunities"
        ],
        "skills_taught": [
            "detailed skill 1 with industry application",
            "detailed skill 2 with practical examples",
            "detailed skill 3 with certification options",
            "detailed skill 4 with project work",
            "detailed skill 5 with industry relevance",
            "detailed skill 6 with future trends"
        ],
        "average_earnings": {{
            "starting": "detailed starting salary range in AUD with industry comparison",
            "mid_career": "detailed mid-career salary range in AUD with progression timeline"
        }}
    }}"""

    try:
        chat_completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4096
        )
        
        response_content = chat_completion.choices[0].message.content.strip()
        
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing course information JSON: {str(e)}")
            st.error("Full response content:")
            st.code(response_content, language="json")
            return None
            
    except Exception as e:
        st.error(f"Error fetching course information: {str(e)}")
        return None

def get_university_info(university):
    """
    Get detailed information about the university using OpenAI API
    """
    system_message = """You are a comprehensive university information bot. Return a detailed JSON object.
    Focus on providing extensive, well-researched information about the university.
    Include historical context, notable achievements, and specific details about facilities and programs."""
    
    prompt = f"""Generate a detailed JSON object about {university} in Australia.
    Provide comprehensive information including history, achievements, and specific details.
    Return a JSON object with this exact structure:
    {{
        "overview": "Provide a detailed 4-5 paragraph overview including: history and establishment, notable achievements, 
        international recognition, research impact, and current standing in the academic community",
        "location": {{
            "city": "city name",
            "state": "state name",
            "campus_description": "Detailed 2-3 paragraph description of the campus, including architecture, 
            layout, notable buildings, and surrounding area"
        }},
        "rankings": {{
            "world_rank": "current world ranking with source",
            "national_rank": "current national ranking with source",
            "subject_strengths": [
                "detailed strength 1 with ranking",
                "detailed strength 2 with ranking",
                "detailed strength 3 with ranking",
                "detailed strength 4 with ranking"
            ]
        }},
        "facilities": [
            "detailed facility 1 with specific features",
            "detailed facility 2 with specific features",
            "detailed facility 3 with specific features",
            "detailed facility 4 with specific features",
            "detailed facility 5 with specific features"
        ],
        "research": {{
            "focus_areas": [
                "detailed research area 1 with achievements",
                "detailed research area 2 with achievements",
                "detailed research area 3 with achievements",
                "detailed research area 4 with achievements"
            ],
            "achievements": [
                "major achievement 1 with year and impact",
                "major achievement 2 with year and impact",
                "major achievement 3 with year and impact"
            ]
        }},
        "student_life": {{
            "total_students": "exact number with breakdown",
            "international_students": "percentage with top 3 countries",
            "clubs_societies": "number with description of notable ones",
            "accommodation": "detailed description of housing options and facilities"
        }}
    }}"""

    try:
        chat_completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4096
        )
        
        response_content = chat_completion.choices[0].message.content.strip()
        
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing university information JSON: {str(e)}")
            st.error("Full response content:")
            st.code(response_content, language="json")
            return None
            
    except Exception as e:
        st.error(f"Error fetching university information: {str(e)}")
        return None

def process_salary(salary):
    """
    Process salary value to extract numeric values from various formats:
    - Pure numbers: 60000
    - Formatted numbers: "$60,000 AUD"
    - Ranges: "60000 - 80000"
    - Descriptive text: "The starting salary range is 60000 to 80000 AUD"
    Returns the average if a range is provided
    """
    if isinstance(salary, (int, float)):
        return float(salary)
    elif isinstance(salary, str):
        try:
            # Remove currency symbols and 'AUD'
            cleaned = salary.replace('$', '').replace('AUD', '').replace(',', '').strip()
            
            # Extract all numbers from the string
            import re
            numbers = re.findall(r'\d+', cleaned)
            
            if numbers:
                # If we have multiple numbers (likely a range), take the average
                numbers = [float(num) for num in numbers]
                return sum(numbers) / len(numbers)
            else:
                raise ValueError(f"No numeric values found in salary: {salary}")
        except Exception as e:
            st.error(f"Error processing salary '{salary}': {str(e)}")
            return 0
    else:
        st.error(f"Unexpected salary format: {salary}")
        return 0

def display_earnings_chart(earnings_data):
    """
    Display earnings chart using Plotly
    """
    try:
        starting_salary = process_salary(earnings_data['starting'])
        mid_career_salary = process_salary(earnings_data['mid_career'])

        # Only create chart if we have valid salary data
        if starting_salary > 0 and mid_career_salary > 0:
            fig = go.Figure(data=[
                go.Bar(
                    x=['Starting Salary', 'Mid-Career Salary'],
                    y=[starting_salary, mid_career_salary],
                    marker_color=['#1e3c72', '#2a5298'],
                    text=[f'${starting_salary:,.0f}', f'${mid_career_salary:,.0f}'],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Average Salary Progression',
                yaxis_title='Salary (AUD)',
                showlegend=False,
                yaxis=dict(tickformat='$,.0f'),
                height=400,
                margin=dict(t=50, b=50),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the original salary descriptions
            with st.expander("📊 Detailed Salary Information"):
                st.write("**Starting Salary:**")
                st.write(earnings_data['starting'])
                st.write("**Mid-Career Salary:**")
                st.write(earnings_data['mid_career'])
        else:
            st.warning("Unable to display salary chart due to invalid salary data")
            st.write("**Original Salary Information:**")
            st.write(earnings_data)
            
    except Exception as e:
        st.error(f"Error displaying earnings chart: {str(e)}")
        st.write("**Original Salary Data:**")
        st.write(earnings_data)

def main():
    st.set_page_config(page_title="Australian Universities Information", layout="wide")
    
    # Custom CSS with enhanced styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
        
        :root {
            --primary-color: #1e3c72;
            --secondary-color: #2a5298;
            --accent-color: #4CAF50;
            --background-color: #f8f9fa;
            --card-background: #ffffff;
            --text-color: #2c3e50;
            --border-radius: 12px;
            --box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        * {
            font-family: 'Inter', sans-serif;
            color: var(--text-color);
        }
        
        h1, h2, h3, .section-header {
            font-family: 'Poppins', sans-serif !important;
        }
        
        /* Main header styling */
        .main-header {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            border-radius: var(--border-radius);
            margin-bottom: 2rem;
            box-shadow: var(--box-shadow);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
            pointer-events: none;
        }
        
        .main-header h1 {
            font-size: 3.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .main-header p {
            font-size: 1.4rem !important;
            color: rgba(255, 255, 255, 0.9) !important;
            max-width: 800px;
            margin: 0 auto !important;
        }
        
        /* Card styling */
        .card {
            background: var(--card-background);
            padding: 1.8rem;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Metric card styling */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.8rem;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            transition: all 0.3s ease;
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Section headers */
        .section-header {
            color: var(--primary-color);
            padding: 1rem 0;
            border-bottom: 2px solid rgba(30, 60, 114, 0.1);
            margin: 2rem 0 1.5rem 0;
            font-size: 2rem !important;
            font-weight: 600 !important;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Form styling */
        .stTextInput input {
            font-size: 1.1rem !important;
            padding: 0.8rem 1rem !important;
            border-radius: var(--border-radius) !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease;
        }
        
        .stTextInput input:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(30, 60, 114, 0.1) !important;
        }
        
        .stButton button {
            font-size: 1.2rem !important;
            padding: 0.8rem 1.5rem !important;
            font-weight: 600 !important;
            border-radius: var(--border-radius) !important;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
            transition: all 0.3s ease !important;
            border: none !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        }
        
        /* Overview text styling */
        .overview-text {
            font-size: 1.2rem !important;
            line-height: 1.8 !important;
            padding: 2rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: var(--border-radius);
            margin-bottom: 2rem;
            border-left: 4px solid var(--primary-color);
            box-shadow: var(--box-shadow);
        }
        
        /* Alert styling */
        .stAlert {
            font-size: 1.1rem !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            border-radius: var(--border-radius) !important;
            animation: slideIn 0.5s ease-out;
        }
        
        /* Statistics cards */
        .stat-card {
            text-align: center;
            padding: 1.5rem;
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            transition: all 0.3s ease;
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .stat-value {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: var(--primary-color) !important;
            margin-bottom: 0.5rem !important;
        }
        
        .stat-label {
            font-size: 1.1rem !important;
            color: #666 !important;
            font-weight: 500 !important;
        }
        
        /* Bullet points */
        .bullet-point {
            font-size: 1.1rem !important;
            line-height: 1.8 !important;
            margin-bottom: 0.8rem !important;
            padding-left: 1.5rem !important;
            position: relative;
        }
        
        .bullet-point::before {
            content: "•";
            color: var(--primary-color);
            font-size: 1.5rem;
            position: absolute;
            left: 0;
            top: -0.2rem;
        }
        
        /* Animations */
        @keyframes slideIn {
            from {
                transform: translateY(20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: var(--primary-color) !important;
            background-color: rgba(30, 60, 114, 0.05) !important;
            border-radius: var(--border-radius) !important;
            transition: all 0.3s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: rgba(30, 60, 114, 0.1) !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary-color);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--secondary-color);
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border-color: var(--primary-color) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🎓 Manish Paneru's AID to Australian Universities")
    st.write("Comprehensive information about courses and universities in Australia")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input form
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            university = st.text_input("🏛️ University Name", placeholder="e.g., University of Sydney")
        
        with col2:
            course = st.text_input("📚 Course Name", placeholder="e.g., Bachelor of Computer Science")
        
        submit_button = st.form_submit_button("🔍 Get Information", use_container_width=True)
    
    if submit_button and university:
        # Fetch university information
        with st.spinner("🔄 Fetching university information..."):
            uni_info = get_university_info(university)
            
            if uni_info:
                # University Overview Section
                st.markdown('<h2 class="section-header">🏛️ University Overview</h2>', unsafe_allow_html=True)
                st.markdown(f'<div class="overview-text">{uni_info["overview"]}</div>', unsafe_allow_html=True)
                
                # Location and Rankings
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<h3 style="color: #1e3c72;">📍 Location</h3>', unsafe_allow_html=True)
                    st.write(f"**City:** {uni_info['location']['city']}")
                    st.write(f"**State:** {uni_info['location']['state']}")
                    st.write(f"**Campus:** {uni_info['location']['campus_description']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<h3 style="color: #1e3c72;">🏆 Rankings</h3>', unsafe_allow_html=True)
                    st.write(f"**World Rank:** {uni_info['rankings']['world_rank']}")
                    st.write(f"**National Rank:** {uni_info['rankings']['national_rank']}")
                    st.write("**Subject Strengths:**")
                    for subject in uni_info['rankings']['subject_strengths']:
                        st.write(f"• {subject}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Student Life Statistics
                st.markdown('<h2 class="section-header">👥 Student Life</h2>', unsafe_allow_html=True)
                stats_cols = st.columns(4)
                
                with stats_cols[0]:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-value">{uni_info["student_life"]["total_students"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">Total Students</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with stats_cols[1]:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-value">{uni_info["student_life"]["international_students"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">International Students</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with stats_cols[2]:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-value">{uni_info["student_life"]["clubs_societies"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">Clubs & Societies</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with stats_cols[3]:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown('<div class="stat-value">24/7</div>', unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">Campus Support</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Facilities and Research
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<h3 style="color: #1e3c72;">🏢 Facilities</h3>', unsafe_allow_html=True)
                    for facility in uni_info['facilities']:
                        st.write(f" {facility}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<h3 style="color: #1e3c72;">🔬 Research Focus</h3>', unsafe_allow_html=True)
                    for area in uni_info['research']['focus_areas']:
                        st.write(f"• {area}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Fetch course information if course is specified
        if course:
            with st.spinner("🔄 Fetching course information..."):
                course_info = get_course_information(university, course)
                
                if course_info:
                    st.markdown('<h2 class="section-header">📋 Course Details</h2>', unsafe_allow_html=True)
                    st.markdown(f'<div class="overview-text">{course_info["course_overview"]}</div>', unsafe_allow_html=True)
                    
                    # Tuition Costs
                    st.markdown('<h2 class="section-header">💰 Tuition Costs</h2>', unsafe_allow_html=True)
                    cost_col1, cost_col2 = st.columns(2)
                    with cost_col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Per Semester", course_info['tuition']['per_semester'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    with cost_col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Full Course", course_info['tuition']['full_course'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Reviews
                    rev_col1, rev_col2 = st.columns(2)
                    
                    with rev_col1:
                        st.markdown('<h2 class="section-header">👨‍🏫 Faculty Reviews</h2>', unsafe_allow_html=True)
                        for review in course_info['faculty_reviews']:
                            st.info(f"💬 {review}")
                    
                    with rev_col2:
                        st.markdown('<h2 class="section-header">👨‍🎓 Alumni Reviews</h2>', unsafe_allow_html=True)
                        for review in course_info['alumni_reviews']:
                            st.success(f"💬 {review}")
                    
                    # Career and Skills
                    career_col1, career_col2 = st.columns(2)
                    
                    with career_col1:
                        st.markdown('<h2 class="section-header">🎯 Career Prospects</h2>', unsafe_allow_html=True)
                        for prospect in course_info['career_prospects']:
                            st.markdown(f'<div class="bullet-point">• {prospect}</div>', unsafe_allow_html=True)
                    
                    with career_col2:
                        st.markdown('<h2 class="section-header">🔧 Skills Taught</h2>', unsafe_allow_html=True)
                        for skill in course_info['skills_taught']:
                            st.markdown(f'<div class="bullet-point">• {skill}</div>', unsafe_allow_html=True)
                    
                    # Earnings
                    st.markdown('<h2 class="section-header">💸 Average Earnings</h2>', unsafe_allow_html=True)
                    display_earnings_chart(course_info['average_earnings'])
                else:
                    st.error("❌ Failed to fetch course information. Please try again.")

if __name__ == "__main__":
    main() 