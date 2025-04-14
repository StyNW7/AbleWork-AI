from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS 
import fitz
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
import re

app = Flask(__name__)

api_key = os.getenv("HF_API_KEY")

client = InferenceClient(
    provider="nebius",
    api_key=api_key,
)

CORS(app)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    
    return text

# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     # Check if a file is part of the request
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files['file']
    
#     # If no file is selected
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
#     # Save the file to the server
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(file_path)

#     # Extract text from the uploaded PDF
#     text = extract_text_from_pdf(file_path)

#     # Return the extracted text as a JSON response
#     return jsonify({"extracted_text": text})

def handle_file_upload(file):

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(file_path)

    return text


def recommend_jobs(user_skills, job_openings, similarity_threshold=0.3):
    vectorizer = TfidfVectorizer(stop_words='english')
    
    job_requirements = [job['requirement'] for job in job_openings]
    
    tfidf_matrix = vectorizer.fit_transform(job_requirements)
    user_tfidf = vectorizer.transform([user_skills])
    
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    
    recommended_jobs = [
        {
            "job_id": job_openings[i]["job_id"],
            "requirement": job_openings[i]["requirement"],
            "similarity": similarities[0][i] #new
        } 
        for i in range(len(similarities[0])) if similarities[0][i] > similarity_threshold
    ]
    
    return recommended_jobs

@app.route('/recommend_jobs', methods=['POST'])
def recommend_jobs_route():
    data = request.get_json()
    user_skills = data.get('user_skills', '')
    job_openings = data.get('job_openings', [])

    if not user_skills or not job_openings:
        return jsonify({"error": "Both user_skills and job_openings must be provided"}), 400

    recommended_jobs = recommend_jobs(user_skills, job_openings)

    return jsonify({"recommended_jobs": recommended_jobs})

@app.route('/review_cv', methods=['POST'])
def ask():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
    
        extracted_text = handle_file_upload(file)
        # extracted_text = request.json.get('extracted_text')

        if not extracted_text:
            return jsonify({"error": "No question provided"}), 400

        user_message = f'''
        Instruction: Please review the following CV for clarity, formatting, and overall effectiveness. Your feedback should focus on the content, highlighting areas for improvement such as spelling errors, missing or vague information, unclear sections, and suggestions for enhancing readability. You do not need to evaluate the CV's appearance (e.g., layout, bullet points, or spacing), only the text itself.

        Context: You are a career advice expert tasked with providing constructive feedback on this CV. Please focus on improving the clarity, conciseness, and impact of the content. The following is the text content of the CV:
        {extracted_text}
        
        Output Format: You only output 2 sections: the strengths and areas for improvement. The output must be in the following exact JSON format:
        "strengths": [
            "strength 1.",
            "strength 2.",
            "strength 3."
        ],
        "improvements": [
            "improvement 1.",
            "improvement 2.",
            "improvement 3."
        ]

        Make sure that your response strictly adheres to this structure. Do not include any additional text or explanations. Only output the JSON. 
        '''

        completion = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct",
            messages=[{
                "role": "user",
                "content": user_message
            }],
            max_tokens=500
        )

        answer = completion.choices[0].message.content
        json_pattern = r'\{(?:[^{}]*|\{(?:[^{}]*|\{[^{}]*\})*\})*\}'

        match = re.search(json_pattern, answer)

        if not match:
            return jsonify({"error": "No valid JSON found in the response"}), 400

        json_string = match.group(0).strip()

        try:
            response_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return jsonify({"error": "Failed to parse model response", "details": str(e)}), 500

        return jsonify(response_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Hello, Team Overclock!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)



