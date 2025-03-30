from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS 
import fitz
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

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

@app.route('/upload', methods=['POST'])
def upload_pdf():
    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(file_path)

    # Return the extracted text as a JSON response
    return jsonify({"extracted_text": text})


def recommend_jobs(user_skills, job_openings):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(job_openings)

    num_clusters = 2 #2 clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(tfidf_matrix)

    centroids = kmeans.cluster_centers_

    user_tfidf = vectorizer.transform([user_skills])
    cluster_similarities = cosine_similarity(user_tfidf, centroids)

    most_similar_cluster = cluster_similarities.argmax()

    cluster_jobs_indices = [i for i, label in enumerate(kmeans.labels_) if label == most_similar_cluster]

    recommended_jobs = [job_openings[i] for i in cluster_jobs_indices]
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
