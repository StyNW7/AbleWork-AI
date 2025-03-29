from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS 
import fitz
import os
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



if __name__ == '__main__':
    app.run(debug=True, port=5000)
