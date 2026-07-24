# main.py
# A powerful Resume Matching application using a hybrid approach:
# - NLP (spaCy) for semantic job description matching.
# - Robust Regex for hard filters (experience, education).
#
# To run this application:
# 1. Make sure you have Python installed.
# 2. Install the required libraries:
#    pip install Flask PyPDF2 python-docx spacy
# 3. Download the spaCy English model:
#    python -m spacy download en_core_web_md
# 4. Save this code as a Python file (e.g., app.py).
# 5. Run it from your terminal:
#    python app.py
# 6. Open your web browser and go to http://127.0.0.1:5000

import os
import re
import shutil
import PyPDF2
import docx
import spacy
from flask import Flask, request, render_template_string, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'resume_uploads'
MATCH_FOLDER = 'matching_resumes'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, UPLOAD_FOLDER)
app.config['MATCH_FOLDER'] = os.path.join(BASE_DIR, MATCH_FOLDER)
app.config['SECRET_KEY'] = 'supersecretkey' # Needed for flashing messages

# --- NLP Model Loading ---
try:
    NLP_MODEL = spacy.load('en_core_web_md')
    print("spaCy model 'en_core_web_md' loaded successfully.")
except OSError:
    print("Error: spaCy model 'en_core_web_md' not found.")
    print("Please run: python -m spacy download en_core_web_md")
    NLP_MODEL = None

# Create the necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MATCH_FOLDER'], exist_ok=True)


# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def extract_text_from_file(file_path, filename):
    """Extracts text from a file based on its extension."""
    extension = filename.rsplit('.', 1)[1].lower()
    if extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif extension == 'docx':
        return extract_text_from_docx(file_path)
    return ""

# CHANGED: find_experience rewritten to also catch "yrs"/"yr" abbreviations
# and "X-Y years" ranges (taking the upper bound), on top of the original
# "X years"/"X months" patterns.
def find_experience(text):
    """
    Finds the maximum years of experience mentioned in the text.

    - Ranges like "3-5 years" contribute their upper bound (5), not the
      lower bound or a naive sum.
    - "Total Experience: 4 yrs" and "4+ yrs exp" style abbreviations are
      matched in addition to the full "years"/"year" word.
    - If nothing is found, this returns 0.0 rather than raising or leaving
      the value undefined. That 0.0 is intentional, not an error state: a
      resume with no detectable experience text is treated the same as one
      stating "0 years" of experience. Combined with a default
      min_experience of 0 in the caller, such resumes are NOT excluded by
      the experience filter unless the caller explicitly raises
      min_experience above 0 — this is existing, deliberate behavior,
      documented here so it isn't mistaken for a bug.
    """
    text_lower = text.lower()
    candidates = []

    # "3-5 years" / "3 - 5 yrs" — take the upper bound of the range.
    for lo, hi in re.findall(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*\+?\s*(?:years?|yrs?)\b', text_lower):
        candidates.append(float(hi))

    # Plain "X years", "X+ yrs", "Total Experience: X yrs" style mentions.
    # (This also re-matches the upper bound of any range caught above, e.g.
    # the "5" in "3-5 years" — harmless, since we only ever take the max.)
    for num in re.findall(r'(\d+\.?\d*)\s*\+?\s*(?:years?|yrs?)\b', text_lower):
        candidates.append(float(num))

    total_experience = max(candidates) if candidates else 0.0

    month_matches = re.findall(r'(\d+)\s*\+?\s*months?', text_lower)
    if month_matches:
        total_experience += max(float(num) for num in month_matches) / 12.0

    return round(total_experience, 1)

def find_education(text):
    """
    Finds educational qualifications using robust regex patterns.
    """
    education_found = set()
    text_lower = text.lower()
    
    ug_patterns = [
        r'bachelor\s*of\s*technology', r'bachelor\s*of\s*engineering',
        r'\b(b\.?e\.?|b\.?tech\.?|b\.?sc\.?|bca|bba)\b',
        r'\b(bachelor|undergraduate|diploma)\b'
    ]
    masters_patterns = [
        r'master\s*of\s*science', r'master\s*of\s*technology',
        r'\b(m\.?s\.?|m\.?e\.?|m\.?tech\.?|mca|mba)\b',
        r'\b(master|postgraduate|pg\s*diploma)\b'
    ]
    pg_patterns = [
        r'\b(ph\.?d|doctorate|doctoral)\b'
    ]

    if any(re.search(pattern, text_lower) for pattern in ug_patterns):
        education_found.add('UG')
    if any(re.search(pattern, text_lower) for pattern in masters_patterns):
        education_found.add('Masters')
    if any(re.search(pattern, text_lower) for pattern in pg_patterns):
        education_found.add('PG')
        
    return education_found

# CHANGED: new helper for Feature 1 (up to 10 keyword/requirement fields).
def find_keyword_matches(text, keywords):
    """
    Checks which of the given keyword/requirement phrases appear in the
    resume text. Matching is case-insensitive and requires an exact match,
    not a partial substring - e.g. "SQL" must not match inside "MySQL".

    This uses alphanumeric-boundary lookaround (`(?<![A-Za-z0-9])` /
    `(?![A-Za-z0-9])`) rather than `\\b`. Plain `\\b` treats "_" as a word
    character (so "SQL" would still wrongly match inside "SQL_DB", though
    that's not the reported case) and, more importantly, breaks for
    keywords whose own edge character isn't a word character at all -
    e.g. `\\bC\\+\\+\\b` fails to match a legitimate "C++" followed by a
    space, because "+" is non-word on both sides and no word/non-word
    transition exists there. Alphanumeric-boundary lookaround only cares
    about whether the character immediately outside the match is a letter
    or digit, so it correctly rejects "SQL" inside "MySQL" while still
    matching symbol-containing keywords like "C++", "Node.js", or ".NET".

    Returns the subset of `keywords` that were found, in the order they
    were submitted.
    """
    if not keywords:
        return []
    matched = []
    for kw in keywords:
        pattern = r'(?<![A-Za-z0-9])' + re.escape(kw) + r'(?![A-Za-z0-9])'
        if re.search(pattern, text, re.IGNORECASE):
            matched.append(kw)
    return matched

def grade_resume_nlp(resume_text, criteria, nlp_model, similarity_threshold=0.60):
    """
    Grades a single resume against criteria using NLP for semantic similarity.
    """
    requirements_text = criteria.get("requirements_text")
    if not requirements_text or not nlp_model:
        return {"passed": False, "similarity_score": 0.0}

    resume_doc = nlp_model(resume_text, disable=['parser', 'ner'])
    requirements_doc = nlp_model(requirements_text, disable=['parser', 'ner'])
    
    score = resume_doc.similarity(requirements_doc)

    if score >= similarity_threshold:
        return {"passed": True, "similarity_score": round(score, 3)}
    else:
        return {"passed": False, "similarity_score": round(score, 3)}


# --- HTML & CSS Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Resume Matcher</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .card {
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        /* Toggle Switch Styles */
        .toggle-bg:after {
            content: '';
            @apply absolute top-0.5 left-0.5 bg-white border border-gray-300 rounded-full h-5 w-5 transition shadow-sm;
        }
        input:checked + .toggle-bg:after {
            transform: translateX(100%);
            @apply border-white;
        }
        input:checked + .toggle-bg {
            @apply bg-indigo-600;
        }
        #criteria-filters fieldset:disabled {
            opacity: 0.5;
            pointer-events: none;
        }
    </style>
</head>
<body class="bg-gray-50 text-gray-800">

    <div class="container mx-auto px-4 py-8 md:py-12">
        <header class="text-center mb-10">
            <h1 class="text-4xl md:text-5xl font-bold text-gray-900">Intelligent Resume Matching System</h1>
            <p class="text-lg text-gray-600 mt-2">Leverage NLP to find the best candidates based on semantic similarity.</p>
        </header>

        <div class="max-w-4xl mx-auto bg-white p-8 rounded-2xl shadow-lg border border-gray-200">
            <form action="/" method="post" enctype="multipart/form-data">
                {% with messages = get_flashed_messages(with_categories=true) %}
                  {% if messages %}
                    {% for category, message in messages %}
                      <div class="mb-4 p-4 rounded-md {{ 'bg-red-100 text-red-700' if category == 'error' else 'bg-blue-100 text-blue-700' }}" role="alert">
                        {{ message }}
                      </div>
                    {% endfor %}
                  {% endif %}
                {% endwith %}

                <div class="mb-6">
                    <label class="block text-lg font-semibold text-gray-700 mb-2">1. Upload Resumes</label>
                    <div class="mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                        <div class="space-y-1 text-center">
                            <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true"><path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
                            <div class="flex text-sm text-gray-600">
                                <label for="resumes" class="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                                    <span>Select files or a folder</span>
                                    <input id="resumes" name="resumes" type="file" class="sr-only" multiple webkitdirectory directory>
                                </label>
                            </div>
                            <p class="text-xs text-gray-500">PDF, DOCX up to 10MB each</p>
                        </div>
                    </div>
                    <div id="file-list" class="mt-3 text-sm text-gray-600"></div>
                </div>

                <div class="mb-6 border-t pt-6">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="text-lg font-semibold text-gray-700">2. Hard Filters (Optional)</h3>
                        <div class="flex items-center">
                            <span class="text-sm font-medium text-gray-900 mr-3" id="toggle-label">Filters Disabled</span>
                            <input type="checkbox" id="criteria-toggle" name="criteria_toggle" class="sr-only" {% if request.form.get('criteria_toggle') %}checked{% endif %}>
                            <label for="criteria-toggle" class="relative inline-flex items-center cursor-pointer">
                                <span class="w-11 h-6 bg-gray-200 rounded-full toggle-bg"></span>
                            </label>
                        </div>
                    </div>
                    <fieldset id="criteria-filters">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <label for="experience" class="block text-sm font-medium text-gray-700">Minimum Experience (Years)</label>
                                <input type="number" name="experience" id="experience" min="0" step="0.5" class="mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" placeholder="e.g., 3" value="{{ request.form.get('experience', '') }}">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Required Education</label>
                                <div class="mt-2 space-y-2">
                                    <div class="flex items-start"><div class="flex items-center h-5"><input id="ug" name="education" type="checkbox" value="UG" class="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded" {% if 'UG' in request.form.getlist('education') %}checked{% endif %}></div><div class="ml-3 text-sm"><label for="ug" class="font-medium text-gray-700">Undergraduate (UG)</label></div></div>
                                    <div class="flex items-start"><div class="flex items-center h-5"><input id="masters" name="education" type="checkbox" value="Masters" class="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded" {% if 'Masters' in request.form.getlist('education') %}checked{% endif %}></div><div class="ml-3 text-sm"><label for="masters" class="font-medium text-gray-700">Masters</label></div></div>
                                    <div class="flex items-start"><div class="flex items-center h-5"><input id="pg" name="education" type="checkbox" value="PG" class="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded" {% if 'PG' in request.form.getlist('education') %}checked{% endif %}></div><div class="ml-3 text-sm"><label for="pg" class="font-medium text-gray-700">Post Graduate (PG/Doctorate)</label></div></div>
                                </div>
                            </div>
                        </div>
                    </fieldset>
                </div>

                <!-- CHANGED: Feature 1 - up to 10 keyword/requirement fields -->
                <div class="mb-6 border-t pt-6">
                    <label class="block text-lg font-semibold text-gray-700 mb-2">3. Keywords / Requirements (Optional, max 10)</label>
                    <p class="text-sm text-gray-500 mb-4">Add up to 10 short keywords or requirement phrases (not full sentences). Each resume is checked for how many of these it mentions, reported alongside the similarity score.</p>
                    {% set kw_values = request.form.getlist('keywords') %}
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {% for i in range(10) %}
                        <input type="text" name="keywords" maxlength="100" placeholder="Keyword {{ i + 1 }} (e.g., Python)" class="focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" value="{{ kw_values[i] if i < kw_values|length else '' }}">
                        {% endfor %}
                    </div>
                </div>

                <div class="mb-6 border-t pt-6">
                    <label for="requirements" class="block text-lg font-semibold text-gray-700 mb-2">4. Job Description</label>
                     <p class="text-sm text-gray-500 mb-2">Enter the full job description below. The NLP model will compare resumes against this text.</p>
                    <textarea id="requirements" name="requirements" rows="8" class="w-full p-4 text-sm border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., We are looking for an experienced Python programmer with a background in SQL and leadership skills..." required>{{ request.form.get('requirements', '') }}</textarea>
                </div>

                <div class="mb-8">
                    <label for="threshold" class="block text-lg font-semibold text-gray-700">5. Similarity Threshold</label>
                    <p class="text-sm text-gray-500 mb-2">Set the minimum similarity score (0.0 to 1.0) to consider a resume as a match. Higher is stricter.</p>
                    <input type="range" id="threshold" name="threshold" min="0.1" max="1.0" step="0.05" value="{{ request.form.get('threshold', '0.6') }}" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
                    <div class="text-center mt-1 font-medium text-indigo-600" id="threshold-value">0.6</div>
                </div>

                <div><button type="submit" class="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-lg font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors">Find Matching Resumes</button></div>
            </form>
        </div>

        {% if results is not none %}
            <div class="max-w-4xl mx-auto mt-12">
                <h2 class="text-3xl font-bold text-center mb-8">Analysis Results</h2>
                {% if results %}
                    <p class="text-center text-gray-600 mb-6">Found {{ results|length }} matching resume(s) above the similarity threshold.</p>
                    <div class="space-y-6">
                        {% for result in results %}
                            <div class="card bg-white p-6 rounded-xl shadow-md border border-gray-200">
                                <div class="flex flex-col md:flex-row md:items-start justify-between">
                                    <div class="mb-4 md:mb-0">
                                        <p class="text-sm text-gray-500">Original: {{ result.original_filename }}</p>
                                        <a href="{{ url_for('matched_file', filename=result.new_filename) }}" target="_blank" class="text-xl font-bold text-indigo-700 hover:underline">{{ result.new_filename }}</a>
                                        <div class="mt-2 flex flex-wrap gap-x-4 gap-y-2 text-sm">
                                            <span class="font-semibold text-gray-700">Experience: <span class="font-normal text-blue-600">{{ result.experience }} years</span></span>
                                            <span class="font-semibold text-gray-700">Education: <span class="font-normal text-blue-600">{{ result.education | join(', ') if result.education else 'N/A' }}</span></span>
                                            {% if result.keyword_total %}
                                            <span class="font-semibold text-gray-700">Keywords: <span class="font-normal text-blue-600">{{ result.keyword_matches|length }}/{{ result.keyword_total }} matched{% if result.keyword_matches %} ({{ result.keyword_matches | join(', ') }}){% endif %}</span></span>
                                            {% endif %}
                                        </div>
                                    </div>
                                    <div class="text-right flex-shrink-0">
                                        <p class="text-3xl font-extrabold text-green-600">{{ "%.2f"|format(result.score * 100) }}%</p>
                                        <p class="text-sm text-gray-600">Similarity Score</p>
                                    </div>
                                </div>
                            </div>
                        {% endfor %}
                    </div>
                {% else %}
                    <div class="text-center bg-white p-8 rounded-lg shadow-md">
                        <h3 class="text-xl font-semibold">No Matches Found</h3>
                        <p class="text-gray-600 mt-2">No resumes met the combined criteria. Try adjusting the filters or lowering the similarity threshold.</p>
                    </div>
                {% endif %}
                <!-- CHANGED: Feature 3 - excluded-resumes summary, distinct from the matched list -->
                {% if excluded %}
                <div class="mt-10">
                    <h3 class="text-xl font-semibold text-gray-700 mb-4">Excluded Resumes ({{ excluded|length }})</h3>
                    <div class="bg-white rounded-lg shadow-md border border-gray-200 divide-y divide-gray-100">
                        {% for item in excluded %}
                        <div class="px-6 py-3 flex flex-col sm:flex-row sm:justify-between sm:items-center text-sm gap-1">
                            <span class="text-gray-700 font-medium">{{ item.filename }}</span>
                            <span class="text-red-600">{{ item.reason }}</span>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
        {% endif %}
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            const toggle = document.getElementById('criteria-toggle');
            const filters = document.getElementById('criteria-filters');
            const label = document.getElementById('toggle-label');

            function updateFilterState() {
                if (toggle.checked) {
                    filters.disabled = false;
                    label.textContent = 'Filters Enabled';
                    label.classList.remove('text-gray-500');
                    label.classList.add('text-gray-900');
                } else {
                    filters.disabled = true;
                    label.textContent = 'Filters Disabled';
                    label.classList.remove('text-gray-900');
                    label.classList.add('text-gray-500');
                }
            }
            toggle.addEventListener('change', updateFilterState);
            updateFilterState();

            const fileInput = document.getElementById('resumes');
            const fileList = document.getElementById('file-list');
            fileInput.addEventListener('change', (event) => {
                const files = event.target.files;
                if (files.length > 0) {
                    let fileNames = `<p class="font-semibold mb-1">${files.length} file(s) selected:</p><ul class="list-disc list-inside">`;
                    for (const file of files) { fileNames += `<li>${file.name}</li>`; }
                    fileNames += '</ul>'
                    fileList.innerHTML = fileNames;
                } else { fileList.innerHTML = ''; }
            });

            const thresholdSlider = document.getElementById('threshold');
            const thresholdValue = document.getElementById('threshold-value');
            thresholdSlider.addEventListener('input', (event) => {
                thresholdValue.textContent = event.target.value;
            });
            thresholdValue.textContent = thresholdSlider.value;
        });
    </script>
</body>
</html>
"""


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main page, form submission, and results display."""
    if NLP_MODEL is None:
        flash("The NLP model failed to load. Please check the console and restart the server.", "error")
        return render_template_string(HTML_TEMPLATE, results=None)

    if request.method == 'POST':
        if 'resumes' not in request.files:
            flash('No resume file part in the request.', 'error')
            return redirect(request.url)

        files = request.files.getlist('resumes')
        requirements_text = request.form.get('requirements', '').strip()
        similarity_threshold = float(request.form.get('threshold', 0.6))
        
        filters_enabled = request.form.get('criteria_toggle') == 'on'
        min_experience_str = request.form.get('experience', '0')
        required_education = set(request.form.getlist('education'))

        if not files or files[0].filename == '':
            flash('No resumes selected for uploading.', 'error')
            return redirect(request.url)
        if not requirements_text:
            flash('Job description cannot be empty.', 'error')
            return redirect(request.url)
        
        min_experience = 0
        if filters_enabled:
            try:
                min_experience = float(min_experience_str) if min_experience_str else 0
            except ValueError:
                flash('Please enter a valid number for experience.', 'error')
                return redirect(request.url)

        # CHANGED: Feature 1 - parse the up-to-10 keyword/requirement fields
        # and enforce the cap server-side, not just via the fixed HTML form.
        keywords_raw = request.form.getlist('keywords')
        if len(keywords_raw) > 10:
            flash('You may submit at most 10 keywords/requirements.', 'error')
            return redirect(request.url)
        keywords = [k.strip() for k in keywords_raw if k.strip()]

        results = []
        excluded_resumes = []  # CHANGED: Feature 3 - (filename, reason) for hard-filter exclusions
        match_counter = 0

        for file in files:
            if file and allowed_file(file.filename):
                original_filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                file.save(file_path)

                resume_text = extract_text_from_file(file_path, original_filename)
                if not resume_text:
                    continue

                candidate_experience = find_experience(resume_text)
                candidate_education = find_education(resume_text)

                # CHANGED: Feature 3 - hard-filter failures are now recorded
                # with a reason instead of a silent `continue`, so the
                # results view can show the user where each resume went.
                if filters_enabled:
                    if candidate_experience < min_experience:
                        excluded_resumes.append({
                            'filename': original_filename,
                            'reason': f"excluded: experience {candidate_experience} yrs < required {min_experience} yrs",
                        })
                        continue
                    missing_education = required_education - candidate_education
                    if missing_education:
                        excluded_resumes.append({
                            'filename': original_filename,
                            'reason': f"excluded: missing {', '.join(sorted(missing_education))}",
                        })
                        continue

                nlp_criteria = {"requirements_text": requirements_text}
                nlp_grade = grade_resume_nlp(resume_text, nlp_criteria, NLP_MODEL, similarity_threshold)

                # CHANGED: Feature 1 - keyword/requirement matching, reported
                # independently of the semantic similarity score.
                matched_keywords = find_keyword_matches(resume_text, keywords)

                if nlp_grade["passed"]:
                    file_extension = original_filename.rsplit('.', 1)[1]
                    # CHANGED (bug fix): use a tentative counter for the
                    # filename. It's only committed to match_counter — and
                    # this resume only added to results — once the copy is
                    # verified to have actually succeeded.
                    tentative_counter = match_counter + 1
                    new_filename = f"{tentative_counter:06d}_{nlp_grade['similarity_score']:.2f}.{file_extension}"
                    new_filepath = os.path.join(app.config['MATCH_FOLDER'], new_filename)

                    copy_succeeded = False
                    try:
                        shutil.copy(file_path, new_filepath)
                        # Don't rely on the absence of an exception alone -
                        # confirm the destination file actually exists.
                        copy_succeeded = os.path.exists(new_filepath)
                        if not copy_succeeded:
                            flash(f"Could not copy {original_filename}: destination file was not created.", 'error')
                    except (IOError, OSError, shutil.Error) as e:
                        print(f"ERROR copying file: {e}")
                        flash(f"Could not copy {original_filename}. Reason: {e}", 'error')

                    if not copy_succeeded:
                        # Skip this resume entirely: don't count it in
                        # match_counter, don't show it in results.
                        continue

                    match_counter = tentative_counter

                    results.append({
                        'original_filename': original_filename,
                        'new_filename': new_filename,
                        'score': nlp_grade['similarity_score'],
                        'experience': candidate_experience,
                        'education': sorted(list(candidate_education)),
                        'keyword_matches': matched_keywords,   # CHANGED
                        'keyword_total': len(keywords),        # CHANGED
                    })

        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        # CORRECTED TYPO
        return render_template_string(HTML_TEMPLATE, results=sorted_results, excluded=excluded_resumes)  # CHANGED

    # CORRECTED TYPO
    return render_template_string(HTML_TEMPLATE, results=None)


@app.route('/matches/<filename>')
def matched_file(filename):
    """Serves the matched files from the 'matching_resumes' folder."""
    return send_from_directory(app.config['MATCH_FOLDER'], filename)


# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)
