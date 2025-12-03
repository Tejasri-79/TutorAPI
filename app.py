import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from rag_system import RAGSystem

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# OpenAI configuration - UPDATE THIS WITH YOUR OPENAI API KEY
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-5-mini')

db = SQLAlchemy(app)
rag_system = None

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('output', exist_ok=True)

# ============================================================
# Database Models
# ============================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# ============================================================
# Authentication Decorator
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# Routes - Authentication
# ============================================================
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ============================================================
# Routes - Main Application
# ============================================================
@app.route('/dashboard')
@login_required
def dashboard():
    global rag_system
    is_trained = rag_system is not None and rag_system.is_trained()
    return render_template('dashboard.html', username=session.get('username'), is_trained=is_trained)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'message': 'No files uploaded'})

        files = request.files.getlist('files[]')
        
        if not files or files[0].filename == '':
            return jsonify({'success': False, 'message': 'No files selected'})

        uploaded_files = []
        for file in files:
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filename)

        if not uploaded_files:
            return jsonify({'success': False, 'message': 'No valid PDF files uploaded'})

        return jsonify({
            'success': True, 
            'message': f'Successfully uploaded {len(uploaded_files)} file(s)',
            'files': uploaded_files
        })

    return render_template('upload.html')

@app.route('/train', methods=['POST'])
@login_required
def train():
    global rag_system
    
    try:
        # Initialize RAG system with configured OpenAI API
        rag_system = RAGSystem(
            upload_folder=app.config['UPLOAD_FOLDER'], 
            openai_api_key=OPENAI_API_KEY,
            openai_model=OPENAI_MODEL
        )
        
        # Train the model
        stats = rag_system.train()
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Training failed: {str(e)}'
        })

@app.route('/query', methods=['GET', 'POST'])
@login_required
def query():
    global rag_system
    
    if request.method == 'POST':
        question = request.form.get('question')
        
        if not rag_system or not rag_system.is_trained():
            return jsonify({
                'success': False,
                'message': 'Please upload and train the model first!'
            })
        
        try:
            answer = rag_system.answer_question(question)
            return jsonify({
                'success': True,
                'answer': answer
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            })
    
    return render_template('query.html')

@app.route('/analytics')
@login_required
def analytics():
    global rag_system
    
    if not rag_system or not rag_system.is_trained():
        flash('Please upload and train the model first!', 'warning')
        return redirect(url_for('dashboard'))
    
    try:
        analytics_data = rag_system.get_analytics()
        return render_template('analytics.html', analytics=analytics_data)
    except Exception as e:
        flash(f'Error loading analytics: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/quiz', methods=['GET', 'POST'])
@login_required
def quiz():
    global rag_system
    
    if not rag_system or not rag_system.is_trained():
        flash('Please upload and train the model first!', 'warning')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        num_chunks = int(request.form.get('num_chunks', 3))
        questions_per_chunk = int(request.form.get('questions_per_chunk', 3))
        
        try:
            quiz_df = rag_system.generate_quiz(num_chunks, questions_per_chunk)
            quiz_data = quiz_df.to_dict('records')
            
            return jsonify({
                'success': True,
                'quiz': quiz_data,
                'message': f'Generated {len(quiz_data)} questions'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error generating quiz: {str(e)}'
            })
    
    return render_template('quiz.html')

@app.route('/download-quiz')
@login_required
def download_quiz():
    try:
        from flask import send_file
        return send_file('output/generated_quiz.csv', as_attachment=True)
    except Exception as e:
        flash(f'Error downloading quiz: {str(e)}', 'error')
        return redirect(url_for('quiz'))

# ============================================================
# Initialize Database
# ============================================================
def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database initialized!")

def print_banner():
    """Print startup banner"""
    api_key_status = 'Set' if OPENAI_API_KEY else 'NOT SET - Please set OPENAI_API_KEY environment variable'
    banner = f"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           🎓 AI STUDY TUTOR - FLASK WEB APPLICATION          ║
║                                                               ║
║              Transform Your PDFs into an AI Tutor            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

📚 Features Available:
   • User Authentication (Login/Register)
   • PDF Upload & Training
   • AI-Powered Q&A (RAG)
   • Analytics & Visualizations
   • Automatic Quiz Generation

🌐 Access the application:
   → http://localhost:5000
   → http://127.0.0.1:5000

📖 Documentation:
   • GETTING_STARTED.md - Step-by-step guide
   • README.md - Full documentation
   • QUICKSTART.md - Command reference

🔧 OpenAI Configuration:
   • Model: {OPENAI_MODEL}
   • API Key: {api_key_status}

⚡ Server starting...
"""
    print(banner)

if __name__ == '__main__':
    print_banner()
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
