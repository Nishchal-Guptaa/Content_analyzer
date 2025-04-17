from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F
import os
import io
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv
import re

load_dotenv()

app = Flask(__name__, instance_relative_config=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.instance_path, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Text classification model
TEXT_MODEL_NAME = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)
text_model.eval()

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Image classification model
IMAGE_MODEL_NAME = "Falconsai/nsfw_image_detection"
image_processor = ViTImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
image_model = AutoModelForImageClassification.from_pretrained(IMAGE_MODEL_NAME)
image_model.eval()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Models & Forms
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), nullable=False, unique=True)
    password = db.Column(db.String(70), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

# Add classification history model to track usage
class ClassificationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_type = db.Column(db.String(20), nullable=False)  # 'text', 'file', or 'image'
    filename = db.Column(db.String(100), nullable=True)
    content_snippet = db.Column(db.String(200), nullable=True)
    prediction = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('classifications', lazy=True))

class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=4, max=20)])
    submit = SubmitField("Register")

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError("Username already exists.")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

class AdminLoginForm(FlaskForm):
    username = StringField("Username", validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=4, max=20)])
    submit = SubmitField("Login")

# Admin access decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need admin privileges to access this page.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Custom Admin index view to require admin login
class SecureAdminIndexView(AdminIndexView):
    @expose('/')
    @admin_required
    def index(self):
        stats = {
            'total_users': User.query.count(),
            'total_classifications': ClassificationHistory.query.count(),
            'text_classifications': ClassificationHistory.query.filter_by(content_type='text').count(),
            'file_classifications': ClassificationHistory.query.filter_by(content_type='file').count(),
            'image_classifications': ClassificationHistory.query.filter_by(content_type='image').count(),
            'recent_activities': ClassificationHistory.query.order_by(ClassificationHistory.timestamp.desc()).limit(10).all()
        }
        return self.render('admin/index.html', stats=stats)

# Custom ModelView that respects admin-only access
class SecureModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin

# Initialize Flask-Admin
admin = Admin(app, name='Content Moderation Admin', template_mode='bootstrap4', 
              index_view=SecureAdminIndexView())

# Add model views to admin
admin.add_view(SecureModelView(User, db.session))
admin.add_view(SecureModelView(ClassificationHistory, db.session))

# Routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated and current_user.is_admin:
        return redirect(url_for('admin.index'))
        
    form = AdminLoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data) and user.is_admin:
            login_user(user)
            return redirect(url_for('admin.index'))
        else:
            flash('Invalid credentials or insufficient permissions')
    return render_template('admin/login.html', form=form)

@app.route('/sign_up', methods=['GET','POST'])
def sign_up():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('sign_up.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template("model.html")  # The page with the classifier UI

def analyze_text_content(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    
    # Handle case where only one item is analyzed (returns single probability, not list)
    if not isinstance(probs, list):
        probs = [probs]
    
    results = {LABELS[i]: round(float(probs[i]) * 100, 2) for i in range(len(LABELS))}
    prediction_tags = [label for label, score in results.items() if score > 50]

    nsfw_keywords = ["xxx", "hotgirls", "sex", "nude"]
    spam_keywords = ["click here", "win", "free", "subscribe", "giveaway"]
    is_nsfw = any(kw in text.lower() for kw in nsfw_keywords)
    is_spam = any(kw in text.lower() for kw in spam_keywords)
    content_type = "nsfw" if is_nsfw else "spam" if is_spam else "clean"

    if not prediction_tags:
        prediction_tags = [content_type]
    elif content_type != "clean":
        prediction_tags.append(content_type)

    return {
        "toxic_scores": results,
        "prediction_tags": prediction_tags
    }

@app.route('/classify', methods=['POST'])
@login_required
def classify_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    analysis = analyze_text_content(text)
    
    # Record this classification
    snippet = text[:100] + "..." if len(text) > 100 else text
    history_entry = ClassificationHistory(
        user_id=current_user.id,
        content_type='text',
        content_snippet=snippet,
        prediction=",".join(analysis["prediction_tags"])
    )
    db.session.add(history_entry)
    db.session.commit()
    
    return jsonify({
        "comment": text,
        "prediction_tags": analysis["prediction_tags"],
        "toxic_scores": analysis["toxic_scores"]
    })

@app.route('/classify_file', methods=['POST'])
@login_required
def classify_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.endswith('.txt'):
        return jsonify({"error": "Only .txt files are supported"}), 400
    
    try:
        content = file.read().decode('utf-8')
        
        # Split into chunks if too large (to avoid model context limit)
        chunks = []
        max_chunk_size = 500  # words
        words = re.findall(r'\S+', content)
        
        for i in range(0, len(words), max_chunk_size):
            chunk = ' '.join(words[i:i + max_chunk_size])
            chunks.append(chunk)
        
        # Analyze each chunk
        results = []
        overall_tags = set()
        overall_scores = {label: 0 for label in LABELS}
        
        for chunk in chunks:
            analysis = analyze_text_content(chunk)
            results.append({
                "chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "analysis": analysis
            })
            
            # Aggregate tags and scores
            overall_tags.update(analysis["prediction_tags"])
            for label, score in analysis["toxic_scores"].items():
                overall_scores[label] = max(overall_scores[label], score)
        
        # Record this classification
        history_entry = ClassificationHistory(
            user_id=current_user.id,
            content_type='file',
            filename=file.filename,
            content_snippet=content[:100] + "..." if len(content) > 100 else content,
            prediction=",".join(overall_tags)
        )
        db.session.add(history_entry)
        db.session.commit()
        
        return jsonify({
            "filename": file.filename,
            "total_chunks": len(chunks),
            "prediction_tags": list(overall_tags),
            "toxic_scores": overall_scores,
            "chunk_results": results
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classify_image', methods=['POST'])
@login_required
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid image format"}), 400
    
    try:
        # Read the uploaded image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process image with model
        inputs = image_processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = image_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # Get results
            predicted_label_id = logits.argmax(-1).item()
            predicted_label = image_model.config.id2label[predicted_label_id]
            
            # Create a dictionary of all class probabilities
            all_probs = {}
            for i, prob in enumerate(probs[0]):
                label = image_model.config.id2label[i]
                all_probs[label] = round(float(prob) * 100, 2)
            
            # Record this classification
            history_entry = ClassificationHistory(
                user_id=current_user.id,
                content_type='image',
                filename=file.filename,
                prediction=predicted_label
            )
            db.session.add(history_entry)
            db.session.commit()
            
            return jsonify({
                "filename": file.filename,
                "predicted_label": predicted_label,
                "probabilities": all_probs
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Admin utility route to create an admin user (for initial setup)
@app.route('/create_admin', methods=['GET', 'POST'])
def create_admin():
    if User.query.filter_by(is_admin=True).first():
        flash('Admin already exists')
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username and password:
            hashed_pw = bcrypt.generate_password_hash(password)
            admin_user = User(username=username, password=hashed_pw, is_admin=True)
            db.session.add(admin_user)
            db.session.commit()
            flash('Admin user created successfully')
            return redirect(url_for('admin_login'))
            
    return render_template('admin/create_admin.html')

@app.route('/admin/stats')
@admin_required
def admin_stats():
    # Get basic stats for admin dashboard
    stats = {
        'total_users': User.query.count(),
        'total_classifications': ClassificationHistory.query.count(),
        'text_classifications': ClassificationHistory.query.filter_by(content_type='text').count(),
        'file_classifications': ClassificationHistory.query.filter_by(content_type='file').count(),
        'image_classifications': ClassificationHistory.query.filter_by(content_type='image').count(),
    }
    
    # Get most active users
    active_users = db.session.query(
        User.username, 
        db.func.count(ClassificationHistory.id).label('count')
    ).join(ClassificationHistory).group_by(User.id).order_by(db.desc('count')).limit(5).all()
    
    stats['active_users'] = active_users
    
    return jsonify(stats)

# Create DB tables if they don't exist
# with app.app_context():
#     db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
