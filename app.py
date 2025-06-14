from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv
from models.budget_summarizer import BudgetSummarizer
from models.budget_chatbot import BudgetChatbot
from utils.text_processor import TextProcessor
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/finsage?charset=utf8mb4'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 280,
    'pool_timeout': 20,
    'pool_size': 10
}

try:
    db = SQLAlchemy(app)
except Exception as e:
    print(f"Error connecting to database: {e}")
    raise

# Initialize models
budget_summarizer = BudgetSummarizer()
budget_chatbot = BudgetChatbot()

# Initialize ML models
text_processor = TextProcessor()

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    income = db.Column(db.Float, nullable=False)
    profession = db.Column(db.String(100), nullable=False)
    recommendations = db.relationship('Recommendation', backref='user', lazy=True)

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    suggestion = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/budget-summary', methods=['GET', 'POST'])
def budget_summary():
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'budget_file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['budget_file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Only PDF files are allowed'}), 400
            
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the budget
            sector_analysis = budget_summarizer.analyze_budget(filepath)
            
            if 'error' in sector_analysis:
                return jsonify({'error': sector_analysis['error']}), 400
            
            # Get overall impact summary
            impact_summary = budget_summarizer.get_impact_summary(sector_analysis)
            
            # Clean up the file
            os.remove(filepath)
            
            return jsonify({
                'sector_analysis': sector_analysis,
                'impact_summary': impact_summary
            })
            
        except Exception as e:
            # Clean up file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return render_template('budget_summary.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({'error': 'No message provided'}), 400
            
            # Get response from chatbot
            response = budget_chatbot.get_response(data['message'])
            return jsonify({'response': response})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('chat.html')

@app.route('/personal-guidance', methods=['GET', 'POST'])
def personal_guidance():
    if request.method == 'POST':
        try:
            income = float(request.form.get('income'))
            profession = request.form.get('profession')
            
            recommendations = {
                'tax_slab': calculate_tax_slab(income),
                'financial_planning': generate_financial_plan(income, profession),
                'asset_allocation': suggest_asset_allocation(income)
            }
            
            return jsonify(recommendations)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return render_template('personal_guidance.html')

# Helper functions
def calculate_tax_slab(income):
    """Calculate tax slab and deductions."""
    if income <= 500000:
        return "You fall under the 0% tax slab. Consider investing in tax-saving instruments for future benefits."
    elif income <= 1000000:
        return "You fall under the 20% tax slab. Maximize your 80C deductions and consider tax-saving investments."
    else:
        return "You fall under the 30% tax slab. Focus on tax-efficient investments and proper tax planning."

def generate_financial_plan(income, profession):
    """Generate personalized financial planning advice."""
    base_plan = "Based on your income and profession, here's a suggested financial plan:\n"
    
    if profession == 'salaried':
        return base_plan + "1. Emergency fund: 6 months of expenses\n2. Health insurance: 5-7% of income\n3. Term insurance: 10-12x annual income\n4. Retirement planning: 15-20% of income"
    elif profession == 'business':
        return base_plan + "1. Business emergency fund: 12 months of expenses\n2. Health insurance: 7-10% of income\n3. Term insurance: 15-20x annual income\n4. Retirement planning: 20-25% of income"
    else:
        return base_plan + "1. Emergency fund: 8-10 months of expenses\n2. Health insurance: 6-8% of income\n3. Term insurance: 12-15x annual income\n4. Retirement planning: 18-22% of income"

def suggest_asset_allocation(income):
    """Suggest asset allocation based on income."""
    if income <= 500000:
        return "Conservative allocation:\n- 40% Fixed Income\n- 30% Equity\n- 20% Gold\n- 10% Cash"
    elif income <= 1000000:
        return "Moderate allocation:\n- 30% Fixed Income\n- 40% Equity\n- 20% Gold\n- 10% Cash"
    else:
        return "Aggressive allocation:\n- 20% Fixed Income\n- 50% Equity\n- 20% Gold\n- 10% Cash"

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Error creating database tables: {e}")
    app.run(debug=True) 