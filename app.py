from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

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

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/budget-summary')
def budget_summary():
    # TODO: Implement budget summary logic
    return render_template('budget_summary.html')

@app.route('/personal-guidance', methods=['GET', 'POST'])
def personal_guidance():
    if request.method == 'POST':
        try:
            income = float(request.form.get('income'))
            profession = request.form.get('profession')
            
            # TODO: Implement recommendation logic
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
    # TODO: Implement tax slab calculation
    return "Tax slab calculation pending"

def generate_financial_plan(income, profession):
    # TODO: Implement financial planning logic
    return "Financial planning suggestions pending"

def suggest_asset_allocation(income):
    # TODO: Implement asset allocation logic
    return "Asset allocation suggestions pending"

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Error creating database tables: {e}")
    app.run(debug=True) 