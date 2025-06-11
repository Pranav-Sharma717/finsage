# FinSage: Your Personalized Financial Assistant

FinSage is a web-based platform that aims to demystify the government's annual budget and offer tailored financial guidance.

## Features

1. **Budget Summary**
   - LLM-powered breakdown of annual budget
   - Simplified, actionable insights
   - Focus on key changes (repo rates, subsidies, taxation)

2. **Personalized Financial Advisory**
   - Tax slab guidance based on income
   - Financial planning recommendations
   - Asset allocation strategies

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Database**: MySQL (XAMPP)
- **ML/NLP**: HuggingFace Transformers

## Project Structure

```
finsage/
├── static/          # Static files (CSS, JS, images)
├── templates/       # HTML templates
├── app.py          # Main Flask application
├── config.py       # Configuration settings
├── models/         # ML models and logic
├── utils/          # Utility functions
└── requirements.txt # Python dependencies
```

## Setup Instructions

1. Install Python 3.8+ and XAMPP
2. Clone this repository
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start XAMPP and ensure MySQL is running
6. Run the Flask application:
   ```bash
   python app.py
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 