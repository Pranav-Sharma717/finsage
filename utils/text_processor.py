import re
from typing import List, Dict, Tuple
import pandas as pd
from bs4 import BeautifulSoup
import PyPDF2

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract numerical values from text."""
        # Find all numbers including those with commas and decimal points
        numbers = re.findall(r'[\d,]+\.?\d*', text)
        # Convert to float, removing commas
        return [float(num.replace(',', '')) for num in numbers]
    
    @staticmethod
    def extract_percentages(text: str) -> List[float]:
        """Extract percentage values from text."""
        percentages = re.findall(r'(\d+\.?\d*)%', text)
        return [float(p) for p in percentages]
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract dates from text."""
        # Match various date formats
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}'
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))
        return dates
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """Extract sections from budget text based on headers."""
        sections = {}
        # Split text into lines
        lines = text.split('\n')
        current_section = "Introduction"
        current_content = []
        
        for line in lines:
            # Check if line is a header (all caps or starts with number)
            if line.isupper() or re.match(r'^\d+\.', line):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    @staticmethod
    def extract_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    @staticmethod
    def extract_from_html(html_content: str) -> str:
        """Extract text from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()
    
    @staticmethod
    def find_key_terms(text: str, terms: List[str]) -> Dict[str, List[str]]:
        """Find sentences containing key terms."""
        sentences = re.split(r'[.!?]+', text)
        term_matches = {term: [] for term in terms}
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            for term in terms:
                if term.lower() in sentence.lower():
                    term_matches[term].append(sentence)
        
        return term_matches
    
    @staticmethod
    def calculate_statistics(text: str) -> Dict[str, float]:
        """Calculate basic statistics about the text."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        numbers = TextProcessor.extract_numbers(text)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'number_count': len(numbers),
            'average_number': sum(numbers) / len(numbers) if numbers else 0
        } 