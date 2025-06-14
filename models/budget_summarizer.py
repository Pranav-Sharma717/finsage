from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple
import gc
import PyPDF2
import re
from collections import defaultdict
import json
import os
from sklearn.model_selection import train_test_split

class BudgetDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.summaries = tokenizer(summaries, truncation=True, padding=True, max_length=max_length)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.summaries['input_ids'][idx])
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

class BudgetSummarizer:
    def __init__(self, model_path=None):
        # Use base model for better compatibility
        self.model_name = "facebook/bart-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if model_path and os.path.exists(model_path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Enable memory optimization
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
            torch.set_num_threads(2)
        
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt"
        )
        
        # Define key sectors with their common terms and context
        self.sectors = {
            "taxation": {
                "keywords": ["tax", "income tax", "gst", "customs", "duty", "exemption", "tariff", "cess"],
                "context": ["rate", "percentage", "reduction", "increase", "reform", "structure"]
            },
            "infrastructure": {
                "keywords": ["infrastructure", "roads", "railways", "ports", "airports", "highways", "metro", "transport"],
                "context": ["project", "development", "construction", "investment", "allocation"]
            },
            "healthcare": {
                "keywords": ["health", "medical", "hospital", "ayushman", "insurance", "pharma", "medicine"],
                "context": ["scheme", "facility", "treatment", "coverage", "benefit"]
            },
            "education": {
                "keywords": ["education", "school", "college", "university", "skill", "training", "research"],
                "context": ["institute", "program", "scholarship", "faculty", "student"]
            },
            "agriculture": {
                "keywords": ["agriculture", "farm", "farmer", "crop", "irrigation", "rural", "food"],
                "context": ["subsidy", "support", "scheme", "production", "income"]
            },
            "defense": {
                "keywords": ["defense", "military", "armed forces", "security", "weapons", "equipment"],
                "context": ["modernization", "procurement", "technology", "capability"]
            },
            "technology": {
                "keywords": ["technology", "digital", "it", "startup", "innovation", "research", "ai"],
                "context": ["development", "investment", "ecosystem", "infrastructure"]
            },
            "social_welfare": {
                "keywords": ["welfare", "pension", "scheme", "subsidy", "benefit", "assistance", "poverty"],
                "context": ["program", "support", "coverage", "beneficiary", "allocation"]
            }
        }
        
        # Load or create training data
        self.training_data = self._load_training_data()
    
    def _load_training_data(self) -> List[Dict]:
        """Load or create training data for fine-tuning."""
        data_path = "data/budget_training.json"
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def add_training_example(self, text: str, summary: str):
        """Add a new training example."""
        self.training_data.append({
            "text": text,
            "summary": summary
        })
        # Save updated training data
        os.makedirs("data", exist_ok=True)
        with open("data/budget_training.json", 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
    
    def fine_tune(self, epochs=3, batch_size=4, learning_rate=2e-5):
        """Fine-tune the model on budget-specific data."""
        if not self.training_data:
            print("No training data available. Please add some examples first.")
            return
        
        # Split data into train and validation sets
        train_data, val_data = train_test_split(self.training_data, test_size=0.1)
        
        # Create datasets
        train_dataset = BudgetDataset(
            [item["text"] for item in train_data],
            [item["summary"] for item in train_data],
            self.tokenizer
        )
        val_dataset = BudgetDataset(
            [item["text"] for item in val_data],
            [item["summary"] for item in val_data],
            self.tokenizer
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Fine-tune the model
        trainer.train()
        
        # Save the fine-tuned model
        self.model.save_pretrained("./models/fine_tuned_budget")
        self.tokenizer.save_pretrained("./models/fine_tuned_budget")
        
        # Update the summarizer with the fine-tuned model
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt"
        )
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file with improved formatting."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    # Extract text and clean it
                    page_text = page.extract_text()
                    # Remove page numbers and headers
                    page_text = re.sub(r'\n\d+\n', '\n', page_text)
                    # Remove multiple newlines
                    page_text = re.sub(r'\n{3,}', '\n\n', page_text)
                    text += page_text + "\n\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return ""
    
    def _extract_sector_content(self, text: str, sector: str, sector_info: Dict) -> Tuple[str, List[Dict]]:
        """Extract content relevant to a specific sector with context."""
        sentences = text.split('.')
        relevant_sentences = []
        key_numbers = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for sector keywords
            if any(keyword.lower() in sentence.lower() for keyword in sector_info["keywords"]):
                # Clean the sentence
                sentence = re.sub(r'\s+', ' ', sentence)
                sentence = re.sub(r'[^\w\s.,;:!?()-]', ' ', sentence)
                
                if sentence:
                    # Extract numbers with context
                    numbers = re.finditer(r'(\d+(?:\.\d+)?%?)\s+([^.]{5,50})', sentence)
                    for match in numbers:
                        number, context = match.groups()
                        if any(ctx.lower() in context.lower() for ctx in sector_info["context"]):
                            key_numbers.append({
                                "value": number,
                                "context": context.strip()
                            })
                    
                    relevant_sentences.append(sentence)
        
        return ' '.join(relevant_sentences), key_numbers
    
    def _simplify_summary(self, text: str) -> str:
        """Convert technical summary to layman's terms with improved clarity."""
        # Common technical terms and their simpler alternatives
        replacements = {
            'allocation': 'money set aside',
            'fiscal': 'financial',
            'deficit': 'shortage',
            'revenue': 'income',
            'expenditure': 'spending',
            'subsidy': 'financial help',
            'infrastructure': 'basic facilities',
            'implementation': 'putting into action',
            'initiative': 'new program',
            'augmentation': 'increase',
            'augmented': 'increased',
            'enhancement': 'improvement',
            'enhanced': 'improved',
            'utilization': 'use',
            'utilized': 'used',
            'approximately': 'about',
            'consequently': 'as a result',
            'furthermore': 'also',
            'moreover': 'also',
            'nevertheless': 'however',
            'notwithstanding': 'despite',
            'subsequently': 'later',
            'therefore': 'so',
            'thus': 'so',
            'whereas': 'while',
            'rationalization': 'simplification',
            'rationalized': 'simplified',
            'provision': 'arrangement',
            'provisions': 'arrangements',
            'facilitation': 'making easier',
            'facilitate': 'make easier',
            'regulatory': 'rule-based',
            'regulations': 'rules',
            'compliance': 'following rules',
            'comply': 'follow rules'
        }
        
        for technical, simple in replacements.items():
            text = re.sub(r'\b' + technical + r'\b', simple, text, flags=re.IGNORECASE)
        
        # Remove redundant phrases
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def analyze_budget(self, pdf_path: str) -> Dict[str, Dict[str, str]]:
        """Analyze budget PDF and provide sector-wise summaries in simple terms."""
        try:
            # Extract text from PDF
            text = self.extract_from_pdf(pdf_path)
            if not text:
                return {"error": "Could not extract text from PDF"}
            
            # Process each sector
            sector_analysis = {}
            for sector, sector_info in self.sectors.items():
                # Extract relevant content with context
                sector_text, key_numbers = self._extract_sector_content(text, sector, sector_info)
                if not sector_text:
                    continue
                
                # Adjust max_length based on input length
                input_length = len(sector_text.split())
                max_length = min(max(input_length // 2, 50), 150)
                min_length = min(max_length // 3, 30)
                
                # Generate summary
                summary = self.summarizer(
                    sector_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                
                # Simplify the summary
                simplified_summary = self._simplify_summary(summary)
                
                # Format key numbers with context
                formatted_numbers = []
                for num_info in key_numbers[:3]:  # Limit to top 3 most relevant numbers
                    formatted_numbers.append(f"{num_info['value']} ({num_info['context']})")
                
                sector_analysis[sector] = {
                    "summary": simplified_summary,
                    "key_numbers": formatted_numbers
                }
                
                gc.collect()
            
            return sector_analysis
            
        except Exception as e:
            print(f"Error in analyze_budget: {str(e)}")
            return {"error": str(e)}
    
    def get_impact_summary(self, sector_analysis: Dict[str, Dict[str, str]]) -> str:
        """Generate an overall impact summary in simple terms."""
        try:
            # Combine summaries from all sectors
            combined_text = " ".join(info["summary"] for info in sector_analysis.values())
            
            # Adjust max_length based on input length
            input_length = len(combined_text.split())
            max_length = min(max(input_length // 2, 100), 200)
            min_length = min(max_length // 3, 50)
            
            # Generate overall summary
            summary = self.summarizer(
                combined_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            
            # Simplify the summary
            return self._simplify_summary(summary)
            
        except Exception as e:
            print(f"Error in get_impact_summary: {str(e)}")
            return "Unable to generate overall impact summary." 