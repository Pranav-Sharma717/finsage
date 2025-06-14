from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Tuple
import gc
import json
import os
from datetime import datetime

class BudgetChatbot:
    def __init__(self, model_path=None):
        # Use a smaller model for better performance
        self.model_name = "facebook/blenderbot-400M-distill"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if model_path and os.path.exists(model_path):
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Enable memory optimization
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
            torch.set_num_threads(2)
        
        self.chatbot = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt"
        )
        
        # Load or create conversation history
        self.conversation_history = self._load_conversation_history()
        
        # Define common budget-related topics and their keywords
        self.topics = {
            "taxation": {
                "keywords": ["tax", "income tax", "gst", "customs", "duty", "exemption"],
                "responses": [
                    "Let me explain the tax changes in simple terms.",
                    "Here's what you need to know about the tax reforms.",
                    "The tax changes affect different income groups in the following ways."
                ]
            },
            "infrastructure": {
                "keywords": ["infrastructure", "roads", "railways", "ports", "airports", "highways"],
                "responses": [
                    "The infrastructure development plans include several key projects.",
                    "Here's how the budget addresses infrastructure needs.",
                    "The government has allocated significant funds for infrastructure development."
                ]
            },
            "healthcare": {
                "keywords": ["health", "medical", "hospital", "ayushman", "insurance", "pharma"],
                "responses": [
                    "The healthcare sector has received important allocations.",
                    "Here are the key healthcare initiatives in the budget.",
                    "The budget includes several measures to improve healthcare access."
                ]
            },
            "education": {
                "keywords": ["education", "school", "college", "university", "skill", "training"],
                "responses": [
                    "The education sector has several new initiatives.",
                    "Here's how the budget supports education and skill development.",
                    "The government has announced important measures for education."
                ]
            },
            "agriculture": {
                "keywords": ["agriculture", "farm", "farmer", "crop", "irrigation", "rural"],
                "responses": [
                    "The budget includes several measures to support farmers.",
                    "Here are the key agricultural initiatives.",
                    "The government has announced important schemes for agriculture."
                ]
            }
        }
    
    def _load_conversation_history(self) -> List[Dict]:
        """Load or create conversation history."""
        history_path = "data/chat_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_conversation_history(self):
        """Save conversation history."""
        os.makedirs("data", exist_ok=True)
        with open("data/chat_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
    
    def _identify_topic(self, question: str) -> str:
        """Identify the main topic of the question."""
        question = question.lower()
        max_matches = 0
        identified_topic = "general"
        
        for topic, info in self.topics.items():
            matches = sum(1 for keyword in info["keywords"] if keyword.lower() in question)
            if matches > max_matches:
                max_matches = matches
                identified_topic = topic
        
        return identified_topic
    
    def _get_context(self, question: str) -> str:
        """Get relevant context for the question."""
        topic = self._identify_topic(question)
        if topic in self.topics:
            return self.topics[topic]["responses"][0]
        return "Let me help you understand the budget better."
    
    def _format_response(self, response: str, topic: str) -> str:
        """Format the response to be more natural and informative."""
        # Add topic-specific context
        if topic in self.topics:
            context = self.topics[topic]["responses"][0]
            response = f"{context} {response}"
        
        # Clean up the response
        response = response.strip()
        response = response.replace("  ", " ")
        
        return response
    
    def get_response(self, question: str) -> str:
        """Generate a response to the user's question."""
        try:
            # Identify the topic
            topic = self._identify_topic(question)
            
            # Get context
            context = self._get_context(question)
            
            # Prepare input for the model
            input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
            
            # Generate response
            response = self.chatbot(
                input_text,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )[0]['generated_text']
            
            # Extract the answer part
            answer = response.split("Answer:")[-1].strip()
            
            # Format the response
            formatted_response = self._format_response(answer, topic)
            
            # Save to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response": formatted_response,
                "topic": topic
            })
            self._save_conversation_history()
            
            return formatted_response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble understanding your question. Could you please rephrase it?"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self._save_conversation_history() 