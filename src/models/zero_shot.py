import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))




import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sklearn.metrics import classification_report
import json

from src.models.base_model import BaseModel
from src.utils import get_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroShotLanguageIdentifier(BaseModel):
    def __init__(self):
        super().__init__(model_name="zero_shot_gpt", category="zero_shot")
        self.setup_model()
        
    def setup_model(self):
        """Setup the ChatGPT model and prompt"""
        # Load configurations
        config_path = get_path('configs.yaml')
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize ChatGPT
        self.llm = ChatOpenAI(
            model_name=configs['zero_shot']['model_name'],
            openai_api_key=api_key,
            openai_api_base=configs['zero_shot']['api_base'],
            temperature=0
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are given a text. Your task is to identify which language "
                      "it is written in. Choose one and only one language from the "
                      "provided list and output only the language name, nothing else.\n\n"
                      "Available languages: {languages}"),
            ("user", "Text: {text}")
        ])
        
        logger.info("Model and prompt setup completed")
        
    def predict(self, texts: List[str], available_languages: List[str]) -> List[str]:
        """Predict languages for given texts"""
        predictions = []
        languages_str = ", ".join(available_languages)
        
        logger.info(f"Starting predictions for {len(texts)} texts")
        
        # Create progress bar
        pbar = tqdm(total=len(texts), desc="Processing texts")
        
        for text in texts:
            try:
                # Format prompt
                formatted_prompt = self.prompt.format_messages(
                    languages=languages_str,
                    text=text
                )
                
                # Get prediction
                response = self.llm(formatted_prompt)
                prediction = response.content.strip()
                
                # Validate prediction
                if prediction not in available_languages:
                    logger.warning(f"Invalid prediction '{prediction}'. Using fallback.")
                    prediction = available_languages[0]  # fallback to first language
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                predictions.append(available_languages[0])  # fallback
            
            pbar.update(1)
        
        pbar.close()
        logger.info("Predictions completed")
        
        return predictions
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """Evaluate model performance and save results"""
        # Get predictions
        predictions = self.predict(texts, list(set(true_labels)))
        
        # Calculate metrics
        report = classification_report(
            true_labels,
            predictions,
            output_dict=True,
            zero_division=0
        )
        
        # Save results
        results_dir = get_path('results', 'evaluations', 'zero_shot')
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"{self.model_name}_report.json")
        with open(results_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return report
    
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Not used in zero-shot setting"""
        raise NotImplementedError("Zero-shot model doesn't require training")
