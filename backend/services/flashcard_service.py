import logging
from typing import List, Dict, Union
import json
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
from services.copilot_service import copilot_client
from services.groq_service import groq_client
from services.rag_service import rag_service
from utils.logger_config import setup_logger

logger = setup_logger('flashcard_service', 'flashcard_service.log')

class FlashcardService:
    def __init__(self, default_count: int = 20):
        """Initialize service with default flashcard count."""
        self.default_count = default_count
        logger.info(f"Initialized FlashcardService (default_count={self.default_count})")

    async def _generate_simple_flashcards(self, text: str, num_flashcards: int = 20) -> List[Dict[str, str]]:
        """Generate simple flashcards by splitting text into sentences."""
        import re
        from nltk.tokenize import sent_tokenize
        import nltk
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Remove common PDF artifacts and excessive whitespace
        import re
        text = re.sub(r'[^a-zA-Z0-9\s.,?!;:()\-\'"]', '', text) # Strip weird invisible characters
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        raw_sentences = sent_tokenize(text)
        
        # Filter strictly for high-quality sentences
        sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Only keep sentences that are reasonably long, start with a letter, and don't end in weird symbols
            if len(s_clean) >= 40 and s_clean[0].isalpha() and not s_clean.endswith((':', ';', ',')):
                sentences.append(s_clean)
        
        # Create flashcards from sentences
        flashcards = []
        for i in range(0, min(len(sentences) - 1, num_flashcards * 2), 2):
            if i + 1 < len(sentences):
                # Formulate cleaner question & answer
                question_snippet = sentences[i][:120].rstrip('.')
                question = f"What is the significance of: \"{question_snippet}...\"?"
                answer = f"{sentences[i]} {sentences[i+1]}"[:300]
                if question and answer and len(answer) > 20:
                    flashcards.append({'question': question, 'answer': answer})
        
        return flashcards[:num_flashcards]

    async def generate_flashcards(self, text: str, num_flashcards: int | None = None, doc_id: str = None) -> List[Dict[str, str]]:
        """
        Generate flashcards using semantic contexts from RAG and Groq/Cohere APIs.
        """
        try:
            logger.info("Starting flashcard generation")
            
            if num_flashcards is None:
                num_flashcards = self.default_count

            # Step 1: Get semantic contexts via RAG if doc_id is provided
            context_text = ""
            if doc_id:
                # Use RAG to get the most "flashcard-ready" content
                retrieved_chunks = rag_service.query("highly detailed explanations key concepts comprehensive definitions mechanisms", n_results=10, filter_dict={"doc_id": doc_id})
                context_text = "\n".join(retrieved_chunks)
            
            if not context_text:
                # Fallback to the original truncation if RAG is skipped or empty
                context_text = text[:4000]

            logger.debug(f"Using context of length: {len(context_text)}")
            
            # Prepare a more focused prompt
            prompt = f"""You are an expert teacher. Create exactly {num_flashcards} high-quality flashcards from the provided text.
            For each, create a clear question and a highly detailed, comprehensive answer that thoroughly explains the concept.
            Format as JSON array: [{{"question":"...", "answer":"..."}}]
            
            Text: {context_text}"""

            message = {"role": "user", "content": prompt}
            
            # Step 2: Try Groq first for high speed
            if groq_client.is_available():
                try:
                    logger.info("Sending request to Groq API")
                    response = await asyncio.wait_for(
                        groq_client.generate_chat_completion([message]),
                        timeout=20
                    )
                    flashcards = self._parse_ai_response(response)
                    if flashcards:
                        logger.info(f"Generated {len(flashcards)} flashcards via Groq")
                        return flashcards[:num_flashcards]
                except Exception as e:
                    logger.warning(f"Groq flashcard generation failed: {str(e)}")

            # Step 3: Fallback to Cohere
            logger.info("Sending request to Cohere API (Fallback)")
            try:
                response = await asyncio.wait_for(
                    copilot_client.generate_chat_completion([message]),
                    timeout=30
                )
                flashcards = self._parse_ai_response(response)
                if flashcards:
                    logger.info(f"Generated {len(flashcards)} flashcards via Cohere")
                    return flashcards[:num_flashcards]
            except Exception as e:
                logger.warning(f"Cohere generation failed: {str(e)}")
            
            # Final Fallback to simple generation
            logger.info("Falling back to simple flashcard generation")
            return await self._generate_simple_flashcards(text, num_flashcards)
            
        except Exception as e:
            logger.error(f"Error in generate_flashcards: {str(e)}", exc_info=True)
            return []

    def _parse_ai_response(self, response: str) -> List[Dict[str, str]]:
        """Helper to parse JSON from AI response."""
        try:
            clean_response = response.strip()
            if '```json' in clean_response:
                clean_response = clean_response.split('```json')[1].split('```')[0].strip()
            elif '```' in clean_response:
                clean_response = clean_response.split('```')[1].split('```')[0].strip()
            
            flashcards = json.loads(clean_response)
            if not isinstance(flashcards, list):
                return []
                
            valid_flashcards = []
            for card in flashcards:
                if isinstance(card, dict) and card.get('question') and card.get('answer'):
                    valid_flashcards.append({
                        'question': str(card['question']).strip(),
                        'answer': str(card['answer']).strip()
                    })
            return valid_flashcards
        except:
            return []

# Create a singleton instance
flashcard_service = FlashcardService() 