import logging
from typing import List, Dict, Any
import json
import random
import re
import asyncio
from services.copilot_service import copilot_client
from services.groq_service import groq_client
from services.rag_service import rag_service
from utils.logger_config import setup_logger

logger = setup_logger('quiz_service', 'quiz_service.log')

def extract_sentences(text: str, min_length=50) -> List[str]:
    """Extract meaningful sentences from text."""
    # Split on sentence endings but keep common abbreviations
    sentences = re.split(r'(?<!\.\w\w.)(?<![A-Z][a-z]\.)(?<=\?|\!|\.|\n)\s+', text)
    # Filter out very short or non-sentences
    return [s.strip() for s in sentences if len(s.strip()) >= min_length and any(c.isalpha() for c in s)]

class QuizService:
    def __init__(self):
        logger.info("Initialized QuizService")

    async def generate_quiz(self, text: str, doc_id: str = None) -> List[Dict[str, Any]]:
        """
        Generate exactly 10 quiz questions using RAG context and Groq/Cohere.
        """
        logger.info("Starting quiz generation")
        
        # Step 1: Get semantic context via RAG
        context_text = ""
        if doc_id:
            # Search for broad conceptual topics to generate comprehensive questions
            retrieved_chunks = rag_service.query("major themes key concepts overall summary important facts", n_results=8, filter_dict={"doc_id": doc_id})
            context_text = "\n".join(retrieved_chunks)
        
        if not context_text:
            context_text = text[:6000]

        all_questions = []
        
        # Step 2: Try AI generation
        try:
            questions = await self._generate_ai_questions(context_text)
            if questions:
                all_questions.extend(questions)
                if len(all_questions) >= 10:
                    return all_questions[:10]
        except Exception as e:
            logger.error(f"AI quiz generation failed: {str(e)}")
        
        # Fallbacks: Simple AI or rule-based
        if len(all_questions) < 10:
            try:
                simple_questions = await self._generate_simple_questions(context_text)
                all_questions.extend([q for q in simple_questions if q['question'] not in {sq['question'] for sq in all_questions}])
                if len(all_questions) >= 10:
                    return all_questions[:10]
            except:
                pass
        
        if len(all_questions) < 10:
            needed = 10 - len(all_questions)
            all_questions.extend(self._generate_fallback_questions(text, num_questions=needed))
        
        return all_questions[:10]
        
    async def _generate_ai_questions(self, text: str, max_retries: int = 2) -> List[Dict[str, Any]]:
        """Generate exactly 10 questions using AI with retry logic."""
        base_prompt = """
        You are an expert quiz creator. Create exactly 10 high-quality multiple-choice questions based on the provided text.
        
        For EACH of the 10 questions, you MUST provide:
        1. A clear, specific, and well-formulated question
        2. Exactly 4 possible answer options (A, B, C, D)
        3. The single correct answer (must be A, B, C, or D)
        4. A highly detailed and in-depth explanation justifying the correct answer and why the other options are incorrect
        
        IMPORTANT: You MUST return EXACTLY 10 questions. No more, no less.
        
        Format your response as a JSON array of exactly 10 objects with these fields:
        - question (string)
        - options (array of exactly 4 strings)
        - correct_answer (single letter A-D)
        - explanation (string)
        
        Text to create questions from:
        {text}
        
        Example response format (showing 1 question, but you must provide 10):
        [
          {{
            "question": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "correct_answer": "C",
            "explanation": "Paris is the capital of France."
          }}
        ]
        """
        
        for attempt in range(max_retries + 1):
            try:
                # Add attempt-specific instructions
                attempt_prompt = base_prompt
                if attempt > 0:
                    attempt_prompt += f"\n\nATTEMPT {attempt + 1}: Please ensure you provide exactly 10 questions as requested."
                
                # Truncate text to fit within token limits while preserving context
                truncated_text = text[:6000]
                prompt = attempt_prompt.format(text=truncated_text)
                
                logger.info(f"Sending request to AI (attempt {attempt + 1})")
                
                # Try Groq for questions first
                questions = []
                if groq_client.is_available():
                    try:
                        logger.info("Trying Groq for quiz generation")
                        response = await asyncio.wait_for(
                            groq_client.generate_chat_completion([{"role": "user", "content": prompt}]),
                            timeout=30
                        )
                        questions = self._parse_ai_json(response)
                    except Exception as e:
                        logger.warning(f"Groq quiz generation attempt failed: {str(e)}")
                
                # Fallback to Cohere
                if not questions:
                    logger.info("Trying Cohere for quiz generation (fallback)")
                    response = await asyncio.wait_for(
                        copilot_client.generate_chat_completion([{"role": "user", "content": prompt}]),
                        timeout=45
                    )
                    questions = self._parse_ai_json(response)
                
                if not questions or not isinstance(questions, list):
                    continue

                # Validate questions
                valid_questions = []
                for q in questions:
                    if (isinstance(q, dict) and 
                        'question' in q and 
                        'options' in q and 
                        'correct_answer' in q):
                        
                        valid_questions.append({
                            'question': str(q['question']).strip(),
                            'options': [str(opt).strip() for opt in q['options'][:4]],
                            'correct_answer': str(q['correct_answer']).upper()[0],
                            'explanation': str(q.get('explanation', 'No explanation provided.')).strip()
                        })
                        if len(valid_questions) == 10:
                            break
                            
                if len(valid_questions) == 10:
                    return valid_questions
                    
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                
        return []

    def _parse_ai_json(self, response: str) -> List[Dict[str, Any]]:
        """Helper to parse JSON array from AI response."""
        try:
            clean_response = response.strip()
            if '```json' in clean_response:
                clean_response = clean_response.split('```json')[1].split('```')[0].strip()
            elif '```' in clean_response:
                clean_response = clean_response.split('```')[1].split('```')[0].strip()
            
            data = json.loads(clean_response)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ['questions', 'quiz', 'data']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
            return []
        except:
            return []

    async def _generate_simple_questions(self, text: str) -> List[Dict[str, Any]]:
        """Generate questions using a simpler prompt."""
        prompt = f"Create 5 simple multiple-choice questions based on this text. Format as JSON array. Text: {text[:4000]}"
        try:
            response = await copilot_client.generate_chat_completion([{"role": "user", "content": prompt}])
            return self._parse_ai_json(response)[:5]
        except:
            return []

    def _generate_fallback_questions(self, text: str, num_questions: int = 10) -> List[Dict[str, Any]]:
        """Rule-based question generator if AI fails completely."""
        questions = []
        sentences = extract_sentences(text)
        if not sentences:
            sentences = ["General knowledge from the document."]
            
        for i in range(num_questions):
            idx = i % len(sentences)
            sentence = sentences[idx]
            questions.append({
                'question': f"What is the key takeaway from the following part: '{sentence[:80]}...'?",
                'options': ["Option A", "Option B", "Option C", "Option D"],
                'correct_answer': 'A',
                'explanation': "Rule-based fallback question."
            })
        return questions

# Create a singleton instance
quiz_service = QuizService()
