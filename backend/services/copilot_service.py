import logging
import os
from typing import List, Dict, AsyncGenerator
import json
import aiohttp
from dotenv import load_dotenv
from utils.logger_config import setup_logger

logger = setup_logger('copilot_service', 'copilot_service.log')

class CopilotClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('COHERE_API_KEY')
        if not self.api_key:
            error_msg = "COHERE_API_KEY not found in environment variables. Please set it in the .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.endpoint = "https://api.cohere.ai/v1/chat"
        self.model = "command-r"  # Using the command-r model since 'command' was deprecated
        logger.info(f"Initialized CopilotClient with model: {self.model}")

    async def generate_chat_completion(self, messages: List[Dict[str, str]], stream: bool = False) -> AsyncGenerator[str, None] | str:
        """
        Generate a chat completion using Cohere's API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            stream: If True, returns an async generator that yields tokens
            
        Returns:
            If stream=True: AsyncGenerator that yields response tokens
            If stream=False: Complete response as a string
        """
        try:
            logger.info("Starting chat completion request")
            logger.debug(f"Using API key: {self.api_key[:5]}...{self.api_key[-5:] if self.api_key else ''}")
            
            if not self.api_key:
                error_msg = "API key not configured. Please set COHERE_API_KEY in .env file."
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Starting {'streaming ' if stream else ''}chat completion request")
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            logger.debug(f"Request headers: {headers}")
            
            # Format messages for Cohere API
            chat_history = []
            for msg in messages[:-1]:  # All messages except the last one
                chat_history.append({
                    'role': 'USER' if msg['role'] == 'user' else 'CHATBOT',
                    'message': msg['content']
                })
            
            payload = {
                'model': self.model,
                'message': messages[-1]['content'],  # Latest user message
                'chat_history': chat_history,
                'temperature': 0.7,
                'max_tokens': 1000,
                'stream': stream
            }
            
            if stream:
                return self._stream_response(payload, headers)
                
            # For non-streaming responses
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    result = await response.json()
                    return result.get('text', '')
                    
        except Exception as e:
            logger.error(f"Error in generate_chat_completion: {str(e)}", exc_info=True)
            raise
            
    async def _stream_response(self, payload: dict, headers: dict) -> AsyncGenerator[str, None]:
        """
        Handle streaming response from Cohere API.
        
        Args:
            payload: The request payload
            headers: Request headers
            
        Yields:
            Response tokens as they arrive
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout for streaming
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    buffer = ""
                    async for chunk in response.content:
                        if chunk:
                            try:
                                # Cohere's streaming format is newline-separated JSON
                                for line in chunk.decode('utf-8').split('\n'):
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    # Parse the JSON response
                                    try:
                                        data = json.loads(line)
                                        if 'text' in data:
                                            yield data['text']
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse JSON: {line}")
                                        continue
                                        
                            except Exception as e:
                                logger.error(f"Error processing chunk: {str(e)}")
                                continue
                    
        except Exception as e:
            logger.error(f"Error in _stream_response: {str(e)}", exc_info=True)
            raise

    # Schema for quiz generation
    QUIZ_SCHEMA = {
        "type": "object",
        "properties": {
            "quiz": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "correct_answer": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["question", "options", "correct_answer", "explanation"]
                }
            }
        },
        "required": ["quiz"]
    }

    # Schema for concept generation
    CONCEPT_SCHEMA = {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string"},
                        "definition": {"type": "string"},
                        "real_world_application": {"type": "string"},
                        "latest_insight": {"type": "string"}
                    },
                    "required": ["concept", "definition", "real_world_application", "latest_insight"]
                }
            }
        },
        "required": ["concepts"]
    }

# Create a singleton instance
copilot_client = CopilotClient() 