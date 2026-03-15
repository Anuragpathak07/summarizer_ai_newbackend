import os
import logging
from typing import List, Dict, AsyncGenerator
import json
import aiohttp
from groq import Groq
from dotenv import load_dotenv
from utils.logger_config import setup_logger

logger = setup_logger('groq_service', 'groq.log')

class GroqClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables. Groq service will be unavailable.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info("Initialized GroqClient")
        
        self.model = "llama-3.1-8b-instant"  # Using active Llama 3.1 8B for high performance and stability

    def is_available(self) -> bool:
        return self.client is not None

    async def generate_chat_completion(self, messages: List[Dict[str, str]], stream: bool = False) -> AsyncGenerator[str, None] | str:
        """
        Generate a chat completion using Groq's API.
        """
        if not self.is_available():
            raise ValueError("Groq API key not configured.")

        try:
            logger.info(f"Starting {'streaming ' if stream else ''}Groq chat completion request")
            
            # Format messages for Groq
            groq_messages = []
            for msg in messages:
                groq_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

            if stream:
                return self._stream_response(groq_messages)
            
            # Non-streaming response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in Groq generate_chat_completion: {str(e)}", exc_info=True)
            raise

    async def _stream_response(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Handle streaming response from Groq API.
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in Groq _stream_response: {str(e)}", exc_info=True)
            raise

# Create a singleton instance
groq_client = GroqClient()
