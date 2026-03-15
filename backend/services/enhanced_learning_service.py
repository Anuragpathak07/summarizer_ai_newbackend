import logging
from typing import List, Dict
import json
import os
from dotenv import load_dotenv
import aiohttp
from services.copilot_service import copilot_client
from services.groq_service import groq_client
from utils.logger_config import setup_logger

logger = setup_logger('enhanced_learning_service', 'enhanced_learning_service.log')

class EnhancedLearningService:
    def __init__(self):
        logger.info("Initialized EnhancedLearningService")

    async def generate_learning_content(self, text: str, doc_id: str = None) -> List[Dict[str, str]]:
        """
        Generate enhanced learning content from the given text using Cohere's API.
        
        Args:
            text (str): The text to generate learning content from
            doc_id (str, optional): The document ID to retrieve contextual chunks from RAG. 
            
        Returns:
            List[Dict[str, str]]: List of learning concepts, each containing:
                - concept: The main concept/topic
                - definition: Clear definition of the concept
                - real_world_application: Practical application or example
                - latest_insight: Recent research or discovery
        """
        try:
            logger.info("Starting enhanced learning content generation")
            logger.debug(f"Input text length: {len(text)} characters")

            # Step 1: Get semantic contexts via RAG if doc_id is provided
            context_text = ""
            if doc_id:
                from services.rag_service import rag_service
                logger.info(f"Querying RAG for enhanced learning concepts (doc_id: {doc_id})")
                # Use RAG to get the best conceptual chunks
                retrieved_chunks = rag_service.query("core concepts definitions theories formulas applications", n_results=10, filter_dict={"doc_id": doc_id})
                context_text = "\n".join(retrieved_chunks)
            
            if not context_text:
                # Fallback
                context_text = text[:10000]

            # Prepare the message for the API
            message = f"""Create highly detailed and comprehensive learning content from the following text.
            For each key concept, provide:
            1. An extremely detailed and in-depth definition that thoroughly explains the concept
            2. A highly elaborate real-world application or example that clearly demonstrates the concept
            3. A deep-dive into a recent research insight or discovery (include a citation if possible)
            
            Format the response as a JSON object with an array of concepts.
            Each concept should have these fields:
            - concept: The main topic name
            - definition: A highly detailed and thorough explanation
            - real_world_application: A comprehensive, practical example or application
            - latest_insight: Detailed recent research or discovery with citation
            
            Text:
            {context_text}"""

            logger.info("Preparing request to AI generation service")
            
            response = None
            # Step 2: Try Groq first for extreme detail and speed
            if groq_client.is_available():
                try:
                    logger.info("Sending request to Groq API")
                    import asyncio
                    response = await asyncio.wait_for(
                        groq_client.generate_chat_completion([{"role": "user", "content": message}]),
                        timeout=30
                    )
                except Exception as e:
                    logger.warning(f"Groq content generation failed: {str(e)}")

            # Fallback to Cohere/Copilot if Groq fails or is not available
            if response is None:
                logger.info("Sending request to Cohere API (Fallback)")
                response = await copilot_client.generate_chat_completion([{"role": "user", "content": message}])
                
            logger.info("Received AI response")
            
            try:
                # Clean the response by removing markdown code blocks if present
                clean_response = response.strip()
                if '```json' in clean_response:
                    clean_response = clean_response.split('```json')[1].split('```')[0].strip()
                elif '```' in clean_response:
                    clean_response = clean_response.split('```')[1].split('```')[0].strip()
                
                # Parse the response
                response_data = json.loads(clean_response)
                
                # Handle different response formats
                if isinstance(response_data, list):
                    learning_content = response_data
                elif 'concepts' in response_data:
                    learning_content = response_data['concepts']
                elif 'learning_content' in response_data:
                    learning_content = response_data['learning_content']
                else:
                    learning_content = []
                
                # Validate the learning content
                valid_content = []
                for concept in learning_content:
                    if not isinstance(concept, dict):
                        continue
                        
                    # Extract fields with flexible field names
                    concept_name = concept.get('concept') or concept.get('title') or concept.get('topic', 'Unnamed Concept')
                    definition = concept.get('definition') or concept.get('description') or concept.get('explanation', '')
                    application = concept.get('real_world_application') or concept.get('application') or concept.get('example', '')
                    insight = concept.get('latest_insight') or concept.get('insight') or concept.get('additional_info', '')
                    
                    # Only add if we have at least a concept name and definition
                    if concept_name and definition:
                        valid_content.append({
                            'concept': str(concept_name).strip(),
                            'definition': str(definition).strip(),
                            'real_world_application': str(application).strip(),
                            'latest_insight': str(insight).strip()
                        })
                
                # Ensure we have at least 7 concepts
                target_count = 7
                if len(valid_content) < target_count:
                    logger.warning(f"Received only {len(valid_content)} valid concepts, expected {target_count}")
                    # If we got fewer than target, try to generate more
                    remaining = target_count - len(valid_content)
                    additional_message = f"""Generate exactly {remaining} highly detailed and substantial learning concepts with the following structure:
                    [
                      {{
                        "concept": "Concept name",
                        "definition": "Extremely detailed, thorough, and in-depth definition",
                        "real_world_application": "Elaborate and specific real-world example",
                        "latest_insight": "Detailed recent research or insight with citation"
                      }}
                    ]
                    
                    Existing concepts (do not include these):
                    {json.dumps(valid_content, indent=2)}
                    
                    Important:
                    - Ensure all {remaining} concepts are distinct and cover different aspects
                    - Each concept should be well-defined and substantial
                    - Include specific examples for real-world applications
                    - Add recent research findings or insights where possible
                    
                    Return ONLY the JSON array with the new concepts, no other text or explanation.
                    """
                    
                    try:
                        additional_response = None
                        if groq_client.is_available():
                            try:
                                import asyncio
                                additional_response = await asyncio.wait_for(
                                    groq_client.generate_chat_completion([{"role": "user", "content": additional_message}]),
                                    timeout=30
                                )
                            except Exception as e:
                                logger.warning(f"Groq additional content generation failed: {str(e)}")

                        if additional_response is None:
                            logger.info("Sending request to Cohere API for additional concepts (Fallback)")
                            additional_response = await copilot_client.generate_chat_completion([{"role": "user", "content": additional_message}])
                            
                        # Clean the additional response
                        clean_additional = additional_response.strip()
                        if '```json' in clean_additional:
                            clean_additional = clean_additional.split('```json')[1].split('```')[0].strip()
                        elif '```' in clean_additional:
                            clean_additional = clean_additional.split('```')[1].split('```')[0].strip()
                        
                        additional_data = json.loads(clean_additional)
                        additional_concepts = additional_data if isinstance(additional_data, list) else []
                    
                        # Validate additional concepts
                        for concept in additional_concepts:
                            if not isinstance(concept, dict):
                                continue
                                
                            # Extract fields with flexible field names
                            concept_name = concept.get('concept') or concept.get('title') or concept.get('topic', 'Unnamed Concept')
                            definition = concept.get('definition') or concept.get('description') or concept.get('explanation', '')
                            application = concept.get('real_world_application') or concept.get('application') or concept.get('example', '')
                            insight = concept.get('latest_insight') or concept.get('insight') or concept.get('additional_info', '')
                            
                            # Only add if we have at least a concept name and definition
                            if concept_name and definition:
                                valid_content.append({
                                    'concept': str(concept_name).strip(),
                                    'definition': str(definition).strip(),
                                    'real_world_application': str(application).strip(),
                                    'latest_insight': str(insight).strip()
                                })
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw response: {additional_response}")
                    except Exception as e:
                        logger.error(f"Error processing additional concepts: {str(e)}")
                
                logger.info(f"Successfully generated {len(valid_content)} learning concepts")
                return valid_content
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response}")
                return []
            
        except Exception as e:
            logger.error(f"Error generating learning content: {str(e)}", exc_info=True)
            return []

# Create a singleton instance
enhanced_learning_service = EnhancedLearningService() 