from flask import Flask, request, jsonify, send_from_directory, send_file, Response, stream_with_context
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
import asyncio
import logging
import traceback
import json
from services.pdf_service import pdf_service
from services.flashcard_service import flashcard_service
from services.enhanced_learning_service import enhanced_learning_service
from services.quiz_service import quiz_service
from services.flashcard_service import flashcard_service
from services.copilot_service import copilot_client
from services.groq_service import groq_client
from werkzeug.utils import secure_filename
from utils.logger_config import setup_logger

# Set up main application logger
logger = setup_logger('app', 'app.log')

# Load environment variables from .env file
logger.info("Loading environment variables...")
load_dotenv(override=True)

app = Flask(__name__)

# Configure CORS to allow all origins for development
# WARNING: This is not suitable for production
CORS(app, 
     resources={
         r"/api/*": {
             "origins": ["http://localhost:8080", "http://127.0.0.1:8080"],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True
         }
     })

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Add error handlers
@app.errorhandler(500)
def handle_500_error(e):
    logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(404)
def handle_404_error(e):
    logger.error(f"Not found error: {str(e)}")
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(408)
def handle_408_error(e):
    logger.error(f"Timeout error: {str(e)}")
    return jsonify({'error': 'Request timeout'}), 408

@app.errorhandler(413)
def handle_413_error(e):
    logger.error(f"File too large: {str(e)}")
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.route('/api/quiz/generate', methods=['POST'])
async def generate_quiz():
    try:
        logger.info("\n" + "="*80)
        logger.info("Received new quiz generation request")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request form data: {request.form}")
        logger.info(f"Request files: {request.files}")
        
        # Log environment for debugging
        logger.info(f"Environment variables: COHERE_API_KEY={'set' if os.getenv('COHERE_API_KEY') else 'not set'}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Python version: {sys.version}")
        logger.info("="*80 + "\n")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            logger.error("Invalid file type")
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # Create unique doc_id
        doc_id = secure_filename(file.filename)
        
        # Save the file
        filename = f"{doc_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved file to {filepath}")

        try:
            # Extract text from PDF with increased timeout
            logger.info("Starting PDF text extraction for quiz generation")
            text = await asyncio.wait_for(
                asyncio.to_thread(pdf_service.extract_text, filepath, doc_id=doc_id),
                timeout=60  # 60 seconds for PDF extraction
            )
            logger.info(f"Extracted {len(text)} characters from PDF")

            if not text.strip():
                raise ValueError("Extracted text is empty. The PDF might be image-based or corrupted.")
                
            if len(text) > 10000:  # If text is too long, truncate it
                logger.warning(f"Text too long ({len(text)} chars), truncating to 10000 chars")
                text = text[:10000]

            # Generate quiz questions with increased timeout
            logger.info("Starting quiz question generation")
            logger.info(f"Text length: {len(text)} characters")
            logger.debug(f"Sample text (first 500 chars):\n{text[:500]}")
            
            try:
                quiz_questions = await asyncio.wait_for(
                    quiz_service.generate_quiz(text, doc_id=doc_id),
                    timeout=120  # Increased to 120 seconds for quiz generation
                )
                logger.info(f"Generated {len(quiz_questions)} quiz questions")
                logger.debug(f"Quiz questions: {json.dumps(quiz_questions, indent=2)[:1000]}...")  # Log first 1000 chars
                
                if not quiz_questions:
                    logger.warning("No quiz questions were generated. Check the logs for more details.")
                    return jsonify({
                        'error': 'Failed to generate quiz questions. The content might be too short or not suitable for quiz generation.',
                        'suggestions': [
                            'Try uploading a longer document with more detailed content.',
                            'Ensure the document contains clear, educational content.',
                            'Try a different document if the issue persists.'
                        ]
                    }), 400

                # Clean up the file
                os.remove(filepath)
                logger.info(f"Removed temporary file {filepath}")

                return jsonify({
                    'quiz': quiz_questions,
                    'message': 'Quiz generated successfully'
                })

            except asyncio.TimeoutError:
                logger.error("Quiz generation timed out")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    'error': 'Quiz generation timed out. The PDF might be too large or complex.',
                    'suggestions': [
                        'Try uploading a smaller document or a document with less complex content.',
                        'The server is currently processing other requests. Please try again in a moment.',
                        'If the problem persists, contact support with the details of your document.'
                    ]
                }), 408

        except asyncio.TimeoutError:
            logger.error("PDF text extraction timed out for quiz")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': 'PDF text extraction timed out. The file might be too large or contain complex formatting.'
            }), 408
        except Exception as e:
            logger.error(f"Error processing PDF for quiz: {str(e)}", exc_info=True)
            # Clean up the file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error in quiz generation: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred while processing your request',
            'details': str(e),
            'debug': {
                'error_type': type(e).__name__,
                'status': 'error'
            }
        }), 500

# Add a test route
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Backend server is running!'})

@app.route('/api/test', methods=['GET'])
def test_route():
    return jsonify({'message': 'API is working!'})

@app.route('/api/flashcards/generate', methods=['POST', 'OPTIONS'])
async def generate_flashcards():
    logger.info("\n" + "="*80)
    logger.info("=== FLASHCARD GENERATION REQUEST STARTED ===")
    
    # Log request details
    logger.info(f"[Request] Method: {request.method}")
    logger.info(f"[Request] Headers: {dict(request.headers)}")
    logger.info(f"[Request] Content Type: {request.content_type}")
    logger.info(f"[Request] Form Data: {request.form}")
    logger.info(f"[Request] Files: {request.files}")
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        logger.info("Handling CORS preflight request")
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
        
    try:
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            logger.error("Invalid file type")
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # Create unique doc_id
        doc_id = secure_filename(file.filename)
        
        # Save the file
        filename = f"{doc_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved file to {filepath}")

        try:
            # Extract text from PDF
            logger.info(f"Extracting text from: {filepath} with doc_id: {doc_id}")
            text = await asyncio.wait_for(
                asyncio.to_thread(pdf_service.extract_text, filepath, doc_id=doc_id),
                timeout=60
            )
            
            if not text.strip():
                raise ValueError("Extracted text is empty")

            # Generate flashcards
            logger.info(f"Starting flashcard generation with doc_id: {doc_id}")
            flashcards = await asyncio.wait_for(
                flashcard_service.generate_flashcards(text, doc_id=doc_id),
                timeout=120
            )
            logger.info(f"✅ Successfully generated {len(flashcards)} flashcards")
        
            if not flashcards:
                logger.warning("No flashcards were generated. Returning empty list instead of error.")
                return jsonify({
                    'flashcards': [],
                    'message': 'No flashcards could be generated from the provided document.',
                    'debug': {
                        'num_flashcards': 0,
                        'status': 'no_content'
                    }
                })

            # Clean up the file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"✅ Removed temporary file: {filepath}")
                else:
                    logger.warning(f"File not found during cleanup: {filepath}")

                logger.info("\n=== FLASHCARD GENERATION COMPLETED SUCCESSFULLY ===")
                return jsonify({
                    'flashcards': flashcards,
                    'message': 'Flashcards generated successfully',
                    'debug': {
                        'num_flashcards': len(flashcards),
                        'status': 'success'
                    }
                })
            except Exception as e:
                logger.error(f"❌ Error during cleanup: {str(e)}", exc_info=True)
                # Still return the flashcards even if cleanup fails
                return jsonify({
                    'flashcards': flashcards,
                    'message': 'Flashcards generated successfully (but cleanup failed)',
                    'warning': 'Failed to clean up temporary files',
                    'debug': {
                        'num_flashcards': len(flashcards),
                        'status': 'warning_cleanup_failed'
                    }
                })

        except asyncio.TimeoutError as e:
                logger.error("❌ Flashcard generation timed out after 120 seconds", exc_info=True)
                error_msg = 'Flashcard generation timed out. The PDF might be too large or complex.'
                logger.error(error_msg)
                return jsonify({
                    'error': error_msg,
                    'suggestions': [
                        'Try uploading a smaller document or a document with less complex content.',
                        'The server is currently processing other requests. Please try again in a moment.',
                        'If the problem persists, contact support with the details of your document.'
                    ],
                    'debug': {
                        'timeout_seconds': 120,
                        'stage': 'flashcard_generation',
                        'status': 'timeout'
                    }
                }), 408

        except asyncio.TimeoutError as e:
            logger.error("❌ PDF text extraction timed out after 60 seconds", exc_info=True)
            error_msg = 'PDF text extraction timed out. The file might be too large or contain complex formatting.'
            logger.error(error_msg)
            response = jsonify({
                'error': error_msg,
                'suggestions': [
                    'Try uploading a smaller PDF file or a file with simpler formatting.',
                    'If the document contains many images, try using a text-based PDF instead.',
                    'For large documents, consider splitting them into smaller parts.'
                ],
                'debug': {
                    'timeout_seconds': 60,
                    'stage': 'pdf_extraction',
                    'status': 'timeout'
                }
            })
            response.status_code = 408
            return response
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in flashcard generation: {str(e)}", exc_info=True)
            error_msg = f'Failed to process file: {str(e)}'
            
            # Clean up the file in case of error
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"✅ Removed temporary file after error: {filepath}")
                except Exception as cleanup_error:
                    logger.error(f"❌ Failed to remove temporary file {filepath}: {str(cleanup_error)}")
            
            response = jsonify({
                'error': error_msg,
                'suggestions': [
                    'The file might be corrupted or in an unsupported format.',
                    'Try uploading a different file or contact support if the problem persists.'
                ],
                'debug': {
                    'error_type': type(e).__name__,
                    'stage': 'processing',
                    'status': 'error'
                }
            })
            response.status_code = 500
            return response

    except Exception as e:
        logger.error(f"Unexpected error in flashcard generation: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred while processing your request',
            'details': str(e),
            'debug': {
                'error_type': type(e).__name__,
                'status': 'error'
            }
        }), 500

@app.route('/api/learning/enhanced', methods=['POST'])
async def generate_enhanced_learning():
    try:
        logger.info("Received request to generate enhanced learning content")
        logger.debug(f"Request headers: {dict(request.headers)}")
        logger.debug(f"Request files: {request.files}")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            logger.error("Invalid file type")
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # Create unique doc_id
        doc_id = secure_filename(file.filename)
        
        # Save the file
        filename = f"{doc_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved file to {filepath}")

        try:
            # Extract text from PDF with increased timeout
            logger.info("Starting PDF text extraction")
            text = await asyncio.wait_for(
                asyncio.to_thread(pdf_service.extract_text, filepath, doc_id=doc_id),
                timeout=30  # 30 seconds for PDF extraction
            )
            logger.info(f"Extracted {len(text)} characters from PDF")

            if len(text) > 10000:  # If text is too long, truncate it
                logger.warning(f"Text too long ({len(text)} chars), truncating to 10000 chars")
                text = text[:10000]

            # Generate enhanced learning content with increased timeout
            logger.info("Starting enhanced learning content generation")
            try:
                learning_content = await asyncio.wait_for(
                    enhanced_learning_service.generate_learning_content(text, doc_id=doc_id),
                    timeout=60  # 60 seconds for content generation
                )
                logger.info(f"Generated {len(learning_content)} learning concepts")

                # Clean up the file
                os.remove(filepath)
                logger.info(f"Removed temporary file {filepath}")

                return jsonify({
                    'learning_content': learning_content,
                    'message': 'PDF processed successfully'
                })

            except asyncio.TimeoutError:
                logger.error("Content generation timed out")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    'error': 'Request timed out. The PDF might be too large or complex. Please try with a smaller file.'
                }), 408

        except asyncio.TimeoutError:
            logger.error("PDF text extraction timed out")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': 'PDF text extraction timed out. The file might be too large or complex.'
            }), 408
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            # Clean up the file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        logger.info("Received chat request")
        
        # Log request headers for debugging
        logger.debug(f"Request headers: {dict(request.headers)}")
        
        try:
            data = request.get_json()
            logger.debug(f"Request data: {data}")
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            return jsonify({'error': 'Invalid JSON in request body'}), 400
            
        if not data or 'messages' not in data or not isinstance(data['messages'], list):
            error_msg = "Invalid request format: missing or invalid 'messages' field"
            logger.error(error_msg)
            return jsonify({'error': error_msg, 'details': 'Expected {\'messages\': [{\'role\': \'user\', \'content\': \'...\'}]}'}), 400
        
        messages = data['messages']
        if not messages:
            logger.error("Empty messages list")
            return jsonify({'error': 'Messages list cannot be empty'}), 400
            
        # Get the last user message
        user_message = next((msg for msg in reversed(messages) if msg.get('role') == 'user'), None)
        if not user_message or not user_message.get('content'):
            logger.error("No valid user message found")
            return jsonify({'error': 'No valid user message found'}), 400
            
        logger.debug(f"Processing chat request with message: {user_message['content']}")
        
        # Check if streaming is requested
        stream = data.get('stream', False)
        
        # Add system message if not present
        if not any(msg.get('role') == 'system' for msg in messages):
            messages.insert(0, {
                'role': 'system',
                'content': 'You are a helpful study assistant. Provide clear, concise, and accurate answers to study-related questions. If a question is not related to studying, politely steer the conversation back to academic topics.'
            })
        
        if stream:
            # For streaming responses
            async def generate_events():
                try:
                    # Stream the response using Groq
                    if groq_client.is_available():
                        logger.info("Streaming response with Groq")
                        response_stream = await groq_client.generate_chat_completion(messages, stream=True)
                    else:
                        logger.info("Streaming response with Cohere (Fallback)")
                        response_stream = await copilot_client.generate_chat_completion(messages, stream=True)
                        
                    # Send start event
                    yield 'data: ' + json.dumps({'status': 'start'}) + '\n\n'
                    
                    # Stream the response chunks directly without nested sync loop
                    async for chunk in response_stream:
                        if chunk:
                            yield 'data: ' + json.dumps({'chunk': chunk}) + '\n\n'
                    
                    # Send completion event
                    yield 'data: ' + json.dumps({'status': 'done'}) + '\n\n'
                    
                except Exception as e:
                    error_msg = f"Error in streaming response: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    yield 'data: ' + json.dumps({'error': error_msg}) + '\n\n'
            
            # Create a thread-safe Queue to pass chunks from async background to sync Flask response
            import queue
            import threading
            q = queue.Queue()

            def start_background_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            # Start a background event loop in another thread
            loop = asyncio.new_event_loop()
            t = threading.Thread(target=start_background_loop, args=(loop,), daemon=True)
            t.start()
            
            async def queue_events():
                try:
                    async for event in generate_events():
                        q.put(event)
                finally:
                    q.put(None)  # Sentinel to signify completion
            
            # Send the async task to the background loop
            asyncio.run_coroutine_threadsafe(queue_events(), loop)

            def sync_generator():
                try:
                    while True:
                        event = q.get()
                        if event is None:
                            break
                        yield event
                finally:
                    # Cleanup the background loop even if client disconnects early
                    loop.call_soon_threadsafe(loop.stop)
                
            # Return the response directly as stream without stream_with_context
            return Response(
                sync_generator(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'  # Disable buffering for nginx
                }
            )
            
        else:
            # For non-streaming responses
            if groq_client.is_available():
                logger.info("Generating response with Groq")
                response = await groq_client.generate_chat_completion(messages)
            else:
                logger.info("Generating response with Cohere (Fallback)")
                response = await copilot_client.generate_chat_completion([
                    {
                        'role': 'system',
                        'content': 'You are a helpful study assistant. Provide clear, concise, and accurate answers to study-related questions. If a question is not related to studying, politely steer the conversation back to academic topics.'
                    },
                    {
                        'role': 'user',
                        'content': user_message['content']
                    }
                ])
            
            logger.info("Successfully generated chat response")
            return jsonify({'response': response})
        
    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': 'Failed to process chat request', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)