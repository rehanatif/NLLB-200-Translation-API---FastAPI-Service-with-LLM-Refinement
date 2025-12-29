from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import threading
import gc
import logging
import time
from typing import Optional, Union, List, Dict, Any
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request timeout (in seconds) - should be less than client timeout
REQUEST_TIMEOUT = 55  # Set to 55 seconds to be under 60s client timeout

# -----------------------------
# Global model variables
# -----------------------------
MODEL_NAME = "facebook/nllb-200-distilled-600M"
REFINEMENT_MODEL_NAME = "microsoft/phi-2"  # Better instruction-following model for localization refinement

# NLLB translation model
tokenizer = None
model = None

# Refinement LLM
refinement_tokenizer = None
refinement_model = None

device = None
model_lock = threading.Lock()
refinement_lock = threading.Lock()  # Separate lock for refinement model

# Batch size limit to prevent memory issues
MAX_BATCH_SIZE = 30  # Reduced for faster processing

# Optimized generation parameters for faster inference
GENERATION_CONFIG = {
    "max_length": 512,
    "num_beams": 3,  # Reduced from 5 for faster inference
    "early_stopping": True,
    "do_sample": False,
    "num_return_sequences": 1,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 3,
}

# Refinement generation parameters (more deterministic for better quality)
REFINEMENT_CONFIG = {
    "max_new_tokens": 128,  # Reduced for shorter outputs
    "temperature": 0.3,  # Lower temperature for more consistent output
    "do_sample": True,
    "top_p": 0.8,  # More focused sampling
    "repetition_penalty": 1.2,
    "pad_token_id": None,  # Will be set dynamically
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown."""
    global tokenizer, model, refinement_tokenizer, refinement_model, device
    
    logger.info("Loading translation model (NLLB)... please wait.")
    start_time = time.time()
    
    try:
        # Load NLLB translation model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Set model to evaluation mode
        model.eval()
        
        # Disable gradient computation globally for inference
        torch.set_grad_enabled(False)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Optimize model for inference (if supported)
        if device.type == "cuda" and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Translation model compiled for faster inference")
            except Exception as e:
                logger.warning(f"Translation model compilation failed (continuing without): {str(e)}")
        
        load_time = time.time() - start_time
        logger.info(f"Translation model loaded successfully on {device} in {load_time:.2f} seconds!")
        
        # Load refinement model (optional, lazy loading)
        logger.info("Loading refinement model (Phi-2)... please wait.")
        refinement_start = time.time()
        
        try:
            refinement_tokenizer = AutoTokenizer.from_pretrained(REFINEMENT_MODEL_NAME)
            
            # Load model with appropriate dtype
            if device.type == "cuda":
                # Use float16 for GPU to save memory
                refinement_model = AutoModelForCausalLM.from_pretrained(
                    REFINEMENT_MODEL_NAME,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                # Use float32 for CPU
                refinement_model = AutoModelForCausalLM.from_pretrained(
                    REFINEMENT_MODEL_NAME,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            refinement_model.eval()
            refinement_model = refinement_model.to(device)
            
            # Set pad token if not set
            if refinement_tokenizer.pad_token is None:
                refinement_tokenizer.pad_token = refinement_tokenizer.eos_token
            
            refinement_load_time = time.time() - refinement_start
            logger.info(f"Refinement model loaded successfully on {device} in {refinement_load_time:.2f} seconds!")
            
        except Exception as e:
            logger.warning(f"Failed to load refinement model: {str(e)}. Refinement feature will be disabled.")
            logger.warning("You can still use translation without refinement (is_context_friendly=false)")
            refinement_tokenizer = None
            refinement_model = None
        
    except Exception as e:
        logger.error(f"Failed to load translation model: {str(e)}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if refinement_model is not None:
        del refinement_model
    if refinement_tokenizer is not None:
        del refinement_tokenizer
    if device and device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


app = FastAPI(
    title="AI Translation API",
    description="Local translation service using Meta NLLB model.",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set PyTorch to use deterministic algorithms if possible (optional, for stability)
try:
    torch.use_deterministic_algorithms(False)  # Keep False for performance
except:
    pass


# -----------------------------
# Input schema
# -----------------------------
class TranslateRequest(BaseModel):
    text: Any = Field(
        ..., 
        description="Text to translate: single string, list of strings, or dictionary (keys preserved, values translated)"
    )
    lang: str = Field(..., description="Target language code (e.g., 'eng_Latn', 'jpn_Jpan', 'spa_Latn')")
    is_context_friendly: bool = Field(
        default=False,
        description="If true, applies LLM refinement to make translation more natural, concise, and suitable for UI/UX localization"
    )
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """Validate that text is either string, list of strings, or dict with string values."""
        if isinstance(v, str):
            return v
        elif isinstance(v, list):
            if not all(isinstance(item, str) for item in v):
                raise ValueError("All items in the list must be strings")
            return v
        elif isinstance(v, dict):
            if not all(isinstance(val, str) for val in v.values()):
                raise ValueError("All dictionary values must be strings")
            return v
        else:
            raise ValueError("Text must be either a string, a list of strings, or a dictionary with string values")


# -----------------------------
# Response schema
# -----------------------------
class TranslateResponse(BaseModel):
    success: bool
    translation: Optional[Union[str, Dict[str, str]]] = Field(
        None, 
        description="Translation result: string for single input, dict for batch input"
    )
    error: Optional[str] = None


# -----------------------------
# Cache for language codes (computed once)
# -----------------------------
_language_codes_cache = None

# -----------------------------
# Helper function to get available language codes
# -----------------------------
def get_available_language_codes():
    """Extract available language codes from tokenizer vocabulary."""
    global _language_codes_cache
    
    if _language_codes_cache is not None:
        return _language_codes_cache
    
    lang_codes = []
    
    # Try different methods to get language codes
    if hasattr(tokenizer, 'lang_code_to_id'):
        lang_codes = list(tokenizer.lang_code_to_id.keys())
    elif hasattr(tokenizer, 'get_vocab'):
        # Extract language codes from vocabulary
        vocab = tokenizer.get_vocab()
        # NLLB language codes are in format like "eng_Latn", "jpn_Jpan", etc.
        for token in vocab.keys():
            if '_' in token and len(token.split('_')) == 2:
                parts = token.split('_')
                # Validate format: 3-letter language code + underscore + script code
                if len(parts[0]) == 3 and len(parts[1]) >= 4:
                    lang_code = token
                    if lang_code not in lang_codes:
                        # Verify it's actually a language code by checking token ID
                        token_id = tokenizer.convert_tokens_to_ids(lang_code)
                        if token_id != tokenizer.unk_token_id:
                            lang_codes.append(lang_code)
    
    # Sort and cache
    _language_codes_cache = sorted(lang_codes)
    return _language_codes_cache


# -----------------------------
# Helper function to validate language code
# -----------------------------
def get_language_token_id(lang_code: str) -> int:
    """Get the token ID for a given language code."""
    # Use convert_tokens_to_ids to get the language token ID
    lang_token_id = tokenizer.convert_tokens_to_ids(lang_code)
    
    # Check if it's a valid token (not UNK)
    if lang_token_id == tokenizer.unk_token_id:
        # Get available codes for error message
        available_codes = get_available_language_codes()
        sample_codes = available_codes[:20] if len(available_codes) > 20 else available_codes
        raise ValueError(
            f"Language code '{lang_code}' not found in tokenizer vocabulary. "
            f"Available codes include: {', '.join(sample_codes)}..."
        )
    
    return lang_token_id


# -----------------------------
# Helper function to get language name from code
# -----------------------------
def get_language_name(lang_code: str) -> str:
    """Convert NLLB language code to readable language name."""
    # Common language mappings
    lang_map = {
        "eng_Latn": "English",
        "jpn_Jpan": "Japanese",
        "spa_Latn": "Spanish",
        "fra_Latn": "French",
        "deu_Latn": "German",
        "zho_Hans": "Simplified Chinese",
        "zho_Hant": "Traditional Chinese",
        "ara_Arab": "Arabic",
        "por_Latn": "Portuguese",
        "rus_Cyrl": "Russian",
        "hin_Deva": "Hindi",
        "kor_Hang": "Korean",
        "ita_Latn": "Italian",
        "nld_Latn": "Dutch",
        "pol_Latn": "Polish",
        "tur_Latn": "Turkish",
        "vie_Latn": "Vietnamese",
        "tha_Thai": "Thai",
        "ind_Latn": "Indonesian",
        "msa_Latn": "Malay",
    }
    
    # Try to get from map, otherwise extract language code
    if lang_code in lang_map:
        return lang_map[lang_code]
    
    # Extract base language code (e.g., "jpn_Jpan" -> "Japanese")
    base_code = lang_code.split("_")[0]
    return base_code.upper()  # Fallback to uppercase code


# -----------------------------
# Helper function to refine translation using LLM
# -----------------------------
def refine_translation(nllb_translation: str, target_lang_code: str, source_text: str = "") -> str:
    """
    Refine NLLB translation using lightweight LLM to make it more natural and UI-friendly.
    
    Args:
        nllb_translation: The raw translation from NLLB
        target_lang_code: Target language code (e.g., 'jpn_Jpan')
        source_text: Optional source text for context
        
    Returns:
        Refined translation
    """
    global refinement_model, refinement_tokenizer
    
    if refinement_model is None or refinement_tokenizer is None:
        logger.warning("Refinement model not loaded, returning NLLB translation as-is")
        return nllb_translation
    
    try:
        target_lang_name = get_language_name(target_lang_code)
        
        # Create a context-aware, localization-focused prompt optimized for Phi-2
        # Include source text for better context understanding
        context_info = ""
        if source_text:
            context_info = f"\nSource text (for context only): {source_text}"
        
        prompt = f"""Task: Refine a {target_lang_name} translation to make it more natural and culturally appropriate, while keeping the EXACT same meaning.

CRITICAL: You MUST output the refined translation in {target_lang_name} language only. Do NOT translate back to English or any other language.

Context: You are refining a machine translation to make it sound like it was written by a native {target_lang_name} speaker, with proper cultural context and local expressions.{context_info}

Rules:
- Output language: MUST be in {target_lang_name} (same as the translation below)
- Preserve the EXACT same meaning - do not change what it says
- Keep all names, proper nouns, and technical terms unchanged
- Use natural, culturally appropriate expressions for {target_lang_name}
- Make it concise and suitable for UI elements (buttons, labels, messages, warnings)
- Consider local context and common phrases used in {target_lang_name}
- Output ONLY the refined {target_lang_name} text, no quotes, no explanations, no English

Machine translation in {target_lang_name} to refine: {nllb_translation}

Refined translation in {target_lang_name} (natural, contextually appropriate):"""
        
        # Tokenize prompt
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Format for Phi-2 chat template (Phi-2 uses a simple format)
        # Phi-2 doesn't have apply_chat_template, so we format manually
        if hasattr(refinement_tokenizer, 'apply_chat_template'):
            try:
                formatted_prompt = refinement_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback for models without chat template
                formatted_prompt = f"Instruct: {prompt}\nOutput:"
        else:
            # Phi-2 format: simple instruction format
            formatted_prompt = f"Instruct: {prompt}\nOutput:"
        
        inputs = refinement_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        # Set pad_token_id for generation
        gen_config = REFINEMENT_CONFIG.copy()
        if refinement_tokenizer.pad_token_id is not None:
            gen_config["pad_token_id"] = refinement_tokenizer.pad_token_id
        else:
            gen_config["pad_token_id"] = refinement_tokenizer.eos_token_id
        
        # Generate refinement
        with refinement_lock, torch.no_grad():
            outputs = refinement_model.generate(
                **inputs,
                **gen_config
            )
        
        # Decode response
        response = refinement_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Clean up response - remove quotes, extra whitespace, and explanations
        response = response.strip()
        
        # Remove surrounding quotes if present
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        
        # Remove common prefixes/suffixes that models sometimes add
        prefixes_to_remove = [
            "Refined translation:",
            "Translation:",
            "Refined:",
            "Output:",
        ]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove any text after newlines (explanations)
        if '\n' in response:
            response = response.split('\n')[0].strip()
        
        # Clean up
        del inputs, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Validation: Check if refinement is reasonable
        # If response is empty, too short, or suspiciously different, use original
        if not response:
            logger.warning("Refinement produced empty output, using NLLB translation")
            return nllb_translation
        
        # If response is much shorter than original (less than 50% length), likely wrong
        if len(response) < len(nllb_translation) * 0.5:
            logger.warning("Refinement produced suspiciously short output, using NLLB translation")
            return nllb_translation
        
        # If response is much longer (more than 200% length), likely added explanations
        if len(response) > len(nllb_translation) * 2.0:
            logger.warning("Refinement produced suspiciously long output, using NLLB translation")
            return nllb_translation
        
        # Language detection: Check if output is in wrong language (English when target is not English)
        # Simple heuristic: If target language is not English but response contains mostly ASCII/English characters
        if target_lang_code != "eng_Latn":
            # Check if response is mostly ASCII (likely English) while NLLB translation is not
            response_ascii_ratio = sum(1 for c in response if ord(c) < 128) / len(response) if response else 0
            nllb_ascii_ratio = sum(1 for c in nllb_translation if ord(c) < 128) / len(nllb_translation) if nllb_translation else 0
            
            # If NLLB has low ASCII (non-English script) but response has high ASCII (English), likely wrong language
            if nllb_ascii_ratio < 0.3 and response_ascii_ratio > 0.7:
                logger.warning(f"Refinement output appears to be in wrong language (English instead of {target_lang_name}), using NLLB translation")
                return nllb_translation
        
        # Log for debugging
        logger.debug(f"Refinement: '{nllb_translation}' -> '{response}'")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during refinement: {str(e)}")
        # Fallback to original translation on error
        return nllb_translation


# -----------------------------
# Helper function to translate single text
# -----------------------------
def translate_single_text(text: str, lang_token_id: int, lang_code: str = "", refine: bool = False) -> str:
    """Translate a single text string, optionally with refinement."""
    try:
        # Tokenize input text with optimized settings
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256  # Reduced for faster processing
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation with thread lock and optimized config
        with model_lock, torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=lang_token_id,
                **GENERATION_CONFIG
            )
        
        # Decode translation
        translation = tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        # Clean up
        del inputs, translated_tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Apply refinement if requested
        if refine and lang_code:
            logger.debug(f"Refining translation: '{translation}'")
            translation = refine_translation(translation, lang_code, text)
            logger.debug(f"Refined translation: '{translation}'")
        
        return translation
    except Exception as e:
        # Clean up on error
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.error(f"Error translating single text: {str(e)}")
        raise


# -----------------------------
# Helper function to translate batch of texts
# -----------------------------
def translate_batch_texts(texts: List[str], lang_token_id: int, lang_code: str = "", refine: bool = False) -> Dict[str, str]:
    """Translate a batch of texts and return as dictionary."""
    start_time = time.time()
    
    # Filter out empty strings
    valid_texts = [text for text in texts if text and text.strip()]
    
    if not valid_texts:
        return {}
    
    # Process in chunks to avoid memory issues and improve speed
    result = {}
    batch_size = min(MAX_BATCH_SIZE, len(valid_texts))
    total_batches = (len(valid_texts) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(valid_texts)} texts in {total_batches} batches of size {batch_size}")
    
    for batch_idx, i in enumerate(range(0, len(valid_texts), batch_size), 1):
        batch_texts = valid_texts[i:i + batch_size]
        batch_start = time.time()
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > REQUEST_TIMEOUT:
            logger.warning(f"Request timeout approaching ({elapsed:.2f}s), stopping batch processing")
            # Fill remaining with original text
            for text in valid_texts[i:]:
                if text not in result:
                    result[text] = text
            break
        
        try:
            # Tokenize batch with optimized settings
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256  # Reduced for faster processing
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate translations with thread lock and optimized config
            with model_lock, torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=lang_token_id,
                    **GENERATION_CONFIG
                )
            
            # Decode translations
            translations = tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Add to result (apply refinement if needed)
            for original, translated in zip(batch_texts, translations):
                if refine and lang_code:
                    result[original] = refine_translation(translated, lang_code, original)
                else:
                    result[original] = translated
            
            batch_time = time.time() - batch_start
            logger.debug(f"Batch {batch_idx}/{total_batches} completed in {batch_time:.2f}s ({len(batch_texts)} texts)")
            
            # Clean up
            del inputs, translated_tokens
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Force garbage collection every few batches
            if batch_idx % 3 == 0:
                gc.collect()
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            # Clean up on error
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Fallback: translate individually if batch fails
            for text in batch_texts:
                if text not in result:
                    try:
                        result[text] = translate_single_text(text, lang_token_id, lang_code, refine)
                    except:
                        result[text] = text  # Fallback to original
    
    total_time = time.time() - start_time
    logger.info(f"Batch translation completed in {total_time:.2f}s ({len(result)} translations)")
    
    # Final cleanup
    gc.collect()
    return result


# -----------------------------
# Helper function to translate dictionary values
# -----------------------------
def translate_dictionary(dictionary: Dict[str, str], lang_token_id: int, lang_code: str = "", refine: bool = False) -> Dict[str, str]:
    """Translate dictionary values while preserving keys."""
    # Extract values to translate
    values = list(dictionary.values())
    
    if not values:
        return {}
    
    # Translate all values
    translated_values = translate_batch_texts(values, lang_token_id, lang_code, refine)
    
    # Create result dictionary with same keys but translated values
    result = {}
    for key, original_value in dictionary.items():
        # Get translated value, fallback to original if translation failed
        translated_value = translated_values.get(original_value, original_value)
        result[key] = translated_value
    
    return result


# -----------------------------
# API endpoint
# -----------------------------
@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    """
    Translate text to a target language using Meta NLLB-200 model.
    
    Supports multiple input formats:
    - Single string: {"text": "Hello", "lang": "jpn_Jpan"} -> returns string
    - Array: {"text": ["Hello", "How are you?"], "lang": "jpn_Jpan"} -> returns dict mapping original to translated
    - Dictionary: {"text": {"key1": "value1", "key2": "value2"}, "lang": "jpn_Jpan"} -> returns dict with same keys, translated values
    
    Args:
        req: TranslateRequest containing text (string, list, or dict) and target language code
        
    Returns:
        TranslateResponse with translation (string or dict) or error message
    """
    request_start = time.time()
    
    try:
        # Validate model is loaded
        if model is None or tokenizer is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Please wait for initialization."
            )
        
        # Get language token ID
        try:
            lang_token_id = get_language_token_id(req.lang)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Determine input type and count
        if isinstance(req.text, str):
            input_count = 1
            input_type = "string"
        elif isinstance(req.text, list):
            input_count = len(req.text)
            input_type = "list"
        elif isinstance(req.text, dict):
            input_count = len(req.text)
            input_type = "dict"
        else:
            input_count = 0
            input_type = "unknown"
        
        logger.info(f"Translation request: type={input_type}, count={input_count}, lang={req.lang}, refine={req.is_context_friendly}")
        
        # Check if refinement is requested but model not loaded
        if req.is_context_friendly and (refinement_model is None or refinement_tokenizer is None):
            logger.warning("Refinement requested but model not loaded, proceeding without refinement")
            req.is_context_friendly = False
        
        # Handle single string input
        if isinstance(req.text, str):
            # Validate input
            if not req.text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Text cannot be empty"
                )
            
            translation = translate_single_text(req.text, lang_token_id, req.lang, req.is_context_friendly)
            
            elapsed = time.time() - request_start
            logger.info(f"Translation completed in {elapsed:.2f}s (refined={req.is_context_friendly})")
            
            return TranslateResponse(
                success=True,
                translation=translation
            )
        
        # Handle array input (list of strings)
        elif isinstance(req.text, list):
            if not req.text:
                raise HTTPException(
                    status_code=400,
                    detail="Text list cannot be empty"
                )
            
            # Validate all items are strings
            if not all(isinstance(item, str) for item in req.text):
                raise HTTPException(
                    status_code=400,
                    detail="All items in the list must be strings"
                )
            
            # Translate all texts - returns dict mapping original to translated
            translation_dict = translate_batch_texts(req.text, lang_token_id, req.lang, req.is_context_friendly)
            
            elapsed = time.time() - request_start
            logger.info(f"Batch translation completed in {elapsed:.2f}s ({len(translation_dict)} items, refined={req.is_context_friendly})")
            
            return TranslateResponse(
                success=True,
                translation=translation_dict
            )
        
        # Handle dictionary input
        elif isinstance(req.text, dict):
            if not req.text:
                raise HTTPException(
                    status_code=400,
                    detail="Dictionary cannot be empty"
                )
            
            # Validate all values are strings
            if not all(isinstance(v, str) for v in req.text.values()):
                raise HTTPException(
                    status_code=400,
                    detail="All dictionary values must be strings"
                )
            
            # Translate dictionary values while preserving keys
            translation_dict = translate_dictionary(req.text, lang_token_id, req.lang, req.is_context_friendly)
            
            elapsed = time.time() - request_start
            logger.info(f"Dictionary translation completed in {elapsed:.2f}s ({len(translation_dict)} items, refined={req.is_context_friendly})")
            
            return TranslateResponse(
                success=True,
                translation=translation_dict
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Text must be either a string, a list of strings, or a dictionary"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - request_start
        logger.error(f"Translation failed after {elapsed:.2f}s: {str(e)}", exc_info=True)
        return TranslateResponse(
            success=False,
            error=f"Translation failed: {str(e)}"
        )


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "translation_model": MODEL_NAME,
        "refinement_model": REFINEMENT_MODEL_NAME if refinement_model is not None else None,
        "refinement_available": refinement_model is not None,
        "device": str(device)
    }


# -----------------------------
# Get available languages endpoint
# -----------------------------
@app.get("/languages")
def get_languages():
    """Get list of available language codes."""
    try:
        lang_codes = get_available_language_codes()
        return {
            "success": True,
            "languages": lang_codes,
            "count": len(lang_codes)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get language codes: {str(e)}"
        }


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8005,
        timeout_keep_alive=65,  # Keep connections alive
        timeout_graceful_shutdown=10,
        log_level="info"
    )
