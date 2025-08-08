import os
import requests
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_KEYS_ENV_VARS = {
  "HUGGINGFACE": 'HF_TOKEN', 
  "GROQ": 'GROQ_API_KEY',
  "OPENROUTER": 'OPENROUTER_API_KEY',
  "TOGETHERAI": 'TOGETHERAI_API_KEY',
  "COHERE": 'COHERE_API_KEY',
  "XAI": 'XAI_API_KEY',
  "OPENAI": 'OPENAI_API_KEY',
  "GOOGLE": 'GOOGLE_API_KEY', # Or GOOGLE_GEMINI_API_KEY etc.
}

API_URLS = {
  "HUGGINGFACE": 'https://api-inference.huggingface.co/models/',
  "GROQ": 'https://api.groq.com/openai/v1/chat/completions',
  "OPENROUTER": 'https://openrouter.ai/api/v1/chat/completions',
  "TOGETHERAI": 'https://api.together.ai/v1/chat/completions',
  "COHERE": 'https://api.cohere.ai/v1/chat', 
  "XAI": 'https://api.x.ai/v1/chat/completions',
  "OPENAI": 'https://api.openai.com/v1/chat/completions',
  "GOOGLE": 'https://generativelanguage.googleapis.com/v1beta/models/',
}

#MODELS_BY_PROVIDER = json.load(open("./models.json")) 
MODELS_BY_PROVIDER = {
    "groq": {
        "default": "llama3-8b-8192",
        "models": {
            "Llama 3 8B (Groq)": "llama3-8b-8192",
        }
    }
}

def _get_api_key(provider: str, ui_api_key_override: str = None) -> str | None:
    """
    Retrieves API key for a given provider.
    Priority: UI Override > Environment Variable from API_KEYS_ENV_VARS > Specific (e.g. HF_TOKEN for HuggingFace).
    """
    provider_upper = provider.upper()
    if ui_api_key_override and ui_api_key_override.strip():
        logger.debug(f"Using UI-provided API key for {provider_upper}.")
        return ui_api_key_override.strip()

    env_var_name = API_KEYS_ENV_VARS.get(provider_upper)
    if env_var_name:
        env_key = os.getenv(env_var_name)
        if env_key and env_key.strip():
            logger.debug(f"Using API key from env var '{env_var_name}' for {provider_upper}.")
            return env_key.strip()

    # Specific fallback for HuggingFace if HF_TOKEN is set and API_KEYS_ENV_VARS['HUGGINGFACE'] wasn't specific enough
    if provider_upper == 'HUGGINGFACE':
         hf_token_fallback = os.getenv("HF_TOKEN")
         if hf_token_fallback and hf_token_fallback.strip():
             logger.debug("Using HF_TOKEN as fallback for HuggingFace provider.")
             return hf_token_fallback.strip()

    logger.warning(f"API Key not found for provider '{provider_upper}'. Checked UI override and environment variable '{env_var_name or 'N/A'}'.")
    return None

def get_available_providers() -> list[str]:
    """Returns a sorted list of available provider names (e.g., 'groq', 'openai')."""
    return sorted(list(MODELS_BY_PROVIDER.keys()))

def get_model_display_names_for_provider(provider: str) -> list[str]:
    """Returns a sorted list of model display names for a given provider."""
    return sorted(list(MODELS_BY_PROVIDER.get(provider.lower(), {}).get("models", {}).keys()))

def get_default_model_display_name_for_provider(provider: str) -> str | None:
    """Gets the default model's display name for a provider."""
    provider_data = MODELS_BY_PROVIDER.get(provider.lower(), {})
    models_dict = provider_data.get("models", {})
    default_model_id = provider_data.get("default")

    if default_model_id and models_dict:
        for display_name, model_id_val in models_dict.items():
            if model_id_val == default_model_id:
                return display_name
    
    # Fallback to the first model in the sorted list if default not found or not set
    if models_dict:
        sorted_display_names = sorted(list(models_dict.keys()))
        if sorted_display_names:
            return sorted_display_names[0]
    return None

def get_model_id_from_display_name(provider: str, display_name: str) -> str | None:
    """Gets the actual model ID from its display name for a given provider."""
    models = MODELS_BY_PROVIDER.get(provider.lower(), {}).get("models", {})
    return models.get(display_name)


def call_model_stream(provider: str, model_display_name: str, messages: list[dict], api_key_override: str = None, temperature: float = 0.7, max_tokens: int = None) -> iter:
    """
    Calls the specified model via its provider and streams the response.
    Handles provider-specific request formatting and error handling.
    Yields chunks of the response text or an error string.
    """
    provider_lower = provider.lower()
    api_key = _get_api_key(provider_lower, api_key_override)
    base_url = API_URLS.get(provider.upper())
    model_id = get_model_id_from_display_name(provider_lower, model_display_name)

    if not api_key:
        env_var_name = API_KEYS_ENV_VARS.get(provider.upper(), 'N/A')
        yield f"Error: API Key not found for {provider}. Please set it in the UI or env var '{env_var_name}'."
        return
    if not base_url:
        yield f"Error: Unknown provider '{provider}' or missing API URL configuration."
        return
    if not model_id:
         yield f"Error: Model ID not found for '{model_display_name}' under provider '{provider}'. Check configuration."
         return

    headers = {}
    payload = {}
    request_url = base_url

    logger.info(f"Streaming from {provider}/{model_display_name} (ID: {model_id})...")
    
    if provider_lower in ["groq", "openrouter", "togetherai", "openai", "xai"]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model_id, "messages": messages, "stream": True, "temperature": temperature}
        if max_tokens: payload["max_tokens"] = max_tokens

        if provider_lower == "openrouter":
             headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERRER") or "http://localhost/gradio" # Example Referer
             headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE") or "Gradio AI Researcher"      # Example Title

        try:
            response = requests.post(request_url, headers=headers, json=payload, stream=True, timeout=180)
            response.raise_for_status()

            buffer = ""
            for chunk in response.iter_content(chunk_size=None): # Process raw bytes
                buffer += chunk.decode('utf-8', errors='replace')
                while '\n\n' in buffer:
                    event_str, buffer = buffer.split('\n\n', 1)
                    if not event_str.strip(): continue

                    content_chunk = ""
                    for line in event_str.splitlines():
                        if line.startswith('data: '):
                            data_json = line[len('data: '):].strip()
                            if data_json == '[DONE]':
                                return # Stream finished
                            try:
                                data = json.loads(data_json)
                                if data.get("choices") and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if delta and delta.get("content"):
                                        content_chunk += delta["content"]
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON from stream line: {data_json}")
                    if content_chunk:
                        yield content_chunk
            if buffer.strip():
                logger.debug(f"Remaining buffer after OpenAI-like stream: {buffer}")


        except requests.exceptions.HTTPError as e:
            err_msg = f"API HTTP Error ({e.response.status_code}): {e.response.text[:500]}"
            logger.error(f"{err_msg} for {provider}/{model_id}", exc_info=False)
            yield f"Error: {err_msg}"
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request Error for {provider}/{model_id}: {e}", exc_info=False)
            yield f"Error: Could not connect to {provider} ({e})"
        except Exception as e:
            logger.exception(f"Unexpected error during {provider} stream:")
            yield f"Error: An unexpected error occurred: {e}"
        return

    elif provider_lower == "google":
        system_instruction = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system": system_instruction = {"parts": [{"text": msg["content"]}]}
            else:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                filtered_messages.append({"role": role, "parts": [{"text": msg["content"]}]})

        payload = {
             "contents": filtered_messages,
             "safetySettings": [ # Example: more permissive settings
                 {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                 {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                 {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
             ],
             "generationConfig": {"temperature": temperature}
        }
        if max_tokens: payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_instruction: payload["system_instruction"] = system_instruction
        
        request_url = f"{base_url}{model_id}:streamGenerateContent?key={api_key}" # API key in query param
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(request_url, headers=headers, json=payload, stream=True, timeout=180)
            response.raise_for_status()
            
            buffer = ""
            for chunk in response.iter_content(chunk_size=None):
                buffer += chunk.decode('utf-8', errors='replace')

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line: continue
                    if line.startswith(','): line = line[1:] 

                    try:
                        if line.startswith('data: '): line = line[len('data: '):]
                        
                        parsed_data = None
                        try:
                            parsed_data = json.loads(line)
                        except json.JSONDecodeError:
                            if line.startswith('{') and line.endswith('}'): 
                                pass 
                            elif line.startswith('{') or line.endswith('}'):
                                try:
                                    temp_parsed_list = json.loads(f"[{line}]")
                                    if temp_parsed_list and isinstance(temp_parsed_list, list):
                                        parsed_data = temp_parsed_list[0] 
                                except json.JSONDecodeError:
                                    logger.warning(f"Google: Still can't parse line even with array wrap: {line}")

                        if parsed_data:
                            data_to_process = [parsed_data] if isinstance(parsed_data, dict) else parsed_data # Ensure list
                            for event_data in data_to_process:
                                if not isinstance(event_data, dict): continue
                                if event_data.get("candidates"):
                                    for candidate in event_data["candidates"]:
                                        if candidate.get("content", {}).get("parts"):
                                            for part in candidate["content"]["parts"]:
                                                if part.get("text"):
                                                    yield part["text"]
                    except json.JSONDecodeError:
                        logger.warning(f"Google: JSONDecodeError for line: {line}")
                    except Exception as e_google_proc:
                        logger.error(f"Google: Error processing stream data: {e_google_proc}, Line: {line}")

        except requests.exceptions.HTTPError as e:
            err_msg = f"Google API HTTP Error ({e.response.status_code}): {e.response.text[:500]}"
            logger.error(err_msg, exc_info=False)
            yield f"Error: {err_msg}"
        except Exception as e:
            logger.exception(f"Unexpected error during Google stream:")
            yield f"Error: An unexpected error occurred with Google API: {e}"
        return

    elif provider_lower == "cohere":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
        
        chat_history_cohere = []
        preamble_cohere = None
        user_message_cohere = ""

        temp_messages = list(messages) 
        if temp_messages and temp_messages[0]["role"] == "system":
            preamble_cohere = temp_messages.pop(0)["content"]
        
        if temp_messages:
            user_message_cohere = temp_messages.pop()["content"] 
            for msg in temp_messages: 
                role = "USER" if msg["role"] == "user" else "CHATBOT"
                chat_history_cohere.append({"role": role, "message": msg["content"]})
        
        if not user_message_cohere:
            yield "Error: User message is empty for Cohere."
            return

        payload = {
            "model": model_id, 
            "message": user_message_cohere, 
            "stream": True, 
            "temperature": temperature
        }
        if max_tokens: payload["max_tokens"] = max_tokens 
        if chat_history_cohere: payload["chat_history"] = chat_history_cohere
        if preamble_cohere: payload["preamble"] = preamble_cohere
        
        try:
            response = requests.post(base_url, headers=headers, json=payload, stream=True, timeout=180)
            response.raise_for_status()
            
            buffer = ""
            for chunk_bytes in response.iter_content(chunk_size=None):
                buffer += chunk_bytes.decode('utf-8', errors='replace')
                while '\n\n' in buffer:
                    event_str, buffer = buffer.split('\n\n', 1)
                    if not event_str.strip(): continue
                    
                    event_type = None
                    data_json_str = None
                    for line in event_str.splitlines():
                        if line.startswith("event:"): event_type = line[len("event:"):].strip()
                        elif line.startswith("data:"): data_json_str = line[len("data:"):].strip()
                    
                    if data_json_str:
                        try:
                            data = json.loads(data_json_str)
                            if event_type == "text-generation" and "text" in data:
                                yield data["text"]
                            elif event_type == "stream-end":
                                logger.debug(f"Cohere stream ended. Finish reason: {data.get('finish_reason')}")
                                return 
                        except json.JSONDecodeError:
                            logger.warning(f"Cohere: Failed to decode JSON: {data_json_str}")
            if buffer.strip():
                 logger.debug(f"Cohere: Remaining buffer: {buffer.strip()}")


        except requests.exceptions.HTTPError as e:
            err_msg = f"Cohere API HTTP Error ({e.response.status_code}): {e.response.text[:500]}"
            logger.error(err_msg, exc_info=False)
            yield f"Error: {err_msg}"
        except Exception as e:
            logger.exception(f"Unexpected error during Cohere stream:")
            yield f"Error: An unexpected error occurred with Cohere API: {e}"
        return

    elif provider_lower == "huggingface":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt_parts = []
        for msg in messages:
            role_prefix = ""
            if msg['role'] == 'system': role_prefix = "System: " 
            elif msg['role'] == 'user': role_prefix = "User: "
            elif msg['role'] == 'assistant': role_prefix = "Assistant: "
            prompt_parts.append(f"{role_prefix}{msg['content']}")
        
        tgi_prompt = "\n".join(prompt_parts) + "\nAssistant: "

        payload = {
            "inputs": tgi_prompt,
            "parameters": {
                "temperature": temperature if temperature > 0 else 0.01, 
                "max_new_tokens": max_tokens or 1024, 
                "return_full_text": False, 
                "do_sample": True if temperature > 0 else False,
            },
            "stream": True
        }
        request_url = f"{base_url}{model_id}" 

        try:
            response = requests.post(request_url, headers=headers, json=payload, stream=True, timeout=180)
            response.raise_for_status()

            buffer = ""
            for chunk_bytes in response.iter_content(chunk_size=None):
                buffer += chunk_bytes.decode('utf-8', errors='replace')
                while '\n' in buffer: 
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line: continue

                    if line.startswith('data:'):
                        data_json_str = line[len('data:'):].strip()
                        try:
                            data = json.loads(data_json_str)
                            if "token" in data and "text" in data["token"]:
                                yield data["token"]["text"]
                            elif "generated_text" in data and data.get("details") is None: 
                                 pass 

                        except json.JSONDecodeError:
                            if not data_json_str.startswith('{') and not data_json_str.startswith('['):
                                yield data_json_str
                            else:
                                logger.warning(f"HF: Failed to decode JSON and not raw string: {data_json_str}")
            if buffer.strip():
                 logger.debug(f"HF: Remaining buffer: {buffer.strip()}")


        except requests.exceptions.HTTPError as e:
            err_msg = f"HF API HTTP Error ({e.response.status_code}): {e.response.text[:500]}"
            logger.error(err_msg, exc_info=False)
            yield f"Error: {err_msg}"
        except Exception as e:
            logger.exception(f"Unexpected error during HF stream:")
            yield f"Error: An unexpected error occurred with HF API: {e}"
        return

    else:
        yield f"Error: Provider '{provider}' is not configured for streaming in this handler."
        return
