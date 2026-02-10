import os 
import litellm
from config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model
from loggers import logger
from common import get_api_key


class DynamicModel:
    """Wrapper to make any model name string behave like a Model enum"""
    def __init__(self, value):
        self.value = value
        self.name = value.replace('-', '_').replace('.', '_')
        self._name_ = self.name
        self._value_ = value
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"DynamicModel({self.value})"


class LanguageModel():
    def __init__(self, model_name):
        # Try to use Model enum, but fall back to DynamicModel if not found
        try:
            self.model_name = Model(model_name)
        except (ValueError, AttributeError):
            # If not in enum, use DynamicModel wrapper
            print(f"Model '{model_name}' not in enum, using as dynamic model")
            self.model_name = DynamicModel(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        
        # Get API key - handle both Model enum and DynamicModel
        try:
            self.api_key = get_api_key(self.model_name)
        except (KeyError, AttributeError):
            # Fallback for models not in the API_KEY_NAMES mapping
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError(f"No API key found for model {model_name}")
        
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        litellm.drop_params=True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        # Handle both Model enum and DynamicModel
        if hasattr(model_name, 'value'):
            model_str = model_name.value
        else:
            model_str = str(model_name)
        
        # Check if it's a TogetherAI model
        try:
            if model_name in TOGETHER_MODEL_NAMES:
                litellm_name = TOGETHER_MODEL_NAMES[model_name]
                self.use_open_source_model = True
            else:
                self.use_open_source_model = False
                litellm_name = model_str
        except (KeyError, TypeError):
            # Model not in TOGETHER_MODEL_NAMES
            self.use_open_source_model = False
            litellm_name = model_str
        
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        try:
            if self.use_open_source_model and model_name in LITELLM_TEMPLATES:
                self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]
            else:
                self.eos_tokens = []
        except (KeyError, TypeError):
            self.eos_tokens = []

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        try:
            if self.model_name in LITELLM_TEMPLATES:
                litellm.register_prompt_template(
                    initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                    model=self.litellm_model_name,
                    roles=LITELLM_TEMPLATES[self.model_name]["roles"]
                )
                self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
            else:
                self.post_message = ""
        except (KeyError, TypeError):
            self.post_message = ""
        
    
    def batched_generate(self, convs_list, max_n_tokens, temperature, top_p=1.0, max_retries=3, **kwargs):
        """
        Generate responses for a batch of conversations with error handling and retries.
        
        Args:
            convs_list: List of conversations
            max_n_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter (default: 1.0)
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Additional parameters to pass to the model
        """
        import time
        from openai import BadRequestError, RateLimitError, APIError
        
        outputs = []
        
        for conv in convs_list:
            success = False
            retry_count = 0
            
            while not success and retry_count < max_retries:
                try:
                    # Convert Model enum to string if needed
                    model_name_str = str(self.model_name)
                    if hasattr(self.model_name, 'value'):
                        model_name_str = self.model_name.value
                    elif hasattr(self.model_name, 'name'):
                        # Handle Model.gpt_3_5 -> "gpt-3.5-turbo"
                        model_name_str = self.model_name.name.replace('_', '-')
                    
                    # Ensure it's a proper model name string
                    if model_name_str.startswith('Model.'):
                        model_name_str = model_name_str.replace('Model.', '').replace('_', '-')
                    
                    # Convert conversation to messages format
                    if hasattr(conv, 'to_openai_api_messages'):
                        messages = conv.to_openai_api_messages()
                    elif hasattr(conv, 'messages'):
                        messages = conv.messages
                    else:
                        messages = [{"role": "user", "content": str(conv)}]
                    
                    # Filter out unsupported parameters for OpenAI API
                    supported_params = {
                        'temperature', 'top_p', 'n', 'stream', 'stop', 
                        'presence_penalty', 'frequency_penalty', 'logit_bias', 'user'
                    }
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
                    
                    response = litellm.completion(
                        model=model_name_str,
                        messages=messages,
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        api_key=self.api_key,
                        **filtered_kwargs
                    )
                    outputs.append(response)
                    success = True
                    
                except BadRequestError as e:
                    print(f"BadRequestError (attempt {retry_count + 1}/{max_retries}): {e}")
                    
                    # Check if it's a context length issue
                    if "maximum context length" in str(e).lower() or "context_length_exceeded" in str(e).lower():
                        print("Context too long! Truncating conversation...")
                        # Truncate the conversation and retry
                        if hasattr(conv, 'messages') and len(conv.messages) > 2:
                            conv.messages = conv.messages[-2:]
                            retry_count += 1
                            continue
                        else:
                            print("Cannot truncate further, skipping this conversation")
                            outputs.append(None)
                            break
                    else:
                        # Other BadRequestError, skip this conversation
                        print(f"Unrecoverable BadRequestError: {e}")
                        outputs.append(None)
                        break
                        
                except RateLimitError as e:
                    print(f"RateLimitError (attempt {retry_count + 1}/{max_retries}): {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        print("Max retries reached for rate limit")
                        outputs.append(None)
                        
                except APIError as e:
                    print(f"APIError (attempt {retry_count + 1}/{max_retries}): {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1)
                    else:
                        print("Max retries reached for API error")
                        outputs.append(None)
                        
                except AttributeError as e:
                    print(f"AttributeError: {e}")
                    outputs.append(None)
                    break
                        
                except Exception as e:
                    print(f"Unexpected error: {type(e).__name__}: {e}")
                    outputs.append(None)
                    break
        
        # Extract responses, handling None values
        responses = []
        for output in outputs:
            if output is None:
                responses.append("")
            elif isinstance(output, dict):
                try:
                    responses.append(output["choices"][0]["message"]["content"])
                except (KeyError, IndexError, TypeError):
                    try:
                        responses.append(output.choices[0].message.content)
                    except (AttributeError, IndexError):
                        print(f"Error extracting response")
                        responses.append("")
            elif hasattr(output, 'choices'):
                try:
                    responses.append(output.choices[0].message.content)
                except (AttributeError, IndexError):
                    print(f"Error extracting response from litellm object")
                    responses.append("")
            else:
                print(f"Unexpected output type: {type(output)}")
                responses.append("")
        
        return responses



