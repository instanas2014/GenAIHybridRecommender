import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Only mock mode will be available.")


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: List[ChatChoice]


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """
        Initialize OpenAI client with option to use mock or actual API.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            use_mock: If True, use mock responses instead of actual API calls
        """
        self.use_mock = use_mock
        
        if not use_mock:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            
            # Get API key from parameter or environment
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def chat_completions_create(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-3.5-turbo", 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        Create chat completion using either OpenAI API or mock response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for OpenAI API
            
        Returns:
            ChatCompletion object with response
        """
        if self.use_mock:
            return self._mock_response(messages, model, temperature)
        else:
            return self._actual_api_call(messages, model, temperature, max_tokens, **kwargs)
    
    def _mock_response(self, messages: List[Dict[str, str]], model: str, temperature: float) -> ChatCompletion:
        """Generate mock response for testing purposes."""
        # You can customize this mock response based on your needs
        mock_content = {
            "adjusted_scores": [0.95, 0.88, 0.82, 0.91, 0.75],
            "reasoning": "Boosted sci-fi and thriller movies based on recent viewing patterns and evening context."
        }
        
        # Create response in the same format as OpenAI API
        message = ChatMessage(
            role="assistant",
            content=json.dumps(mock_content)
        )
        choice = ChatChoice(message=message)
        
        return ChatCompletion(choices=[choice])
    
    def _actual_api_call(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> ChatCompletion:
        """Make actual API call to OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Convert OpenAI response to our format
            choices = []
            for choice in response.choices:
                message = ChatMessage(
                    role=choice.message.role,
                    content=choice.message.content
                )
                choices.append(ChatChoice(message=message))
            
            return ChatCompletion(choices=choices)
            
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def toggle_mock(self, use_mock: bool):
        """Toggle between mock and actual API usage."""
        if not use_mock and not OPENAI_AVAILABLE:
            raise ImportError("Cannot switch to actual API: OpenAI library not installed")
        
        if not use_mock and not self.api_key:
            raise ValueError("Cannot switch to actual API: No API key provided")
        
        self.use_mock = use_mock
        
        if not use_mock and self.client is None:
            self.client = openai.OpenAI(api_key=self.api_key)


# # Example usage
# if __name__ == "__main__":
#     # Example 1: Using mock mode
#     print("=== Mock Mode ===")
#     client_mock = OpenAIClient(use_mock=True)
    
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Analyze movie preferences and adjust scores."}
#     ]
    
#     response = client_mock.chat_completions_create(messages=messages)
#     print(f"Mock response: {response.choices[0].message.content}")
    
#     # Example 2: Using actual API (uncomment when you have API key)
#     # print("\n=== Actual API Mode ===")
#     # client_real = OpenAIClient(api_key="your-api-key-here", use_mock=False)
#     # response = client_real.chat_completions_create(messages=messages)
#     # print(f"API response: {response.choices[0].message.content}")
    
#     # Example 3: Toggling between modes
#     print("\n=== Toggle Example ===")
#     client = OpenAIClient(use_mock=True)
#     print(f"Currently using mock: {client.use_mock}")
    
#     # Toggle to actual API (will raise error if no API key)
#     try:
#         client.toggle_mock(False)
#         print(f"Switched to actual API: {not client.use_mock}")
#     except (ValueError, ImportError) as e:
#         print(f"Cannot switch to actual API: {e}")