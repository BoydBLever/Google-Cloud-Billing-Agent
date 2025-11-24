import os
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types


class LLMProcessor:
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.7):
        """
        Initialize Gemini LLM processor.
        """
        self.model_name = model_name
        self.temperature = temperature

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing from environment variables.")

        # Gemini client
        self.client = genai.Client(api_key=api_key)

        # Default system prompt
        self.default_system_prompt = (
            "You are a professional customer service representative who gives concise, "
            "friendly, and accurate answers. If unsure, say you don’t know."
        )

    def _format_messages(self, prompt: str, conversation_history):
        """
        Convert your conversation history into Gemini format.
        """
        contents = []

        # System message
        contents.append(types.Content(role="model", parts=[types.Part(text=self.default_system_prompt)]))

        # Past turns
        if conversation_history:
            for message in conversation_history:
                role = message["role"]
                text = message["content"]
                contents.append(
                    types.Content(
                        role="user" if role == "user" else "model",
                        parts=[types.Part(text=text)]
                    )
                )

        # Current user prompt
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        return contents

    def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response using Gemini.
        """
        # Optional system override
        if system_prompt:
            self.default_system_prompt = system_prompt

        # Format content
        contents = self._format_messages(prompt, conversation_history)

        # Call Gemini
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=200
            )
        )

        return response.text

    def customize_for_call_center(self) -> None:
        """
        Apply call center–specific system prompt.
        """
        self.default_system_prompt = (
            "You are a professional call center agent. Guidelines:\n"
            "1. Be friendly, concise, and helpful.\n"
            "2. Ask for missing details politely.\n"
            "3. Provide clear answers without rambling.\n"
            "4. Offer escalation to a human agent when needed."
        )

    def customize_for_lead_generation(self) -> None:
        """
        Apply lead generation–specific system prompt.
        """
        self.default_system_prompt = (
            "You are a professional lead-gen assistant. Guidelines:\n"
            "1. Greet warmly.\n"
            "2. Ask about customer needs.\n"
            "3. Highlight value briefly.\n"
            "4. Collect key contact details.\n"
            "5. Suggest next steps."
        )

    def analyze_conversation(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Use Gemini to analyze the conversation and extract structured insights.
        """
        analysis_text = "Analyze the following conversation:\n"

        for msg in conversation_history:
            who = "Customer" if msg["role"] == "user" else "Assistant"
            analysis_text += f"{who}: {msg['content']}\n"

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=analysis_text)]
            )
        ]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=300
            )
        )

        return {"analysis": response.text}
