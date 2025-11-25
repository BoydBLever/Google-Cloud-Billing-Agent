import os
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from google.generativeai import types


class LLMProcessor:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.model = None     

        self.default_system_prompt = (
            "You are a professional customer service representative who gives concise, "
            "friendly, and accurate answers. If unsure, say you don’t know."
        )

    def _lazy_init(self):
        if self.model is not None:
            return  # already initialized

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing from environment variables.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _build_prompt(self, prompt: str, conversation_history: Optional[List[Dict[str, str]]], system_prompt: Optional[str]) -> str:
        system_content = system_prompt if system_prompt else self.default_system_prompt
        parts = [f"SYSTEM: {system_content}"]

        if conversation_history:
            for m in conversation_history:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role.upper()}: {content}")

        parts.append(f"USER: {prompt}")
        return "\n".join(parts)

    def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:

        self._lazy_init()

        full_prompt = self._build_prompt(prompt, conversation_history, system_prompt)

        resp = self.model.generate_content(
            full_prompt,
            generation_config=types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=200,
            ),
        )
        return (resp.text or "").strip()

    def customize_for_call_center(self) -> None:
        self.default_system_prompt = (
            "You are a professional call center agent. Guidelines:\n"
            "1. Be friendly, concise, and helpful.\n"
            "2. Ask for missing details politely.\n"
            "3. Provide clear answers without rambling.\n"
            "4. Offer escalation to a human agent when needed."
        )

    def customize_for_lead_generation(self) -> None:
        self.default_system_prompt = (
            "You are a professional lead-gen assistant. Guidelines:\n"
            "1. Greet warmly.\n"
            "2. Ask about customer needs.\n"
            "3. Highlight value briefly.\n"
            "4. Collect key contact details.\n"
            "5. Suggest next steps."
        )

    def analyze_conversation(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        analysis_prompt = (
            "Analyze the following conversation and extract:\n"
            "1. Customer's main issues\n"
            "2. Customer's emotional state\n"
            "3. Key info points\n"
            "4. Suggested follow-up actions\n\n"
        )
        for m in conversation_history:
            who = "Customer" if m["role"] == "user" else "Assistant"
            analysis_prompt += f"{who}: {m['content']}\n"

        resp = self.model.generate_content(
            analysis_prompt,
            generation_config=types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=300,
            ),
        )
        return {"analysis": (resp.text or "").strip()}
