import os, json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai import types
from utils.mock_data import get_account, get_policy, create_ticket

class LLMProcessor:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.model = None     

        self.default_system_prompt = (
        "You are Geebot, a Google Cloud Billing assistant. Take exactly one action: "
        "lookup_account(account_id), lookup_policy(policy_key), create_ticket(account_id,issue), or respond_final(text)."
        'Output strictly one JSON object like: "{\"action\":\"...\",\"args\":{\"key\":\"value\"}}".'
        )

    def _lazy_init(self):
        if self.model is not None:
            return

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

    def run_agent_step(
        self,
        user_text: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """
        One agentic step:
        1) Ask Gemini for a JSON action.
        2) Parse it.
        3) Pull the corresponding lever.
        4) Return a natural-language reply string for the UI / TTS.
        """
        if self.model is None:
            self._lazy_init()

        # Flatten conversation to text; simple but works
        history_text = ""
        for m in conversation_history:
            history_text += f"{m['role']}: {m['content']}\n"
        prompt = (
            self.default_system_prompt
            + "\n\nConversation so far:\n"
            + history_text
            + f"user: {user_text}\n"
        )

        resp = self.model.generate_content(prompt)
        raw = resp.text.strip()

        # Try to parse the JSON action
        try:
            action_obj = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: just surface model output
            return f"(Internal error: expected JSON action, got: {raw})"

        action = action_obj.get("action")
        args = action_obj.get("args", {}) or {}

        if action == "lookup_account":
            account_id = args.get("account_id")
            result = get_account(account_id)
            return f"Here’s what I found for account {account_id}:\n{result}"

        if action == "lookup_policy":
            policy_key = args.get("policy_key")
            result = get_policy(policy_key)
            return f"Here’s the policy for {policy_key}:\n{result}"

        if action == "create_ticket":
            account_id = args.get("account_id")
            issue = args.get("issue")
            ticket_id = create_ticket(account_id, issue)
            return f"I’ve created support ticket {ticket_id} for account {account_id} about: {issue}"

        if action == "respond_final":
            # Expect args like {"text": "..."}
            return args.get("text", "I’m returning a final response, but no text was provided.")

        # Unknown action fallback
        return f"(Internal error: unknown action '{action}' with args {args})"
    
    
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
