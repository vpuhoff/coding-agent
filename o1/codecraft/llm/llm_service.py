import os
import openai

class LLMError(Exception):
    pass

class LLMService:
    def __init__(self, model=None, system_prompt=None, temperature=None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OpenAI API key not found in environment.")
        openai.api_key = api_key
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.system_prompt = system_prompt or os.getenv(
            "OPENAI_SYSTEM_PROMPT", "You are an expert software engineering assistant."
        )
        self.temperature = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "300"))

    def generate_plan(self, task_description: str, tech_summary: str) -> str:
        user_content = (
            f"You are provided with a project description and its technology stack.\n"
            f"Project tech stack: {tech_summary}\n"
            f"Task description: {task_description}\n"
            f"Provide a concise plan outlining the steps or changes required to implement the task."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            raise LLMError(f"LLM API call failed: {e}")
        try:
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMError(f"Unexpected LLM response format: {e}")
