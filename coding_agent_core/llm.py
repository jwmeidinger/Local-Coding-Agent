from __future__ import annotations

import logging
import os
from typing import Optional

from .config import AgentConfig


class LLMManager:
    """Manages LLM interactions.

    Auto-detects whether the server is OpenAI-compatible (LM Studio, vLLM, etc.)
    or Ollama and uses the correct API format.

    Source IP binding is handled globally via apply_source_ip_binding() in main().
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.base_url = config.llm_url.rstrip("/")
        self.model = config.model
        self.temperature = config.temperature
        self.num_predict = config.num_predict

        # Setup dedicated LLM logger
        self.llm_logger = logging.getLogger("coding-agent.llm")
        self.llm_logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)
        
        # Add file handler for LLM logs
        llm_log_path = config.workspace_dir / "llm.log"
        try:
            llm_handler = logging.FileHandler(llm_log_path, mode="w", encoding="utf-8")
            llm_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
            self.llm_logger.addHandler(llm_handler)
            self.llm_logger.info(f"LLM log file: {llm_log_path}")
        except Exception as e:
            self.llm_logger.warning(f"Could not create LLM log file: {e}")

        # Detect server type
        self.server_type = self._detect_server_type()
        logging.getLogger("coding-agent").info(
            "LLM server detected as: %s at %s", self.server_type, self.base_url
        )

    def _detect_server_type(self) -> str:
        """Detect whether the LLM server is OpenAI-compatible or Ollama."""
        import requests as _req

        # Try OpenAI-compat first (LM Studio, vLLM, etc.)
        try:
            r = _req.get(f"{self.base_url}/v1/models", timeout=10)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data.get("data"), list):
                    return "openai"
        except Exception:
            pass

        # Try Ollama
        try:
            r = _req.get(f"{self.base_url}/api/tags", timeout=10)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data.get("models"), list):
                    return "ollama"
        except Exception:
            pass

        # Default to openai-compat (most common for LM Studio)
        logging.getLogger("coding-agent").warning(
            "Could not detect server type, defaulting to openai-compat"
        )
        return "openai"

    def generate(self, prompt: str, system_prompt: str = "", force_tool: str = None) -> str:
        """Generate text using the LLM, with automatic prompt truncation.
        
        Args:
            prompt: The user prompt
            system_prompt: The system prompt
            force_tool: If set, force the model to call this specific tool
        """
        import requests as _req

        max_chars = getattr(self.config, "max_prompt_chars", 40000)

        total = len(system_prompt) + len(prompt)
        if total > max_chars:
            logger = logging.getLogger("coding-agent")
            budget = max_chars - len(system_prompt)
            if budget < 2000:
                system_prompt = system_prompt[:2000] + "\n...(truncated)"
                budget = max_chars - len(system_prompt)
            prompt = prompt[:budget] + "\n\n...(prompt truncated to fit context window)"
            logger.warning(
                "Prompt truncated from %d to %d chars (limit %d)",
                total, len(system_prompt) + len(prompt), max_chars,
            )

        # Log the prompt being sent
        self.llm_logger.info("=" * 60)
        self.llm_logger.info(f"MODEL: {self.model}")
        self.llm_logger.info(f"TEMP: {self.temperature}")
        if force_tool:
            self.llm_logger.info(f"FORCED TOOL: {force_tool}")
        if system_prompt:
            self.llm_logger.info(f"SYSTEM PROMPT:\n{system_prompt}")
        self.llm_logger.info(f"USER PROMPT:\n{prompt}")
        self.llm_logger.info("-" * 60)

        if self.server_type == "openai":
            response = self._generate_openai(prompt, system_prompt, _req, force_tool)
        else:
            response = self._generate_ollama(prompt, system_prompt, _req, force_tool)

        self.llm_logger.info(f"RESPONSE:\n{response}")
        self.llm_logger.info("=" * 60)
        
        return response

    def _generate_openai(self, prompt: str, system_prompt: str, _req, force_tool: str = None) -> str:
        """Generate via OpenAI-compatible /v1/chat/completions (LM Studio, vLLM)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.num_predict,
            "stream": False,
        }

        # Add tool_choice if forcing a specific tool
        if force_tool:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": force_tool}
            }

        logger = logging.getLogger("coding-agent")

        try:
            resp = _req.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=300,  # LLM generation can be slow
            )
            if resp.status_code != 200:
                body = resp.text[:500]
                logger.error(
                    "OpenAI-compat error %s: %s (prompt ~%d chars)",
                    resp.status_code, body, len(prompt) + len(system_prompt),
                )
                return ""
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("OpenAI-compat LLM call failed: %s", e)
            return ""

    def _generate_ollama(self, prompt: str, system_prompt: str, _req, force_tool: str = None) -> str:
        """Generate via Ollama /api/generate.
        
        Note: Ollama doesn't support tool_choice, so we add guidance in the prompt instead.
        """
        # For Ollama, we can't force tool_choice, but we can modify the prompt
        # to strongly encourage the model to call the tool
        if force_tool:
            prompt = f"{prompt}\n\nIMPORTANT: You MUST call the {force_tool} tool now. Do not explain. Just call the tool."
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
            "stream": False,
        }

        logger = logging.getLogger("coding-agent")

        try:
            resp = _req.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            if resp.status_code != 200:
                body = resp.text[:500]
                logger.error(
                    "Ollama error %s: %s (prompt ~%d chars)",
                    resp.status_code, body, len(prompt) + len(system_prompt),
                )
                return ""
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            logger.error("Ollama LLM call failed: %s", e)
            return ""

    def extract_tool_calls(self, text: str) -> list:
        """Extract tool calls from LLM output.

        Uses a simple state-machine parser to correctly handle parentheses
        inside quoted string arguments (e.g. code content with function calls).
        Only extracts the FIRST tool call found to prevent multiple executions.
        """
        tool_names = ['file_read', 'file_write', 'bash', 'list_files', 'grep', 'git_status', 'web_search', 'done']
        
        # First check for tool simulation (model describes tool use instead of calling it)
        # Patterns like: "I will use web_search", "using file_read to..."
        simulation_patterns = [
            r'I will use (?:the )?(\w+)',
            r'using (?:the )?(\w+) to',
            r'call (?:the )?(\w+)',
            r'execute (?:the )?(\w+)',
        ]
        
        # Check for JSON-style tool calls in text
        import re
        json_patterns = [
            r'\{["\']tool["\']:\s*["\'](\w+)["\']',
            r'\{["\']name["\']:\s*["\'](\w+)["\']',
        ]
        
        # Find FIRST occurrence of any tool call
        earliest_pos = len(text)
        earliest_call = None
        
        for tool_name in tool_names:
            # Look for tool_name(
            pos = text.find(tool_name + '(')
            if pos != -1 and pos < earliest_pos:
                start = pos + len(tool_name)
                call_str = self._parse_balanced_parens(text, start)
                if call_str is not None:
                    earliest_call = tool_name + call_str
                    earliest_pos = pos
        
        if earliest_call:
            return [earliest_call]
        
        # Check for done signal in text
        text_lower = text.lower()
        if any(word in text_lower for word in ['done', 'complete', 'finished', 'all changes', 'task complete', 'all done']):
            return ['done(message="Task completed")']
        
        return []

    @staticmethod
    def _parse_balanced_parens(text: str, start: int) -> Optional[str]:
        """Parse from an opening '(' to its matching ')', respecting quotes.

        Returns the substring from '(' to ')' inclusive, or None if not found.
        """
        if start >= len(text) or text[start] != '(':
            return None

        depth = 0
        in_quote = None  # None, '"', or "'"
        escape = False
        i = start

        while i < len(text):
            ch = text[i]

            if escape:
                escape = False
                i += 1
                continue

            if ch == '\\' and in_quote:
                escape = True
                i += 1
                continue

            if in_quote:
                if ch == in_quote:
                    in_quote = None
            else:
                if ch in ('"', "'"):
                    in_quote = ch
                elif ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]

            i += 1

        return None
