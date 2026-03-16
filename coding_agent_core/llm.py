from __future__ import annotations

import logging
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

    # Maximum characters to send to the LLM (rough estimate: ~4 chars per token).
    # Adjust this based on your model's context window size.
    # 50k tokens ≈ 150k chars; leave room for the response.
    MAX_PROMPT_CHARS = 100000

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using the LLM, with automatic prompt truncation."""
        import requests as _req

        # Truncate if the combined prompt is too large
        total = len(system_prompt) + len(prompt)
        if total > self.MAX_PROMPT_CHARS:
            logger = logging.getLogger("coding-agent")
            budget = self.MAX_PROMPT_CHARS - len(system_prompt)
            if budget < 2000:
                # System prompt is huge too — trim both
                system_prompt = system_prompt[:2000] + "\n...(truncated)"
                budget = self.MAX_PROMPT_CHARS - len(system_prompt)
            prompt = prompt[:budget] + "\n\n...(prompt truncated to fit context window)"
            logger.warning(
                "Prompt truncated from %d to %d chars to fit model context",
                total, len(system_prompt) + len(prompt),
            )

        if self.server_type == "openai":
            return self._generate_openai(prompt, system_prompt, _req)
        else:
            return self._generate_ollama(prompt, system_prompt, _req)

    def _generate_openai(self, prompt: str, system_prompt: str, _req) -> str:
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

    def _generate_ollama(self, prompt: str, system_prompt: str, _req) -> str:
        """Generate via Ollama /api/generate."""
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
        """
        tool_names = ['file_read', 'file_write', 'bash', 'list_files', 'git_status', 'web_search']
        tool_calls = []

        for tool_name in tool_names:
            # Find all occurrences of this tool name followed by (
            idx = 0
            while True:
                pos = text.find(tool_name + '(', idx)
                if pos == -1:
                    break

                # Parse from the opening ( to find the matching )
                start = pos + len(tool_name)
                call_str = self._parse_balanced_parens(text, start)
                if call_str is not None:
                    full_call = tool_name + call_str
                    tool_calls.append(full_call)

                idx = pos + 1

        return tool_calls

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
