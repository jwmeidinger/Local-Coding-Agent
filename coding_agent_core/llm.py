from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import AgentConfig


@dataclass
class ToolCall:
    """A structured tool call from the LLM."""
    name: str
    arguments: Dict[str, Any]
    raw_id: str = ""  # OpenAI tool_call id for multi-turn

    def to_call_string(self) -> str:
        """Convert back to the text format for logging/display."""
        args = ", ".join(f'{k}="{v}"' for k, v in self.arguments.items())
        return f"{self.name}({args})"


@dataclass
class ChatResponse:
    """Response from a chat() call — either text, a tool call, or both."""
    text: str = ""
    tool_call: Optional[ToolCall] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    # True when the server rejected the request for context/token limits (recover by trimming).
    context_window_error: bool = False

    @property
    def is_tool_call(self) -> bool:
        return self.tool_call is not None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and self.tool_call is None


class LLMManager:
    """Manages LLM interactions.

    Supports two modes:
      1. generate() — single-shot text generation (planning, review).
      2. chat_with_tools() — multi-turn conversation with native tool calling.

    Auto-detects whether the server is OpenAI-compatible (LM Studio, vLLM, etc.)
    or Ollama and uses the correct API format.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.base_url = config.llm_url.rstrip("/")
        self.model = config.model
        self.temperature = config.temperature
        self.num_predict = config.num_predict

        # Retry settings for transient failures (model reloads can exceed short backoff windows)
        self.max_retries = getattr(config, "max_retries", 7)  # see AgentConfig.max_retries
        self.retry_base_delay = 2  # seconds — jittered exponential (2, 4, 8, 16, 32…)

        # Setup dedicated LLM logger
        self.llm_logger = logging.getLogger("coding-agent.llm")
        self.llm_logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)

        # Add file handler for LLM logs
        llm_log_path = config.workspace_dir / "llm.log"
        try:
            llm_handler = logging.FileHandler(llm_log_path, mode="w", encoding="utf-8")
            llm_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            ))
            self.llm_logger.addHandler(llm_handler)
            self.llm_logger.info(f"LLM log file: {llm_log_path}")
        except Exception as e:
            self.llm_logger.warning(f"Could not create LLM log file: {e}")

        # Detect server type
        self.server_type = self._detect_server_type()
        self._tools_supported: Optional[bool] = None  # Lazy probe
        logging.getLogger("coding-agent").info(
            "LLM server detected as: %s at %s", self.server_type, self.base_url
        )

        # Token usage tracking (cumulative per task)
        self.usage_prompt_tokens = 0
        self.usage_completion_tokens = 0
        self.usage_total_tokens = 0
        self.usage_call_count = 0

    def reset_usage(self) -> None:
        """Reset cumulative token counters (call at the start of each task)."""
        self.usage_prompt_tokens = 0
        self.usage_completion_tokens = 0
        self.usage_total_tokens = 0
        self.usage_call_count = 0

    def get_usage_summary(self) -> str:
        """Return a human-readable summary of token usage for the current task."""
        return (
            f"Token usage — prompt: {self.usage_prompt_tokens:,}, "
            f"completion: {self.usage_completion_tokens:,}, "
            f"total: {self.usage_total_tokens:,} "
            f"({self.usage_call_count} LLM calls)"
        )

    def _track_usage(self, raw: Dict[str, Any]) -> None:
        """Extract and accumulate token usage from an API response."""
        self.usage_call_count += 1

        # OpenAI format: {"usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}}
        usage = raw.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", 0) or 0
            completion = usage.get("completion_tokens", 0) or 0
            total = usage.get("total_tokens", 0) or (prompt + completion)
            self.usage_prompt_tokens += prompt
            self.usage_completion_tokens += completion
            self.usage_total_tokens += total
            self.llm_logger.info(
                "Tokens — prompt: %d, completion: %d, total: %d (cumulative: %d)",
                prompt, completion, total, self.usage_total_tokens,
            )
            return

        # Ollama format: {"prompt_eval_count": N, "eval_count": N}
        prompt = raw.get("prompt_eval_count", 0) or 0
        completion = raw.get("eval_count", 0) or 0
        if prompt or completion:
            total = prompt + completion
            self.usage_prompt_tokens += prompt
            self.usage_completion_tokens += completion
            self.usage_total_tokens += total
            self.llm_logger.info(
                "Tokens — prompt: %d, completion: %d, total: %d (cumulative: %d)",
                prompt, completion, total, self.usage_total_tokens,
            )

    # ------------------------------------------------------------------
    # Retry wrapper for transient HTTP failures
    # ------------------------------------------------------------------

    # Exceptions that are worth retrying (server hiccup, not a logic error)
    _TRANSIENT_ERRORS = (
        "ConnectionError", "ConnectionRefusedError", "ConnectionResetError",
        "Timeout", "ReadTimeout", "ConnectTimeout",
    )

    @staticmethod
    def _is_context_window_failure(status_code: int, body: str) -> bool:
        """Detect HTTP errors caused by prompt/context exceeding model limits."""
        b = (body or "").lower()
        hints = (
            "context length", "maximum context", "token limit", "too many tokens",
            "exceeds the context", "context window", "max token", "maximum token",
            "prompt is too long", "input length", "too long",
            "requested token", "reduce the length", "context overflow",
        )
        if status_code == 413:
            return True
        if status_code in (400, 422) and any(h in b for h in hints):
            return True
        if status_code == 500 and any(h in b for h in hints):
            return True
        return False

    def _request_with_retry(self, method: str, url: str, **kwargs):
        """HTTP request with exponential backoff for transient failures.

        Retries on connection errors and timeouts but NOT on 4xx responses
        (those indicate a real problem with the request).
        """
        import random as _random
        import requests as _req
        import time

        logger = logging.getLogger("coding-agent")
        last_err = None

        def _jittered_delay(attempt: int) -> float:
            """Exponential backoff with ±50% jitter to decorrelate parallel retries."""
            base = min(self.retry_base_delay * (2 ** (attempt - 1)), 60)
            # Seed from nanosecond time XOR attempt counter so parallel agents diverge
            seed = (time.time_ns() ^ (attempt * 0x9E3779B9)) & 0xFFFFFFFF
            rng = _random.Random(seed)
            return base + rng.uniform(0, 0.5 * base)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = _req.request(method, url, **kwargs)

                # Retry on 5xx server errors (LM Studio/Ollama overloaded)
                if resp.status_code >= 500 and attempt < self.max_retries:
                    delay = _jittered_delay(attempt)
                    logger.warning(
                        "LLM server returned %d (attempt %d/%d). "
                        "Retrying in %.1fs...",
                        resp.status_code, attempt, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    continue

                return resp

            except Exception as e:
                last_err = e
                err_type = type(e).__name__

                if attempt < self.max_retries:
                    delay = _jittered_delay(attempt)
                    logger.warning(
                        "LLM request failed: %s (%s) — attempt %d/%d. "
                        "Retrying in %ds...",
                        err_type, str(e)[:100], attempt, self.max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "LLM request failed after %d attempts: %s (%s)",
                        self.max_retries, err_type, str(e)[:200],
                    )

        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Server detection
    # ------------------------------------------------------------------

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

        logging.getLogger("coding-agent").warning(
            "Could not detect server type, defaulting to openai-compat"
        )
        return "openai"

    @property
    def supports_tools(self) -> bool:
        """Probe once whether the model supports native tool calling."""
        if self._tools_supported is not None:
            return self._tools_supported

        self._tools_supported = self._probe_tool_support()
        logging.getLogger("coding-agent").info(
            "Native tool calling: %s",
            "supported" if self._tools_supported else "NOT supported (using text fallback)",
        )
        return self._tools_supported

    def _probe_tool_support(self) -> bool:
        """Send a tiny request with a dummy tool to see if the server handles it."""
        import requests as _req

        dummy_tool = {
            "type": "function",
            "function": {
                "name": "test_probe",
                "description": "A test probe",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            },
        }
        messages = [{"role": "user", "content": "Call test_probe with x=\"hello\"."}]

        try:
            if self.server_type == "openai":
                resp = _req.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": [dummy_tool],
                        "max_tokens": 128,
                        "temperature": 0,
                        "stream": False,
                    },
                    timeout=60,
                )
                if resp.status_code != 200:
                    return False
                data = resp.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                return "tool_calls" in msg and msg["tool_calls"] is not None
            else:
                # Ollama /api/chat with tools
                resp = _req.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": [dummy_tool],
                        "stream": False,
                        "options": {"num_predict": 128},
                    },
                    timeout=60,
                )
                if resp.status_code != 200:
                    return False
                data = resp.json()
                msg = data.get("message", {})
                return bool(msg.get("tool_calls"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Single-shot generate (for planning, review, etc.)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Single-shot text generation. Used for planning + review phases."""
        import requests as _req

        max_chars = getattr(self.config, "max_prompt_chars", 80000)
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

        self.llm_logger.info("=" * 60)
        self.llm_logger.info("GENERATE (single-shot)")
        self.llm_logger.info(f"MODEL: {self.model}")
        if system_prompt:
            self.llm_logger.info(f"SYSTEM PROMPT:\n{system_prompt[:500]}...")
        self.llm_logger.info(f"USER PROMPT:\n{prompt[:1000]}...")
        self.llm_logger.info("-" * 60)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.server_type == "openai":
            response = self._call_openai(messages, _req)
        else:
            response = self._call_ollama(messages, _req)

        text = response.text
        self._track_usage(response.raw)
        self.llm_logger.info(f"RESPONSE:\n{text[:2000]}...")
        self.llm_logger.info("=" * 60)
        return text

    # ------------------------------------------------------------------
    # Multi-turn chat with tools (the main execution loop)
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> ChatResponse:
        """Send a multi-turn conversation with tool definitions.

        Returns a ChatResponse with either text or a structured ToolCall.
        Falls back to text-based tool extraction if native tools aren't supported.
        """
        import requests as _req

        self.llm_logger.info("=" * 60)
        self.llm_logger.info(f"CHAT (multi-turn, {len(messages)} msgs, {len(tools)} tools)")
        last_msg = messages[-1] if messages else {}
        self.llm_logger.info(f"Last msg role={last_msg.get('role')}: {str(last_msg.get('content', ''))[:500]}...")
        self.llm_logger.info("-" * 60)

        if self.supports_tools:
            response = self._chat_native(messages, tools, _req)
        else:
            response = self._chat_text_fallback(messages, tools, _req)

        if response.tool_call:
            self.llm_logger.info(f"TOOL CALL: {response.tool_call.to_call_string()}")
        elif response.text:
            self.llm_logger.info(f"TEXT RESPONSE: {response.text[:1000]}...")
        else:
            self.llm_logger.info("EMPTY RESPONSE")
        self._track_usage(response.raw)
        self.llm_logger.info("=" * 60)

        return response

    def _chat_native(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        _req,
    ) -> ChatResponse:
        """Call the LLM with native tool definitions."""
        logger = logging.getLogger("coding-agent")

        try:
            if self.server_type == "openai":
                resp = self._request_with_retry(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "temperature": self.temperature,
                        "max_tokens": self.num_predict,
                        "stream": False,
                    },
                    timeout=300,
                )
            else:
                # Ollama /api/chat
                resp = self._request_with_retry(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.num_predict,
                        },
                    },
                    timeout=300,
                )

            if resp.status_code != 200:
                body = resp.text[:2000]
                if self._is_context_window_failure(resp.status_code, body):
                    logger.warning(
                        "LLM context window exceeded (%s) — caller should trim and retry",
                        resp.status_code,
                    )
                    return ChatResponse(context_window_error=True, raw={"status": resp.status_code})
                logger.error("LLM error %s: %s", resp.status_code, body[:500])
                return ChatResponse()

            data = resp.json()
            return self._parse_native_response(data)

        except Exception as e:
            logger.error("LLM chat call failed: %s", e)
            return ChatResponse()

    def _parse_native_response(self, data: Dict[str, Any]) -> ChatResponse:
        """Parse a native tool-calling response from OpenAI or Ollama format."""
        # OpenAI format
        if "choices" in data:
            msg = data["choices"][0].get("message", {})
            text = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc = tool_calls[0]  # Take only the first
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                return ChatResponse(
                    text=text,
                    tool_call=ToolCall(
                        name=func.get("name", ""),
                        arguments=args,
                        raw_id=tc.get("id", ""),
                    ),
                    raw=data,
                )
            return ChatResponse(text=text, raw=data)

        # Ollama format
        if "message" in data:
            msg = data["message"]
            text = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc = tool_calls[0]
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                return ChatResponse(
                    text=text,
                    tool_call=ToolCall(
                        name=func.get("name", ""),
                        arguments=args,
                    ),
                    raw=data,
                )
            return ChatResponse(text=text, raw=data)

        return ChatResponse()

    # ------------------------------------------------------------------
    # Text-based fallback (for models without native tool support)
    # ------------------------------------------------------------------

    def _chat_text_fallback(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        _req,
    ) -> ChatResponse:
        """Fall back to text-mode: send messages without tools param, parse text."""
        if self.server_type == "openai":
            response = self._call_openai(messages, _req)
        else:
            response = self._call_ollama(messages, _req)

        if response.is_empty:
            return response

        # Try to parse a tool call from the text
        tool_call = self._extract_tool_call_from_text(response.text)
        if tool_call:
            return ChatResponse(text=response.text, tool_call=tool_call)

        return response

    def _extract_tool_call_from_text(self, text: str) -> Optional[ToolCall]:
        """Parse tool calls from freeform LLM text output (fallback mode)."""
        tool_names = [
            'file_tree', 'file_read', 'file_write', 'file_edit',
            'bash', 'list_files', 'grep',
            'git_status', 'git_diff', 'revert_file',
            'web_search', 'done',
        ]

        earliest_pos = len(text)
        earliest_name = None
        earliest_raw = None

        for tool_name in tool_names:
            pos = text.find(tool_name + '(')
            if pos != -1 and pos < earliest_pos:
                start = pos + len(tool_name)
                paren_str = self._parse_balanced_parens(text, start)
                if paren_str is not None:
                    earliest_pos = pos
                    earliest_name = tool_name
                    earliest_raw = tool_name + paren_str

        if earliest_name and earliest_raw:
            args = self._parse_text_args(earliest_raw)
            return ToolCall(name=earliest_name, arguments=args)

        # Check for explicit done phrases
        text_lower = text.lower().strip()
        done_phrases = [
            'task complete', 'task completed', 'all done', 'all changes done',
            'all tasks complete', 'work is done', 'work is complete',
        ]
        if any(phrase in text_lower for phrase in done_phrases):
            return ToolCall(name="done", arguments={"message": "Task completed"})

        return None

    @staticmethod
    def _parse_text_args(call_str: str) -> Dict[str, Any]:
        """Parse key=value arguments from a text tool call string."""
        match = re.match(r'\w+\((.*)\)$', call_str, re.DOTALL)
        if not match:
            return {}

        args_str = match.group(1)
        kwargs: Dict[str, Any] = {}
        i = 0

        while i < len(args_str):
            while i < len(args_str) and args_str[i] in ' ,\t\n':
                i += 1
            if i >= len(args_str):
                break

            key_match = re.match(r'(\w+)\s*=\s*', args_str[i:])
            if not key_match:
                break
            key = key_match.group(1)
            i += key_match.end()

            if i < len(args_str) and args_str[i] in ('"', "'"):
                quote = args_str[i]
                i += 1
                value_chars = []
                while i < len(args_str):
                    if args_str[i] == '\\' and i + 1 < len(args_str):
                        next_ch = args_str[i + 1]
                        if next_ch == 'n':
                            value_chars.append('\n')
                        elif next_ch == 't':
                            value_chars.append('\t')
                        else:
                            value_chars.append(next_ch)
                        i += 2
                    elif args_str[i] == quote:
                        i += 1
                        break
                    else:
                        value_chars.append(args_str[i])
                        i += 1
                kwargs[key] = ''.join(value_chars)
            else:
                val_match = re.match(r'([^,\s]*)', args_str[i:])
                if val_match:
                    kwargs[key] = val_match.group(1)
                    i += val_match.end()

        return kwargs

    # ------------------------------------------------------------------
    # Low-level API calls (shared by generate and chat fallback)
    # ------------------------------------------------------------------

    def _call_openai(self, messages: List[Dict[str, Any]], _req) -> ChatResponse:
        """Call OpenAI-compatible /v1/chat/completions (no tools)."""
        logger = logging.getLogger("coding-agent")
        try:
            resp = self._request_with_retry(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.num_predict,
                    "stream": False,
                },
                timeout=300,
            )
            if resp.status_code != 200:
                body = resp.text[:500]
                logger.error("OpenAI-compat error %s: %s", resp.status_code, body)
                return ChatResponse()
            data = resp.json()
            text = data["choices"][0]["message"].get("content") or ""
            return ChatResponse(text=text, raw=data)
        except Exception as e:
            logger.error("OpenAI-compat LLM call failed: %s", e)
            return ChatResponse()

    def _call_ollama(self, messages: List[Dict[str, Any]], _req) -> ChatResponse:
        """Call Ollama /api/chat (NOT /api/generate — chat supports roles + tools)."""
        logger = logging.getLogger("coding-agent")
        try:
            resp = self._request_with_retry(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.num_predict,
                    },
                },
                timeout=300,
            )
            if resp.status_code != 200:
                body = resp.text[:500]
                logger.error("Ollama error %s: %s", resp.status_code, body)
                return ChatResponse()
            data = resp.json()
            text = data.get("message", {}).get("content") or ""
            return ChatResponse(text=text, raw=data)
        except Exception as e:
            logger.error("Ollama LLM call failed: %s", e)
            return ChatResponse()

    # ------------------------------------------------------------------
    # Legacy helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def extract_tool_calls(self, text: str) -> list:
        """Legacy: extract tool call strings from text. Returns list of strings."""
        tc = self._extract_tool_call_from_text(text)
        if tc:
            return [tc.to_call_string()]
        return []

    @staticmethod
    def _parse_balanced_parens(text: str, start: int) -> Optional[str]:
        """Parse from an opening '(' to its matching ')', respecting quotes."""
        if start >= len(text) or text[start] != '(':
            return None

        depth = 0
        in_quote = None
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