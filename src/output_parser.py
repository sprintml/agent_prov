"""Parse tool calls from model outputs across different formats.

Handles:
- Qwen format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- Llama pythonic format: [func_name(arg1=val1, ...)]
- Fallback JSON format: {"name": "...", "arguments": {...}}
"""

import ast
import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """Normalized representation of a parsed tool call."""
    name: str
    arguments: dict
    raw: str  # original text that was parsed

    def __post_init__(self):
        # Ensure arguments survive JSON serialisation - some parsers can
        # produce values (e.g. Ellipsis from a literal ``...`` emitted by
        # Hammer) that would otherwise crash ``json.dump`` when writing the
        # results file. Kept here so every construction site is covered.
        self.arguments = _sanitize_arguments(self.arguments)


@dataclass
class ParseResult:
    """Full parsing result for one model generation."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_output: str = ""
    has_tool_call: bool = False
    is_malformed: bool = False  # attempted but failed to parse
    direct_answer: bool = False  # model answered without calling tools
    error: str = ""


def _strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from model output.

    Handles both complete blocks and unclosed <think> tags (when the model
    ran out of tokens mid-thought).
    """
    # Remove complete <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove unclosed <think> block (model ran out of tokens mid-thought)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def _normalize_bpe_artifacts(text: str) -> str:
    """Repair GPT-2-style byte-level BPE decode artifacts.

    Some tokenizer/decoder combinations (notably DeepSeek-R1-Distill-Llama-8B
    in this project) leak raw byte-level whitespace markers when
    ``tokenizer.decode`` is called with ``skip_special_tokens=False`` without
    ``clean_up_tokenization_spaces=True``:

        Ġ (U+0120) -> ASCII space
        Ċ (U+010A) -> newline

    These markers corrupt JSON payloads (``"name":Ġ"X"`` is not valid JSON),
    so we normalize them before downstream parsing. For already-clean text
    this is a no-op.
    """
    if "Ġ" in text or "Ċ" in text:
        text = text.replace("Ġ", " ").replace("Ċ", "\n")
    return text


def parse_output(raw_text: str, model_family: str) -> ParseResult:
    """Parse tool calls from raw model output.

    Args:
        raw_text: Raw decoded output from the model.
        model_family: Model family name - determines parsing strategy.

    Returns:
        ParseResult with extracted tool calls and metadata.
    """
    result = ParseResult(raw_output=raw_text)

    # Normalise GPT-2 BPE decode artifacts (Ġ->space, Ċ->newline) before any
    # other processing - otherwise JSON payloads like ``"name":Ġ"X"`` fail
    # downstream parsers. This affects r1-distill-llama-8b most visibly.
    raw_text = _normalize_bpe_artifacts(raw_text)

    # Strip <think> blocks before parsing (e.g. DeepSeek-R1 reasoning).
    # Keep raw_output intact for debugging; parse the stripped text.
    text = _strip_think_blocks(raw_text)

    if model_family in ("qwen", "qwen3", "hermes"):
        result = _parse_qwen(text, result)
    elif model_family == "hammer":
        # Hammer2.1 emits markdown-fenced Python-literal tool calls that the
        # standard Qwen parser cannot read. Try the Hammer-specific parser
        # first, then fall back to the Qwen template for the rare cases
        # where the model does emit a proper <tool_call> wrapper.
        result = _parse_hammer(text, result)
        if not result.has_tool_call and not result.is_malformed:
            result = _parse_qwen(text, result)
    elif model_family == "llama":
        result = _parse_llama(text, result)
    elif model_family == "phi":
        result = _parse_phi(text, result)
    elif model_family == "llama31":
        result = _parse_llama31(text, result)
    elif model_family == "mistral":
        result = _parse_mistral(text, result)
    elif model_family == "nemotron":
        result = _parse_nemotron(text, result)
    elif model_family == "internlm":
        result = _parse_internlm(text, result)
    elif model_family == "gorilla":
        # Gorilla outputs pythonic function calls: func_name(arg=val)
        result = _parse_llama(text, result)
    elif model_family == "smollm3":
        # SmolLM3's chat template emits the Hermes-style
        # <tool_call>{...}</tool_call> wrapper (rendered verbatim in the
        # system prompt as emit instructions). Parser-wise it's a Qwen match.
        result = _parse_qwen(text, result)
    elif model_family in ("deepseek-r1", "glm-z1", "phi-reasoning"):
        # Reasoning-model family cohort: output format is variable because
        # the training data mixes several tool-call conventions. Try the
        # family's preferred parser first (GLM's name-on-line format for
        # glm-z1, R1-fenced JSON for the others), then every other parser,
        # then the generic JSON fallback. We reset `is_malformed` between
        # parsers so an earlier partial-match detection does not block
        # later parsers from succeeding.
        if model_family == "glm-z1":
            # Run GLM-format parser on the un-stripped text so we can match
            # the `<|assistant|>function_name\n{JSON}<|observation|>` shape
            # before <think> stripping loses the anchor tokens.
            result = _parse_glm(raw_text, result)
        if not result.has_tool_call:
            # Reset before trying the general reasoning-cohort parsers so
            # the GLM-specific partial-match doesn't poison later parsers.
            result.is_malformed = False
            result.error = ""
            any_malformed = False
            for parser in [_parse_r1_fenced_json, _parse_qwen, _parse_mistral,
                            _parse_internlm, _parse_phi, _parse_llama,
                            _parse_llama31, _parse_json_fallback]:
                result = parser(text, result)
                if result.has_tool_call:
                    break
                any_malformed = any_malformed or result.is_malformed
                # Reset malformed so next parser gets a clean shot
                result.is_malformed = False
                result.error = ""
            if not result.has_tool_call and any_malformed:
                result.is_malformed = True
                result.error = "Multiple parsers detected partial tool call attempts"
    else:
        # For unknown families (e.g. gemma, deepseek, glm with fallback formatting),
        # try all parsers in order, resetting is_malformed between attempts so
        # a false positive from an early parser doesn't block later ones.
        any_malformed = False
        for parser in [_parse_qwen, _parse_mistral, _parse_nemotron, _parse_internlm, _parse_phi, _parse_llama, _parse_llama31, _parse_json_fallback]:
            result = parser(text, result)
            if result.has_tool_call:
                break
            any_malformed = any_malformed or result.is_malformed
            result.is_malformed = False
            result.error = ""
        if not result.has_tool_call and any_malformed:
            result.is_malformed = True
            result.error = "Multiple parsers detected partial tool call attempts"

    # Validate extracted tool names - reject BPE artifacts (Ġ, Ċ) and
    # names that aren't valid identifiers (contain spaces, punctuation, etc.)
    if result.has_tool_call:
        valid_calls = [tc for tc in result.tool_calls if re.match(r'^[a-zA-Z_]\w*$', tc.name)]
        if len(valid_calls) < len(result.tool_calls):
            if not valid_calls:
                result.is_malformed = True
                result.error = "All tool calls had invalid names (BPE artifacts)"
            result.tool_calls = valid_calls
            result.has_tool_call = len(valid_calls) > 0

    # If no tool call was found or attempted, it's a direct answer
    if not result.has_tool_call and not result.is_malformed:
        result.direct_answer = True

    return result


def _parse_qwen(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse Qwen-style tool calls: <tool_call>{"name": ..., "arguments": ...}</tool_call>"""

    # Pattern for Qwen tool calls
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, raw_text, re.DOTALL)

    if not matches:
        # Also try without XML tags - sometimes Qwen outputs bare JSON
        json_pattern = r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}'
        matches = re.findall(json_pattern, raw_text, re.DOTALL)

    if not matches:
        # Check if there was an attempt (partial match suggests malformation)
        if '<tool_call>' in raw_text or '"name"' in raw_text:
            result.is_malformed = True
            result.error = "Partial tool call pattern detected but could not parse"
        return result

    for match in matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=match))
            result.has_tool_call = True
        except (json.JSONDecodeError, TypeError) as e:
            result.is_malformed = True
            result.error = f"JSON parse error: {e}"

    return result


def _parse_llama(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse Llama-style tool calls.

    Llama models emit tool calls in several shapes depending on the version:

    1. Llama-3.2 JSON variant (the current default for 1B/3B instruct tunes):

           <|python_tag|>{"type": "function", "function": "NAME",
                           "parameters": {"arg": "val"}}<|eom_id|>

       Note that ``"function"`` holds the *name* as a string - this is
       different from Llama-3.1 (``"name"``) and from Hammer (which nests
       a dict under ``"function"``).

    2. Llama-3.1 style JSON (handled by ``_parse_llama31`` when the family
       is ``llama31``; we also accept it here as a fallback because
       Llama-3.2 occasionally falls into that shape).

    3. Legacy pythonic calls: ``[func_name(arg="val")]`` or ``func_name(arg="val")``.

    We try (1)/(2) first - they're cheap and unambiguous - and only fall
    back to the pythonic regex if no JSON was found.
    """
    # ------------------------------------------------------------------
    # Attempt 1: JSON after <|python_tag|> (Llama-3.2 and 3.1 shapes)
    # ------------------------------------------------------------------
    json_text = raw_text
    if "<|python_tag|>" in json_text:
        json_text = json_text.split("<|python_tag|>", 1)[1]
    json_text = re.sub(r"<\|eom_id\|>|<\|eot_id\|>|<\|end_of_text\|>", "", json_text).strip()

    for obj_src in _extract_balanced_json_objects(json_text):
        try:
            data = json.loads(obj_src)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        # Name key - Llama-3.2 uses "function" (with a string value),
        # Llama-3.1 uses "name". Accept either. Also tolerate a nested
        # {"function": {"name": ...}} shape for robustness.
        name = data.get("name")
        if not isinstance(name, str) or not name:
            fn_field = data.get("function")
            if isinstance(fn_field, str):
                name = fn_field
            elif isinstance(fn_field, dict):
                name = fn_field.get("name", "")
        if not isinstance(name, str) or not name:
            continue
        arguments = data.get("parameters", data.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        if not arguments and isinstance(data.get("function"), dict):
            nested = data["function"]
            arguments = nested.get("arguments", nested.get("parameters", {})) or {}
        result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=obj_src))
        result.has_tool_call = True

    if result.has_tool_call:
        return result

    # ------------------------------------------------------------------
    # Attempt 2: legacy pythonic function-call syntax
    # ------------------------------------------------------------------
    pattern = r'(?:\[)?(\w+)\(([^)]*)\)(?:\])?'
    matches = re.finditer(pattern, raw_text)

    found_any = False
    for match in matches:
        func_name = match.group(1)
        args_str = match.group(2).strip()
        found_any = True

        # Skip common false positives (Python builtins that appear in inline code)
        if func_name in ('print', 'len', 'str', 'int', 'float', 'list', 'dict', 'type',
                         'range', 'sum', 'map', 'filter', 'sorted', 'enumerate', 'zip',
                         'max', 'min', 'abs', 'round', 'set', 'tuple', 'bool', 'open',
                         'input', 'format', 'iter', 'next', 'reversed', 'any', 'all',
                         'isinstance', 'hasattr', 'getattr', 'super', 'repr', 'hash'):
            continue

        try:
            arguments = _parse_pythonic_args(args_str)
            result.tool_calls.append(
                ToolCall(name=func_name, arguments=arguments, raw=match.group(0))
            )
            result.has_tool_call = True
        except Exception as e:
            result.is_malformed = True
            result.error = f"Pythonic arg parse error: {e}"

    if not found_any and not result.has_tool_call:
        if "<|python_tag|>" in raw_text:
            result.is_malformed = True
            result.error = "Llama python_tag detected but no tool call extracted"
        elif any(kw in raw_text.lower() for kw in ['function', 'tool_call', 'call']):
            # Only mark as malformed if it really looks like an attempt
            if re.search(r'\w+\(', raw_text):
                result.is_malformed = True
                result.error = "Possible malformed pythonic call"

    return result


def _parse_llama31(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse Llama-3.1 style tool calls: JSON after <|python_tag|> token.

    Format: <|python_tag|>{"name": "func", "parameters": {"arg": "val"}}
    Note: Llama-3.1 uses "parameters" instead of "arguments".
    """
    # Strip the python_tag prefix if present
    text = raw_text
    if '<|python_tag|>' in text:
        text = text.split('<|python_tag|>', 1)[1]

    # Remove end tokens
    text = re.sub(r'<\|eom_id\|>|<\|eot_id\|>|<\|end_of_text\|>', '', text).strip()

    # Try to parse as JSON
    json_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:parameters|arguments)"\s*:\s*\{.*?\}\s*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if not matches:
        # Check for partial attempt
        if '<|python_tag|>' in raw_text or ('"name"' in text and '"parameters"' in text):
            result.is_malformed = True
            result.error = "Llama-3.1 tool call pattern detected but could not parse"
        return result

    for match in matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            # Llama-3.1 uses "parameters" key, not "arguments"
            arguments = data.get("parameters", data.get("arguments", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=match))
            result.has_tool_call = True
        except (json.JSONDecodeError, TypeError) as e:
            result.is_malformed = True
            result.error = f"Llama-3.1 JSON parse error: {e}"

    return result


def _parse_phi(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse Phi-4 style tool calls.

    Phi-4-mini emits tool calls in two shapes depending on the prompt:

    1. Wrapped:   ``<|tool_call|>{"name": ..., "arguments": ...}<|/tool_call|>``
    2. Bare:      ``[{"name": ..., "arguments": {...}}]<|end|>`` (plain JSON
       array, no special token wrapper).

    We try the wrapped form first, then fall through to a bare-array
    extraction.
    """
    # Pattern for Phi tool calls with special tokens
    pattern = r'<\|tool_call\|>\s*(\{.*?\})\s*(?:<\|/tool_call\|>|<\|end\|>|$)'
    matches = re.findall(pattern, raw_text, re.DOTALL)

    if matches:
        for match in matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=match))
                result.has_tool_call = True
            except (json.JSONDecodeError, TypeError) as e:
                result.is_malformed = True
                result.error = f"Phi JSON parse error: {e}"
        return result

    if '<|tool_call|>' in raw_text:
        result.is_malformed = True
        result.error = "Phi tool_call tag found but could not parse content"
        return result

    # ------------------------------------------------------------------
    # Bare JSON-array form: [{"name": "...", "arguments": {...}}]
    # ------------------------------------------------------------------
    text = re.sub(r'<\|end\|>|<\|endoftext\|>', '', raw_text).strip()
    arr_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                arguments = item.get("arguments", item.get("parameters", {}))
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                if name:
                    result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=str(item)))
                    result.has_tool_call = True

    return result


def _parse_mistral(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse Mistral-style tool calls: [TOOL_CALLS] [{"name": ..., "arguments": {...}}]

    Also handles API variants where the provider strips [TOOL_CALLS] and
    returns raw JSON arrays, optionally prefixed with <s>, [ACTION],
    or [ACTIONS].
    """

    # Normalize: strip provider-added prefixes that wrap the JSON array
    normalized = raw_text.strip()
    for prefix in ("<s>", "[ACTION]", "[ACTIONS]"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
    # Remove trailing </s> or [/ACTION] etc.
    for suffix in ("</s>", "[/ACTION]", "[/ACTIONS]", "[/s>"):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()

    # Try raw JSON array (API format: no [TOOL_CALLS] prefix)
    if not result.tool_calls and normalized.startswith("[{"):
        try:
            data = json.loads(normalized)
            if isinstance(data, list) and all(isinstance(d, dict) and "name" in d for d in data):
                for item in data:
                    name = item.get("name", "")
                    arguments = item.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=str(item)))
                    result.has_tool_call = True
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Pattern: [TOOL_CALLS] followed by a JSON array
    if '[TOOL_CALLS]' in raw_text:
        text = raw_text.split('[TOOL_CALLS]', 1)[1].strip()
        # Try to parse as JSON array
        try:
            # Find the array
            arr_match = re.search(r'\[.*\]', text, re.DOTALL)
            if arr_match:
                data = json.loads(arr_match.group(0))
                if isinstance(data, list):
                    for item in data:
                        name = item.get("name", "")
                        arguments = item.get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=str(item)))
                        result.has_tool_call = True
                    return result
        except (json.JSONDecodeError, TypeError):
            pass
        result.is_malformed = True
        result.error = "Mistral [TOOL_CALLS] found but could not parse"
        return result

    # Also try single JSON object after [TOOL_CALLS] marker
    pattern = r'\[TOOL_CALLS\]\s*(\{.*?\})'
    matches = re.findall(pattern, raw_text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=match))
            result.has_tool_call = True
        except (json.JSONDecodeError, TypeError) as e:
            result.is_malformed = True
            result.error = f"Mistral JSON parse error: {e}"

    return result


def _parse_nemotron(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse Nemotron-style tool calls: <TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>"""

    pattern = r'<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>'
    match = re.search(pattern, raw_text, re.DOTALL)

    if not match:
        # Check for partial attempt
        if '<TOOLCALL>' in raw_text:
            result.is_malformed = True
            result.error = "Nemotron <TOOLCALL> tag found but could not parse content"
        return result

    try:
        data = json.loads(match.group(1))
        if isinstance(data, list):
            for item in data:
                if item is None:
                    continue
                name = item.get("name", "")
                arguments = item.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=str(item)))
                result.has_tool_call = True
    except (json.JSONDecodeError, TypeError) as e:
        result.is_malformed = True
        result.error = f"Nemotron JSON parse error: {e}"

    return result


def _parse_internlm(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse InternLM-style tool calls.

    Format: <|action_start|><|plugin|>{"name": "...", "parameters": {...}}<|action_end|>
    Or sometimes just bare JSON with "name" and "parameters" keys.
    """
    # Pattern with action tags
    pattern = r'<\|action_start\|><\|plugin\|>\s*(\{.*?\})\s*<\|action_end\|>'
    matches = re.findall(pattern, raw_text, re.DOTALL)

    if not matches:
        # Try without end tag
        pattern2 = r'<\|action_start\|><\|plugin\|>\s*(\{.*?\})'
        matches = re.findall(pattern2, raw_text, re.DOTALL)

    if not matches:
        # Check for partial attempt
        if '<|action_start|>' in raw_text or '<|plugin|>' in raw_text:
            result.is_malformed = True
            result.error = "InternLM action tags found but could not parse"
        return result

    for match in matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("parameters", data.get("arguments", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=match))
            result.has_tool_call = True
        except (json.JSONDecodeError, TypeError) as e:
            result.is_malformed = True
            result.error = f"InternLM JSON parse error: {e}"

    return result


def _parse_json_fallback(raw_text: str, result: ParseResult) -> ParseResult:
    """Generic fallback: look for JSON objects with "name"/"tool_name" and "arguments" keys.

    Used for models that receive tool descriptions via text injection (e.g. Gemma,
    DeepSeek-R1-Distill) and respond with raw JSON.

    Also accepts the **name-as-key** shape ``{"<tool_name>": {"<arg>": <val>}}``
    emitted by reasoning-only distillations (R1-Distill-Qwen 1.5B, DistilQwen2.5)
    that haven't been tool-SFT'd and invent their own format. Enabled only as a
    last resort (after the "name"/"tool_name" variants fail) to avoid
    false-positives on non-tool JSON.
    """
    # Strip markdown code fences - R1-Distill models often wrap JSON in ```json ... ```
    text = re.sub(r'```(?:json|python)?\s*\n?', '', raw_text)

    # Try multiple key variants: "name", "tool_name"
    for name_key in ["name", "tool_name"]:
        pattern = r'\{\s*"' + name_key + r'"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            try:
                name = match.group(1)
                arguments = json.loads(match.group(2))
                result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=match.group(0)))
                result.has_tool_call = True
            except (json.JSONDecodeError, TypeError) as e:
                result.is_malformed = True
                result.error = f"JSON fallback parse error: {e}"

        if result.has_tool_call or result.is_malformed:
            return result

    # Name-as-key fallback: {"<tool_name>": {...args...}}. Only accept when
    # the object has exactly one top-level key whose value is a dict, and
    # the key looks like a valid identifier and isn't one of the reserved
    # wrapper words used by other tool-call formats.
    _RESERVED_WRAPPER_KEYS = {
        "name", "tool_name", "function", "parameters", "arguments", "type",
        "tool_call", "tool_calls", "action", "actions", "result", "response",
        "answer", "content", "role", "message", "messages", "id",
    }
    for obj_src in _extract_balanced_json_objects(text):
        try:
            data = json.loads(obj_src)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or len(data) != 1:
            continue
        key = next(iter(data))
        value = data[key]
        if not isinstance(key, str):
            continue
        if key.lower() in _RESERVED_WRAPPER_KEYS:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if not isinstance(value, dict):
            continue
        result.tool_calls.append(ToolCall(name=key, arguments=value, raw=obj_src))
        result.has_tool_call = True
    if result.has_tool_call:
        return result

    # Check for partial JSON tool call attempt
    if any(k in text for k in ['"name"', '"tool_name"']) and '"arguments"' in text:
        result.is_malformed = True
        result.error = "JSON with name/arguments keys detected but could not parse"

    return result


def _extract_balanced_json_objects(text: str) -> list[str]:
    """Return every top-level ``{...}`` JSON object substring in ``text``.

    A simple brace-counter that tracks whether we're inside a double-quoted
    string (with backslash escape support). More robust than a regex for
    multi-line, nested tool-call payloads like those emitted by R1-distill
    models.
    """
    objects = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != '{':
            i += 1
            continue
        start = i
        depth = 0
        in_str = False
        escape = False
        while i < n:
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        objects.append(text[start:i + 1])
                        i += 1
                        break
            i += 1
        else:
            # unterminated - give up on this object
            break
    return objects


def _sanitize_arguments(args: dict) -> dict:
    """Ensure tool-call arguments are JSON-serialisable.

    The Hammer and pythonic parsers can produce values like ``Ellipsis`` when
    the model emits literal ``...`` as a placeholder (e.g.
    ``{'payment_id': ...}``). These are valid Python but not valid JSON, and
    would crash ``json.dump`` on the results file. We walk the dict and
    replace any non-serialisable leaf with its string representation.
    """
    def _clean(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): _clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clean(v) for v in value]
        return repr(value)

    if not isinstance(args, dict):
        return {}
    return {str(k): _clean(v) for k, v in args.items()}


def _py_literal_to_json(text: str) -> str:
    """Rewrite a Python-literal-style dict/list as a JSON string.

    Hammer2.1 emits tool calls with single quotes (Python repr style) rather
    than JSON. We convert quotes and Python constants *character by character*
    with a small state machine so that apostrophes inside string values
    (``{'message': "it's fine"}``) don't get clobbered.

    Recognises only: ``'...'`` -> ``"..."``, ``None`` -> ``null``, ``True`` /
    ``False`` -> ``true`` / ``false``. Intentionally narrower than
    ``ast.literal_eval`` - we avoid any form of code evaluation.
    """
    out = []
    i = 0
    n = len(text)
    in_sq = False  # inside a single-quoted string we're rewriting
    in_dq = False  # inside an already-double-quoted string
    while i < n:
        ch = text[i]
        if in_sq:
            if ch == "\\" and i + 1 < n:
                out.append(text[i:i + 2])
                i += 2
                continue
            if ch == "'":
                out.append('"')
                in_sq = False
                i += 1
                continue
            if ch == '"':
                out.append('\\"')
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        if in_dq:
            if ch == "\\" and i + 1 < n:
                out.append(text[i:i + 2])
                i += 2
                continue
            if ch == '"':
                in_dq = False
            out.append(ch)
            i += 1
            continue
        # Not inside any string.
        if ch == "'":
            in_sq = True
            out.append('"')
            i += 1
            continue
        if ch == '"':
            in_dq = True
            out.append(ch)
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            if word == "None":
                out.append("null")
            elif word == "True":
                out.append("true")
            elif word == "False":
                out.append("false")
            else:
                out.append(word)
            i = j
            continue
        # Handle Python ellipsis literal "..." - JSON has no equivalent,
        # so we map it to null and rely on ``_sanitize_arguments`` at the
        # ToolCall level to keep things consistent.
        if ch == "." and text[i:i + 3] == "...":
            out.append("null")
            i += 3
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_hammer(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse MadeAgents Hammer2.1 tool calls.

    Observed output format::

        ``` [{'type': 'function',
              'function': {'name': 'get_weather',
                           'arguments': {'location': 'Tokyo'}}}] ```<|im_end|>

    Quirks vs the standard Qwen template:
      * wrapped in markdown code fences
      * single quotes (Python repr style) - we convert via ``_py_literal_to_json``
      * nested shape: the tool name may sit under ``item["function"]["name"]``
        rather than top-level ``item["name"]``
    """
    text = raw_text
    text = re.sub(r"```(?:python|json)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|end\|>", "", text).strip()

    arr_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if not arr_match:
        return result
    arr_src = arr_match.group(0)

    # Try JSON directly first (Hammer-0.5b sometimes emits valid JSON).
    data = None
    try:
        data = json.loads(arr_src)
    except (json.JSONDecodeError, TypeError):
        pass
    if data is None:
        try:
            data = json.loads(_py_literal_to_json(arr_src))
        except (json.JSONDecodeError, TypeError):
            last = arr_src.rfind("}]")
            if last != -1:
                trimmed = arr_src[: last + 2]
                try:
                    data = json.loads(_py_literal_to_json(trimmed))
                except (json.JSONDecodeError, TypeError):
                    result.is_malformed = True
                    result.error = "Hammer JSON-from-literal parse failed"
                    return result
            else:
                result.is_malformed = True
                result.error = "Hammer JSON-from-literal parse failed"
                return result

    if not isinstance(data, list):
        data = [data]

    for item in data:
        if not isinstance(item, dict):
            continue
        if "name" in item and isinstance(item["name"], str):
            name = item["name"]
            arguments = item.get("arguments", item.get("parameters", {}))
        else:
            fn = item.get("function", {})
            if isinstance(fn, dict):
                name = fn.get("name", "")
                arguments = fn.get("arguments", fn.get("parameters", {}))
            else:
                continue
        if not isinstance(arguments, dict):
            arguments = {}
        if not name:
            continue
        result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=str(item)))
        result.has_tool_call = True

    return result


def _parse_glm(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse GLM-Z1 (and GLM-4) tool calls.

    GLM-Z1 emits tool calls as::

        <|assistant|>function_name
        {"arg": "value", ...}<|observation|>

    The function name sits on its own line; the JSON argument object follows.
    The ``<|assistant|>`` prefix and ``<|observation|>`` terminator may or
    may not be present depending on chat-template rendering. We accept the
    format with or without the wrappers.
    """
    text = raw_text
    # Drop GLM-specific special tokens that would otherwise confuse the
    # identifier/JSON split. We keep a trailing marker to anchor the end
    # of the tool call region so we don't match trailing commentary.
    text = re.sub(r"<\|assistant\|>|<\|user\|>|<\|system\|>", "\n", text)
    # Truncate at the first observation/end marker - anything after is
    # post-tool-call narration or a placeholder.
    for sep in ("<|observation|>", "<|user|>", "<|endoftext|>"):
        idx = text.find(sep)
        if idx != -1:
            text = text[:idx]
            break

    # Pattern: (identifier-line) followed by (JSON object). Identifier
    # may include double underscores (mcp__email__send_message style).
    # Only match when the JSON body clearly looks like tool arguments.
    pattern = re.compile(
        r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*(\{.*?\})\s*$",
        re.DOTALL,
    )
    # Prefer the *last* match - reasoning preambles sometimes list tool
    # names as examples before the committed call.
    matches = list(pattern.finditer(text))
    if not matches:
        return result
    m = matches[-1]
    name = m.group(1)
    try:
        args = json.loads(m.group(2))
    except (json.JSONDecodeError, TypeError):
        result.is_malformed = True
        result.error = "GLM JSON argument parse failed"
        return result
    if not isinstance(args, dict):
        return result
    result.tool_calls.append(ToolCall(name=name, arguments=args, raw=m.group(0)))
    result.has_tool_call = True
    return result


def _parse_r1_fenced_json(raw_text: str, result: ParseResult) -> ParseResult:
    """Parse markdown-fenced JSON tool calls as emitted by R1-style models.

    DeepSeek-R1-Distill-Qwen/Llama write tool calls like::

        ```json
        {
          "name": "get_current_conditions",
          "arguments": {"location": "Tokyo"}
        }
        ```

    The ``<think>...</think>`` reasoning block is stripped earlier by
    ``_strip_think_blocks``, so here we only look for the fenced JSON
    block. If no fence is present, we fall back to any balanced
    ``{"name": ..., "arguments"|"parameters": ...}`` object found in the
    tail of the text.

    This parser is intentionally forgiving: models sometimes show an
    example schema first and the *real* call at the end, so we always
    prefer the **last** JSON object that looks like a valid tool call.
    """
    text = re.sub(
        r"<｜end▁of▁sentence｜>|<\|end_of_sentence\|>|<\|im_end\|>|<\|eot_id\|>",
        "",
        raw_text,
    )

    # Markdown-fenced blocks first.
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not candidates:
        # Fall back to any balanced top-level JSON object whose body looks
        # like a tool call.
        candidates = [
            obj for obj in _extract_balanced_json_objects(text)
            if '"name"' in obj and ('"arguments"' in obj or '"parameters"' in obj)
        ]

    # Take the last three candidates, newest-first, so reasoning preambles
    # that reference schemas don't win over the final committed call.
    chosen = None
    for obj_src in reversed(candidates[-5:]):
        try:
            data = json.loads(obj_src)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and (data.get("name") or data.get("function")):
            chosen = (obj_src, data)
            break

    if chosen is None:
        return result

    obj_src, data = chosen
    name = data.get("name", "")
    if not name and isinstance(data.get("function"), str):
        name = data["function"]
    if not name and isinstance(data.get("function"), dict):
        name = data["function"].get("name", "")
    if not name:
        return result

    arguments = data.get("arguments", data.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    result.tool_calls.append(ToolCall(name=name, arguments=arguments, raw=obj_src))
    result.has_tool_call = True
    return result


def _parse_pythonic_args(args_str: str) -> dict:
    """Parse pythonic keyword arguments like: query="hello", count=5"""
    if not args_str:
        return {}

    # Try to parse as Python dict via ast
    # Wrap in dict() call for ast parsing
    try:
        tree = ast.parse(f"dict({args_str})", mode='eval')
        # Extract keyword arguments
        call_node = tree.body
        result = {}
        for kw in call_node.keywords:
            key = kw.arg
            # Evaluate the value safely
            value = ast.literal_eval(kw.value)
            result[key] = value
        return result
    except (SyntaxError, ValueError):
        pass

    # Fallback: manual regex parsing for key=value pairs
    result = {}
    # Match key="value" or key=value patterns
    kv_pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\d+(?:\.\d+)?)|(\w+))'
    for m in re.finditer(kv_pattern, args_str):
        key = m.group(1)
        value = m.group(2) or m.group(3) or m.group(4) or m.group(5)
        # Try to convert numeric values
        if value and value.isdigit():
            value = int(value)
        elif value:
            try:
                value = float(value)
            except ValueError:
                pass
        result[key] = value

    return result
