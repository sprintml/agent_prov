"""Format tool schemas and messages using each model's native chat template."""

import json
from typing import Any


def build_tool_schema(tool_config: dict) -> dict:
    """Convert our YAML tool config to the OpenAI-style function schema
    expected by HuggingFace's apply_chat_template.

    Args:
        tool_config: Tool definition from tool_sets.yaml.

    Returns:
        OpenAI-style tool schema dict.
    """
    return {
        "type": "function",
        "function": {
            "name": tool_config["name"],
            "description": tool_config["description"],
            "parameters": tool_config["parameters"],
        },
    }


def build_tools_for_category(category_config: dict) -> list[dict]:
    """Build all tool schemas for a category."""
    return [build_tool_schema(t) for t in category_config["tools"]]


def format_prompt(
    tokenizer,
    user_message: str,
    tools: list[dict],
    system_prompt: str = "You are a helpful assistant. Use the provided tools when needed to answer the user's question.",
    model_family: str = "",
    enable_thinking: bool = False,
    extra_messages: list[dict] | None = None,
    prefill: bool = False,
) -> str:
    """Format a prompt with tools using the model's native chat template.

    This is the key normalization layer: identical tool schemas and user messages
    go in, model-specific formatting comes out.

    Args:
        tokenizer: HuggingFace tokenizer with chat template support.
        user_message: The user's query. Ignored when ``prefill`` is True.
        tools: List of OpenAI-style tool schemas.
        system_prompt: System instruction.
        model_family: Model family name for family-specific formatting.
        enable_thinking: When True, request the model's thinking/reasoning
            mode via ``apply_chat_template(enable_thinking=True)``. Defaults
            to False, which preserves legacy behavior for `runner.py` and
            `classifier.py` callers. Probes that need the thinking dial
            (e.g. S_think_gate) pass True.
        extra_messages: Optional list of messages to splice between the
            system prompt and the final user turn - used by probes that
            require synthetic prior history (S_post_err, S_hierarchy_conflict,
            S_budget). Each entry is an OpenAI-style chat message. When None
            or empty, behavior matches legacy two-message (system + user)
            prompting exactly.
        prefill: When True, treat the last assistant message in
            ``extra_messages`` as a partial response the model should
            *continue*. No new user turn is appended. The tokenizer is
            called with ``continue_final_message=True,
            add_generation_prompt=False``. Used by probes that measure
            mid-trajectory behavior (s_self_verify - does the model
            spontaneously correct a seeded wrong path). Requires the
            last message in ``extra_messages`` to have role ``assistant``.

    Returns:
        Formatted prompt string ready for tokenization.
    """
    # Gorilla OpenFunctions v2 uses a unique <<question>> / <<function>> format
    if model_family == "gorilla":
        if prefill:
            raise ValueError("prefill mode not supported for gorilla family")
        return _gorilla_format(tokenizer, user_message, tools)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if extra_messages:
        # Mistral's chat template merges system prompt into the LAST [INST]
        # turn, not the first. When extra_messages provides multi-turn
        # history, this causes the system prompt to appear after the
        # assistant reply, garbling the final user turn. Fix by merging
        # the system prompt into the first user message instead.
        if (model_family in ("mistral",)
                and any(m.get("role") == "user" for m in extra_messages)):
            first_user_idx = next(
                i for i, m in enumerate(extra_messages)
                if m.get("role") == "user"
            )
            patched = list(extra_messages)
            patched[first_user_idx] = {
                **patched[first_user_idx],
                "content": system_prompt + "\n\n" + patched[first_user_idx].get("content", ""),
            }
            messages = list(patched)
        else:
            messages.extend(extra_messages)

    if prefill:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(
                "format_prompt(prefill=True) requires extra_messages to end "
                "with an assistant message; last role was "
                f"{messages[-1].get('role') if messages else 'none'}"
            )
        # Don't append a new user turn - model continues the last assistant.
    elif (not user_message or not user_message.strip()) and extra_messages:
        # Agentic continuation: empty user_turn + non-empty history means
        # "let the model generate the next assistant turn from the existing
        # trajectory." Used by s_post_err (history ends with a tool-error
        # response; no user ask is involved in the state the probe measures).
        # apply_chat_template with add_generation_prompt=True emits the
        # assistant-start marker after whatever the last history role is.
        pass
    else:
        messages.append({"role": "user", "content": user_message})

    chat_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": not prefill,
    }
    if prefill:
        chat_kwargs["continue_final_message"] = True

    # Pass tools=None when the list is empty. Some chat templates (Llama)
    # inject "Environment: ipython" whenever tools is non-None (even []),
    # which causes the model to emit <|python_tag|> tool-call tokens
    # even when no tools are available.
    effective_tools = tools if tools else None

    try:
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tools=effective_tools,
                enable_thinking=enable_thinking,
                **chat_kwargs,
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking - use standard call
            formatted = tokenizer.apply_chat_template(
                messages,
                tools=effective_tools,
                **chat_kwargs,
            )
    except Exception:
        if prefill:
            # Text-injection fallback doesn't make sense in prefill mode -
            # surface the error so the caller knows this tokenizer doesn't
            # support continue_final_message.
            raise
        # Fallback: some models don't support tools= parameter.
        # Inject tool descriptions into the system prompt manually.
        formatted = _fallback_format(tokenizer, messages, tools)
        return formatted

    # Some tokenizers silently ignore the tools= parameter (e.g. DistilQwen2.5,
    # SuperNova-Lite, DeepSeek-R1). Detect this by checking if any tool name
    # appears in the formatted output; if not, fall back to text injection.
    # Only applies in normal mode - prefill probes typically declare no tools.
    if tools and not prefill:
        tool_names = [t["function"]["name"] for t in tools]
        if not any(name in formatted for name in tool_names):
            formatted = _fallback_format(tokenizer, messages, tools)

    return formatted


def _gorilla_format(tokenizer, user_message: str, tools: list[dict]) -> str:
    """Format prompt for Gorilla OpenFunctions v2.

    Gorilla expects: <<question>> {query} <<function>> {json_array_of_functions}
    Functions use OpenAI-style format (just the function object, not the wrapper).
    See: https://gorilla.cs.berkeley.edu/blogs/7_open_functions_v2.html
    """
    # Extract the function objects from the OpenAI-style wrapper
    functions = [tool["function"] for tool in tools]
    prompt = f"<<question>> {user_message} <<function>> {json.dumps(functions)}"

    # Wrap in chat template so BOS/EOS tokens are correct
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        # Raw fallback if chat template fails
        bos = tokenizer.bos_token or ""
        return f"{bos}{prompt}\n"


def build_tool_text_block(tools: list[dict]) -> str:
    """Build a plain-text tool description block for text-injection mode.

    Used by _fallback_format (local models whose tokenizers lack native tool
    support) and by API adapters (Gemini, OpenRouter) that send tools as text.
    """
    tool_desc_parts = []
    for tool in tools:
        func = tool["function"]
        params = func["parameters"]
        param_strs = []
        for pname, pinfo in params.get("properties", {}).items():
            required = pname in params.get("required", [])
            req_marker = " (required)" if required else " (optional)"
            param_strs.append(f"    - {pname}: {pinfo.get('type', 'string')}{req_marker} - {pinfo.get('description', '')}")
        tool_desc_parts.append(
            f"  {func['name']}: {func['description']}\n"
            f"  Parameters:\n" + "\n".join(param_strs)
        )

    tool_block = "You have access to the following tools:\n\n" + "\n\n".join(tool_desc_parts)
    tool_block += (
        "\n\nTo call a tool, respond with a JSON object: "
        '{{"name": "tool_name", "arguments": {{...}}}}'
    )
    return tool_block


def _fallback_format(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    """Fallback formatting when the tokenizer doesn't support tools= natively.

    Injects tool descriptions into the system message as structured text.
    """
    tool_block = build_tool_text_block(tools)

    messages = messages.copy()
    messages[0] = {
        "role": "system",
        "content": messages[0]["content"] + "\n\n" + tool_block,
    }

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def shuffle_tools(tools: list[dict], rng) -> list[dict]:
    """Shuffle tool order to counterbalance position bias.

    Args:
        tools: List of tool schemas.
        rng: numpy or random RNG instance with .shuffle method.

    Returns:
        A new list with shuffled order.
    """
    shuffled = tools.copy()
    rng.shuffle(shuffled)
    return shuffled
