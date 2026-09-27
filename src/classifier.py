"""Action classification: map a single tool-call sample to an option label."""


def trial_to_choice(trial, available_tools):
    """Convert a single trial result to a tool choice string.

    Responses with no tool call map to "_no_call", unparseable tool-call
    syntax to "_malformed", a recognized tool name to that name, and any
    other tool name to "_other".
    """
    if trial.get("is_malformed"):
        return "_malformed"
    elif trial.get("direct_answer") or not trial.get("tool_calls"):
        return "_no_call"
    else:
        name = trial["tool_calls"][0]["name"]
        return name if name in available_tools else "_other"
