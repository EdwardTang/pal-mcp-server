"""DeepThink-inspired behavior tests for ThinkDeepTool."""

from tools.thinkdeep import ThinkDeepTool, ThinkDeepWorkflowRequest


def _make_request(**overrides):
    payload = {
        "step": "Analyze authentication architecture and performance tradeoffs",
        "step_number": 1,
        "total_steps": 2,
        "next_step_required": True,
        "findings": "Initial findings: session handling and token validation paths need comparison.",
        "confidence": "medium",
        "focus_areas": ["architecture", "security", "performance"],
    }
    payload.update(overrides)
    return ThinkDeepWorkflowRequest(**payload)


def test_thinkdeep_schema_exposes_deepthink_fields():
    tool = ThinkDeepTool()
    schema = tool.get_input_schema()
    properties = schema["properties"]

    assert "deepthink_samples" in properties
    assert "confidence_threshold" in properties
    assert "enable_self_discover" in properties
    assert "reasoning_modules_limit" in properties


def test_build_deepthink_strategy_includes_reasoning_structure():
    tool = ThinkDeepTool()
    request = _make_request(enable_self_discover=True, reasoning_modules_limit=5)

    strategy = tool._build_deepthink_strategy(request)

    assert strategy["enable_self_discover"] is True
    assert strategy["reasoning_structure"] is not None
    assert len(strategy["selected_modules"]) >= 1
    assert len(strategy["selected_modules"]) <= 5
    assert strategy["uncertainty_routing"]["routing_decision"] in {"majority_vote", "greedy"}


def test_uncertainty_routing_prefers_majority_vote_for_high_confidence():
    tool = ThinkDeepTool()
    request = _make_request(confidence="very_high", confidence_threshold=0.7, deepthink_samples=4)

    strategy = tool._build_deepthink_strategy(request)
    routing = strategy["uncertainty_routing"]

    assert routing["routing_decision"] == "majority_vote"
    assert routing["confidence_score"] >= 0.7


def test_uncertainty_routing_prefers_greedy_for_low_confidence():
    tool = ThinkDeepTool()
    request = _make_request(confidence="exploring", confidence_threshold=0.8, deepthink_samples=3)

    strategy = tool._build_deepthink_strategy(request)
    routing = strategy["uncertainty_routing"]

    assert routing["routing_decision"] == "greedy"
    assert routing["confidence_score"] < 0.8
