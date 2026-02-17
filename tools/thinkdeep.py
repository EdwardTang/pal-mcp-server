"""
ThinkDeep Workflow Tool - Extended Reasoning with Systematic Investigation

This tool provides step-by-step deep thinking capabilities using a systematic workflow approach.
It enables comprehensive analysis of complex problems with expert validation at completion.

Key Features:
- Systematic step-by-step thinking process
- Multi-step analysis with evidence gathering
- Confidence-based investigation flow
- Expert analysis integration with external models
- Support for focused analysis areas (architecture, performance, security, etc.)
- Confidence-based workflow optimization
"""

import logging
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from config import TEMPERATURE_CREATIVE
from systemprompts import THINKDEEP_PROMPT
from tools.shared.base_models import WorkflowRequest

from .workflow.base import WorkflowTool

logger = logging.getLogger(__name__)


# Lightweight adaptation of SELF-DISCOVER reasoning modules for engineering analysis.
# This keeps the tool practical while surfacing an explicit reasoning plan.
REASONING_MODULE_LIBRARY = [
    {
        "id": 1,
        "name": "problem_decomposition",
        "description": "Break the problem into smaller, testable sub-problems.",
        "tags": {"decompose", "scope", "module", "subsystem", "boundary"},
    },
    {
        "id": 2,
        "name": "assumption_analysis",
        "description": "Identify and verify assumptions behind the proposed approach.",
        "tags": {"assumption", "constraint", "premise", "dependency"},
    },
    {
        "id": 3,
        "name": "critical_thinking",
        "description": "Challenge the current approach and look for logical gaps.",
        "tags": {"risk", "gap", "flaw", "tradeoff", "challenge"},
    },
    {
        "id": 4,
        "name": "systems_thinking",
        "description": "Evaluate how changes affect neighboring components and flows.",
        "tags": {"architecture", "integration", "system", "dependency", "coupling"},
    },
    {
        "id": 5,
        "name": "risk_analysis",
        "description": "Assess reliability, safety, and failure-mode risks.",
        "tags": {"risk", "failure", "security", "regression", "stability"},
    },
    {
        "id": 6,
        "name": "performance_analysis",
        "description": "Analyze latency, throughput, memory, and scaling pressure.",
        "tags": {"performance", "latency", "throughput", "memory", "scale"},
    },
    {
        "id": 7,
        "name": "verification_plan",
        "description": "Define concrete validation criteria and acceptance checks.",
        "tags": {"test", "validation", "verify", "assertion", "acceptance"},
    },
    {
        "id": 8,
        "name": "alternative_paths",
        "description": "Generate alternative implementations and compare tradeoffs.",
        "tags": {"alternative", "option", "tradeoff", "approach"},
    },
    {
        "id": 9,
        "name": "step_by_step_plan",
        "description": "Produce a precise sequence of execution steps.",
        "tags": {"plan", "step", "sequence", "implementation"},
    },
    {
        "id": 10,
        "name": "long_term_maintainability",
        "description": "Evaluate operational burden and long-term maintainability.",
        "tags": {"maintainability", "operational", "debt", "evolution"},
    },
]


class ThinkDeepWorkflowRequest(WorkflowRequest):
    """Request model for thinkdeep workflow tool with comprehensive investigation capabilities"""

    # Core workflow parameters
    step: str = Field(description="Current work step content and findings")
    step_number: int = Field(description="Current step number (starts at 1)", ge=1)
    total_steps: int = Field(description="Estimated total steps needed", ge=1)
    next_step_required: bool = Field(description="Whether another step is needed")
    findings: str = Field(
        description="Discoveries: insights, connections, implications, evidence. "
        "Document contradictions to earlier assumptions. Update past findings."
    )

    # Investigation tracking
    files_checked: list[str] = Field(
        default_factory=list,
        description="All files examined (absolute paths). Include ruled-out files.",
    )
    relevant_files: list[str] = Field(
        default_factory=list,
        description="Files relevant to problem/goal (absolute paths). Include root cause, solution, key insights.",
    )
    relevant_context: list[str] = Field(
        default_factory=list,
        description="Key concepts/methods: 'concept_name' or 'ClassName.methodName'. Focus on core insights, decision points.",
    )
    hypothesis: Optional[str] = Field(
        default=None,
        description="Current theory based on evidence. Revise in later steps.",
    )

    # Analysis metadata
    issues_found: list[dict] = Field(
        default_factory=list,
        description="Issues with dict: 'severity' (critical/high/medium/low), 'description'.",
    )
    confidence: str = Field(
        default="low",
        description="exploring/low/medium/high/very_high/almost_certain/certain. CRITICAL: 'certain' PREVENTS external validation.",
    )

    # Expert analysis configuration - keep these fields available for configuring the final assistant model
    # in expert analysis (commented out exclude=True)
    temperature: Optional[float] = Field(
        default=None,
        description="Creative thinking temp (0-1, default 0.7)",
        ge=0.0,
        le=1.0,
    )
    thinking_mode: Optional[str] = Field(
        default=None,
        description="Depth: minimal/low/medium/high/max. Default 'high'.",
    )
    # Context files and investigation scope
    problem_context: Optional[str] = Field(
        default=None,
        description="Additional context about problem/goal. Be expressive.",
    )
    focus_areas: Optional[list[str]] = Field(
        default=None,
        description="Focus aspects (architecture, performance, security, etc.)",
    )
    # DeepThink-style controls (adapted from optillm deepthink extension).
    deepthink_samples: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of reasoning samples to simulate for uncertainty routing (1-10).",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for uncertainty routing: high confidence -> majority_vote, else greedy.",
    )
    enable_self_discover: bool = Field(
        default=True,
        description="Enable SELF-DISCOVER-style reasoning structure generation.",
    )
    reasoning_modules_limit: int = Field(
        default=7,
        ge=3,
        le=15,
        description="Maximum reasoning modules to include in the generated reasoning plan.",
    )


class ThinkDeepTool(WorkflowTool):
    """
    ThinkDeep Workflow Tool - Systematic Deep Thinking Analysis

    Provides comprehensive step-by-step thinking capabilities with expert validation.
    Uses workflow architecture for systematic investigation and analysis.
    """

    name = "thinkdeep"
    description = (
        "Performs multi-stage investigation and reasoning for complex problem analysis. "
        "Use for architecture decisions, complex bugs, performance challenges, and security analysis. "
        "Provides systematic hypothesis testing, evidence-based investigation, and expert validation."
    )

    def __init__(self):
        """Initialize the ThinkDeep workflow tool"""
        super().__init__()
        # Storage for request parameters to use in expert analysis
        self.stored_request_params = {}

    def get_name(self) -> str:
        """Return the tool name"""
        return self.name

    def get_description(self) -> str:
        """Return the tool description"""
        return self.description

    def get_model_category(self) -> "ToolModelCategory":
        """Return the model category for this tool"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_workflow_request_model(self):
        """Return the workflow request model for this tool"""
        return ThinkDeepWorkflowRequest

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema using WorkflowSchemaBuilder with thinkdeep-specific overrides."""
        from .workflow.schema_builders import WorkflowSchemaBuilder

        # ThinkDeep workflow-specific field overrides
        thinkdeep_field_overrides = {
            "problem_context": {
                "type": "string",
                "description": "Additional context about problem/goal. Be expressive.",
            },
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Focus aspects (architecture, performance, security, etc.)",
            },
            "deepthink_samples": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "DeepThink-style sample count for uncertainty routing metadata (1-10).",
            },
            "confidence_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence threshold for routing decision (majority_vote vs greedy).",
            },
            "enable_self_discover": {
                "type": "boolean",
                "description": "Enable SELF-DISCOVER-style reasoning module planning.",
            },
            "reasoning_modules_limit": {
                "type": "integer",
                "minimum": 3,
                "maximum": 15,
                "description": "Maximum reasoning modules in the generated structure.",
            },
        }

        # Use WorkflowSchemaBuilder with thinkdeep-specific tool fields
        return WorkflowSchemaBuilder.build_schema(
            tool_specific_fields=thinkdeep_field_overrides,
            model_field_schema=self.get_model_field_schema(),
            auto_mode=self.is_effective_auto_mode(),
            tool_name=self.get_name(),
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for this workflow tool"""
        return THINKDEEP_PROMPT

    def get_default_temperature(self) -> float:
        """Return default temperature for deep thinking"""
        return TEMPERATURE_CREATIVE

    def get_default_thinking_mode(self) -> str:
        """Return default thinking mode for thinkdeep"""
        from config import DEFAULT_THINKING_MODE_THINKDEEP

        return DEFAULT_THINKING_MODE_THINKDEEP

    def customize_workflow_response(self, response_data: dict, request, **kwargs) -> dict:
        """
        Customize the workflow response for thinkdeep-specific needs
        """
        # Store request parameters for later use in expert analysis
        self.stored_request_params = {}
        try:
            self.stored_request_params["temperature"] = request.temperature
        except AttributeError:
            self.stored_request_params["temperature"] = None

        try:
            self.stored_request_params["thinking_mode"] = request.thinking_mode
        except AttributeError:
            self.stored_request_params["thinking_mode"] = None
        self.stored_request_params["deepthink_strategy"] = self._build_deepthink_strategy(request)

        # Add thinking-specific context to response
        response_data.update(
            {
                "thinking_status": {
                    "current_step": request.step_number,
                    "total_steps": request.total_steps,
                    "files_checked": len(request.files_checked),
                    "relevant_files": len(request.relevant_files),
                    "thinking_confidence": request.confidence,
                    "analysis_focus": request.focus_areas or ["general"],
                    "deepthink": self.stored_request_params["deepthink_strategy"],
                }
            }
        )

        # Add thinking_complete field for final steps (test expects this)
        if not request.next_step_required:
            response_data["thinking_complete"] = True

            # Add complete_thinking summary (test expects this)
            response_data["complete_thinking"] = {
                "steps_completed": len(self.work_history),
                "final_confidence": request.confidence,
                "relevant_context": list(self.consolidated_findings.relevant_context),
                "key_findings": self.consolidated_findings.findings,
                "issues_identified": self.consolidated_findings.issues_found,
                "files_analyzed": list(self.consolidated_findings.relevant_files),
                "deepthink_strategy": self.stored_request_params.get("deepthink_strategy"),
            }

        # Add thinking-specific completion message based on confidence
        if request.confidence == "certain":
            response_data["completion_message"] = (
                "Deep thinking analysis is complete with high certainty. "
                "All aspects have been thoroughly considered and conclusions are definitive."
            )
        elif not request.next_step_required:
            response_data["completion_message"] = (
                "Deep thinking analysis phase complete. Expert validation will provide additional insights and recommendations."
            )

        return response_data

    def _build_deepthink_strategy(self, request) -> dict[str, Any]:
        """
        Build an optillm-inspired DeepThink strategy:
        - SELF-DISCOVER-like module selection and reasoning structure
        - Uncertainty-routed decision metadata (majority_vote vs greedy)
        """
        enable_self_discover = bool(getattr(request, "enable_self_discover", True))
        modules_limit = int(getattr(request, "reasoning_modules_limit", 7) or 7)
        modules_limit = max(3, min(15, modules_limit))
        samples = int(getattr(request, "deepthink_samples", 3) or 3)
        samples = max(1, min(10, samples))
        confidence_threshold = float(getattr(request, "confidence_threshold", 0.7) or 0.7)
        confidence_threshold = max(0.0, min(1.0, confidence_threshold))

        task_text = self._compose_task_text(request)
        selected_modules = []
        reasoning_structure = None
        if enable_self_discover:
            selected_modules = self._select_reasoning_modules(task_text, self._get_focus_areas(request), modules_limit)
            reasoning_structure = self._build_reasoning_structure(task_text, selected_modules)

        uncertainty = self._estimate_uncertainty_route(request, confidence_threshold)

        return {
            "enable_self_discover": enable_self_discover,
            "deepthink_samples": samples,
            "confidence_threshold": confidence_threshold,
            "selected_modules": selected_modules,
            "reasoning_structure": reasoning_structure,
            "uncertainty_routing": uncertainty,
        }

    def _compose_task_text(self, request) -> str:
        parts: list[str] = []
        for field_name in ("step", "findings", "problem_context"):
            value = getattr(request, field_name, None)
            if value:
                parts.append(str(value))
        focus = self._get_focus_areas(request)
        if focus:
            parts.append(" ".join(focus))
        return " ".join(parts).strip()

    def _select_reasoning_modules(self, task_text: str, focus_areas: list[str], limit: int) -> list[dict[str, Any]]:
        """
        Heuristic SELF-DISCOVER adaptation.
        Scores module relevance using request text and focus areas.
        """
        haystack = f"{task_text} {' '.join(focus_areas)}".lower()
        tokens = {token for token in re.split(r"[^a-z0-9_]+", haystack) if token}
        scored: list[tuple[int, dict[str, Any]]] = []

        for module in REASONING_MODULE_LIBRARY:
            all_tags = set(module["tags"])
            all_tags.add(module["name"])
            score = 0
            for tag in all_tags:
                if tag in haystack or tag in tokens:
                    score += 1
            if score > 0:
                scored.append((score, module))

        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        chosen = [module for score, module in scored][:limit]

        if not chosen:
            default_names = {"problem_decomposition", "critical_thinking", "step_by_step_plan", "verification_plan"}
            chosen = [m for m in REASONING_MODULE_LIBRARY if m["name"] in default_names][:limit]

        return [
            {
                "id": module["id"],
                "name": module["name"],
                "description": module["description"],
            }
            for module in chosen
        ]

    def _build_reasoning_structure(self, task_text: str, modules: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert selected modules into a concrete, execution-oriented reasoning plan.
        """
        execution_plan = []
        for idx, module in enumerate(modules, start=1):
            execution_plan.append(
                {
                    "step": idx,
                    "module": module["name"],
                    "objective": module["description"],
                }
            )

        task_summary = task_text[:240] + ("..." if len(task_text) > 240 else "")
        return {
            "task_summary": task_summary or "General technical analysis task",
            "selected_module_count": len(modules),
            "execution_plan": execution_plan,
            "output_contract": [
                "State assumptions",
                "Present evidence and tradeoffs",
                "Identify risks and mitigations",
                "Provide concrete next actions",
            ],
        }

    def _estimate_uncertainty_route(self, request, threshold: float) -> dict[str, Any]:
        """
        Estimate confidence and route using an uncertainty-routed-CoT style policy.
        """
        confidence_map = {
            "exploring": 0.25,
            "low": 0.35,
            "medium": 0.55,
            "high": 0.72,
            "very_high": 0.82,
            "almost_certain": 0.90,
            "certain": 0.97,
        }
        confidence_label = str(getattr(request, "confidence", "low") or "low")
        score = confidence_map.get(confidence_label, 0.5)

        findings_text = str(getattr(request, "findings", "") or "")
        if len(findings_text) >= 200:
            score += 0.05
        if len(getattr(request, "relevant_context", []) or []) >= 2:
            score += 0.05
        if len(getattr(request, "relevant_files", []) or []) >= 2:
            score += 0.03

        # Penalize highly divergent context signals when available.
        context_items = [str(item) for item in (getattr(request, "relevant_context", []) or []) if item]
        if len(context_items) >= 2:
            pairwise = []
            bounded_items = context_items[:12]
            lowered = [item.lower()[:500] for item in bounded_items]
            for i, first in enumerate(lowered):
                for second in lowered[i + 1 :]:
                    pairwise.append(SequenceMatcher(None, first, second).ratio())
            if pairwise:
                avg_similarity = sum(pairwise) / len(pairwise)
                if avg_similarity < 0.25:
                    score -= 0.04

        score = max(0.0, min(1.0, score))
        synthetic_answer_votes = max(1, min(int(getattr(request, "deepthink_samples", 3) or 3), 10))
        seed_material = "|".join(
            [
                confidence_label,
                findings_text[:200],
                f"{threshold:.3f}",
                str(synthetic_answer_votes),
            ]
        )
        rng = random.Random(seed_material)
        sampled_decisions = []
        for _ in range(synthetic_answer_votes):
            sampled_score = max(0.0, min(1.0, score + rng.uniform(-0.08, 0.08)))
            sampled_decisions.append("majority_vote" if sampled_score >= threshold else "greedy")
        vote_counter = Counter(sampled_decisions)
        routing_decision = vote_counter.most_common(1)[0][0]

        return {
            "confidence_score": round(score, 3),
            "routing_decision": routing_decision,
            "routing_rationale": (
                f"confidence_score {score:.3f} {'>=' if score >= threshold else '<'} "
                f"threshold {threshold:.3f}"
            ),
            "synthetic_vote_distribution": dict(vote_counter),
            "sampling_mode": "seeded_monte_carlo",
        }

    def should_skip_expert_analysis(self, request, consolidated_findings) -> bool:
        """
        ThinkDeep tool skips expert analysis when the CLI agent has "certain" confidence.
        """
        return request.confidence == "certain" and not request.next_step_required

    def get_completion_status(self) -> str:
        """ThinkDeep tools use thinking-specific status."""
        return "deep_thinking_complete_ready_for_implementation"

    def get_completion_data_key(self) -> str:
        """ThinkDeep uses 'complete_thinking' key."""
        return "complete_thinking"

    def get_final_analysis_from_request(self, request):
        """ThinkDeep tools use 'findings' field."""
        return request.findings

    def get_skip_expert_analysis_status(self) -> str:
        """Status when skipping expert analysis for certain confidence."""
        return "skipped_due_to_certain_thinking_confidence"

    def get_skip_reason(self) -> str:
        """Reason for skipping expert analysis."""
        return "Expressed 'certain' confidence in the deep thinking analysis - no additional validation needed"

    def get_completion_message(self) -> str:
        """Message for completion without expert analysis."""
        return "Deep thinking analysis complete with certain confidence. Proceed with implementation based on the analysis."

    def customize_expert_analysis_prompt(self, base_prompt: str, request, file_content: str = "") -> str:
        """
        Customize the expert analysis prompt for deep thinking validation
        """
        thinking_context = f"""
DEEP THINKING ANALYSIS VALIDATION

You are reviewing a comprehensive deep thinking analysis completed through systematic investigation.
Your role is to validate the thinking process, identify any gaps, challenge assumptions, and provide
additional insights or alternative perspectives.

ANALYSIS SCOPE:
- Problem Context: {self._get_problem_context(request)}
- Focus Areas: {', '.join(self._get_focus_areas(request))}
- Investigation Confidence: {request.confidence}
- Steps Completed: {request.step_number} of {request.total_steps}

THINKING SUMMARY:
{request.findings}

KEY INSIGHTS AND CONTEXT:
{', '.join(request.relevant_context) if request.relevant_context else 'No specific context identified'}

VALIDATION OBJECTIVES:
1. Assess the depth and quality of the thinking process
2. Identify any logical gaps, missing considerations, or flawed assumptions
3. Suggest alternative approaches or perspectives not considered
4. Validate the conclusions and recommendations
5. Provide actionable next steps for implementation

Be thorough but constructive in your analysis. Challenge the thinking where appropriate,
but also acknowledge strong insights and valid conclusions.
"""

        if file_content:
            thinking_context += f"\n\nFILE CONTEXT:\n{file_content}"

        return f"{thinking_context}\n\n{base_prompt}"

    def get_expert_analysis_instructions(self) -> str:
        """
        Return instructions for expert analysis specific to deep thinking validation
        """
        return (
            "DEEP THINKING ANALYSIS IS COMPLETE. You MUST now summarize and present ALL thinking insights, "
            "alternative approaches considered, risks and trade-offs identified, and final recommendations. "
            "Clearly prioritize the top solutions or next steps that emerged from the analysis. "
            "Provide concrete, actionable guidance based on the deep thinking—make it easy for the user to "
            "understand exactly what to do next and how to implement the best solution."
        )

    # Override hook methods to use stored request parameters for expert analysis

    def get_request_temperature(self, request) -> float:
        """Use stored temperature from initial request."""
        try:
            stored_params = self.stored_request_params
            if stored_params and stored_params.get("temperature") is not None:
                return stored_params["temperature"]
        except AttributeError:
            pass
        return super().get_request_temperature(request)

    def get_request_thinking_mode(self, request) -> str:
        """Use stored thinking mode from initial request."""
        try:
            stored_params = self.stored_request_params
            if stored_params and stored_params.get("thinking_mode") is not None:
                return stored_params["thinking_mode"]
        except AttributeError:
            pass
        return super().get_request_thinking_mode(request)

    def _get_problem_context(self, request) -> str:
        """Get problem context from request. Override for custom context handling."""
        try:
            return request.problem_context or "General analysis"
        except AttributeError:
            return "General analysis"

    def _get_focus_areas(self, request) -> list[str]:
        """Get focus areas from request. Override for custom focus area handling."""
        try:
            return request.focus_areas or ["comprehensive analysis"]
        except AttributeError:
            return ["comprehensive analysis"]

    def get_required_actions(
        self, step_number: int, confidence: str, findings: str, total_steps: int, request=None
    ) -> list[str]:
        """
        Return required actions for the current thinking step.
        """
        actions = []

        if step_number == 1:
            actions.extend(
                [
                    "Begin systematic thinking analysis",
                    "Identify key aspects and assumptions to explore",
                    "Establish initial investigation approach",
                ]
            )
        elif confidence == "low":
            actions.extend(
                [
                    "Continue gathering evidence and insights",
                    "Test initial hypotheses",
                    "Explore alternative perspectives",
                ]
            )
        elif confidence == "medium":
            actions.extend(
                [
                    "Deepen analysis of promising approaches",
                    "Validate key assumptions",
                    "Consider implementation challenges",
                ]
            )
        elif confidence == "high":
            actions.extend(
                [
                    "Refine and validate key findings",
                    "Explore edge cases and limitations",
                    "Document assumptions and trade-offs",
                ]
            )
        elif confidence == "very_high":
            actions.extend(
                [
                    "Synthesize findings into cohesive recommendations",
                    "Validate conclusions against all evidence",
                    "Prepare comprehensive implementation guidance",
                ]
            )
        elif confidence == "almost_certain":
            actions.extend(
                [
                    "Finalize recommendations with high confidence",
                    "Document any remaining minor uncertainties",
                    "Prepare for expert analysis or implementation",
                ]
            )
        else:  # certain
            actions.append("Analysis complete - ready for implementation")

        return actions

    def should_call_expert_analysis(self, consolidated_findings, request=None) -> bool:
        """
        Determine if expert analysis should be called based on confidence and completion.
        """
        if request:
            try:
                # Don't call expert analysis if confidence is "certain"
                if request.confidence == "certain":
                    return False
            except AttributeError:
                pass

        # Call expert analysis if investigation is complete (when next_step_required is False)
        if request:
            try:
                return not request.next_step_required
            except AttributeError:
                pass

        # Fallback: call expert analysis if we have meaningful findings
        return (
            len(consolidated_findings.relevant_files) > 0
            or len(consolidated_findings.findings) >= 2
            or len(consolidated_findings.issues_found) > 0
        )

    def prepare_expert_analysis_context(self, consolidated_findings) -> str:
        """
        Prepare context for expert analysis specific to deep thinking.
        """
        context_parts = []

        context_parts.append("DEEP THINKING ANALYSIS SUMMARY:")
        context_parts.append(f"Steps completed: {len(consolidated_findings.findings)}")
        context_parts.append(f"Final confidence: {consolidated_findings.confidence}")

        if consolidated_findings.findings:
            context_parts.append("\nKEY FINDINGS:")
            for i, finding in enumerate(consolidated_findings.findings, 1):
                context_parts.append(f"{i}. {finding}")

        if consolidated_findings.relevant_context:
            context_parts.append(f"\nRELEVANT CONTEXT:\n{', '.join(consolidated_findings.relevant_context)}")

        # Get hypothesis from latest hypotheses entry if available
        if consolidated_findings.hypotheses:
            latest_hypothesis = consolidated_findings.hypotheses[-1].get("hypothesis", "")
            if latest_hypothesis:
                context_parts.append(f"\nFINAL HYPOTHESIS:\n{latest_hypothesis}")

        if consolidated_findings.issues_found:
            context_parts.append(f"\nISSUES IDENTIFIED: {len(consolidated_findings.issues_found)} issues")
            for issue in consolidated_findings.issues_found:
                context_parts.append(
                    f"- {issue.get('severity', 'unknown')}: {issue.get('description', 'No description')}"
                )

        deepthink_strategy = self.stored_request_params.get("deepthink_strategy")
        if deepthink_strategy:
            context_parts.append("\nDEEPTHINK STRATEGY:")
            context_parts.append(f"- routing: {deepthink_strategy['uncertainty_routing']['routing_decision']}")
            context_parts.append(f"- score: {deepthink_strategy['uncertainty_routing']['confidence_score']}")
            context_parts.append(
                f"- selected_modules: {len(deepthink_strategy.get('selected_modules') or [])}"
            )

        return "\n".join(context_parts)

    def get_step_guidance_message(self, request) -> str:
        """
        Generate guidance for the next step in thinking analysis
        """
        if request.next_step_required:
            next_step_number = request.step_number + 1

            if request.confidence == "certain":
                guidance = (
                    f"Your thinking analysis confidence is CERTAIN. Consider if you truly need step {next_step_number} "
                    f"or if you should complete the analysis now with expert validation."
                )
            elif request.confidence == "almost_certain":
                guidance = (
                    f"Your thinking analysis confidence is ALMOST_CERTAIN. For step {next_step_number}, consider: "
                    f"finalizing recommendations, documenting any minor uncertainties, or preparing for implementation."
                )
            elif request.confidence == "very_high":
                guidance = (
                    f"Your thinking analysis confidence is VERY_HIGH. For step {next_step_number}, consider: "
                    f"synthesis of all findings, comprehensive validation, or creating implementation roadmap."
                )
            elif request.confidence == "high":
                guidance = (
                    f"Your thinking analysis confidence is HIGH. For step {next_step_number}, consider: "
                    f"exploring edge cases, documenting trade-offs, or stress-testing key assumptions."
                )
            elif request.confidence == "medium":
                guidance = (
                    f"Your thinking analysis confidence is MEDIUM. For step {next_step_number}, focus on: "
                    f"deepening insights, exploring alternative approaches, or gathering additional evidence."
                )
            else:  # low or exploring
                guidance = (
                    f"Your thinking analysis confidence is {request.confidence.upper()}. For step {next_step_number}, "
                    f"continue investigating: gather more evidence, test hypotheses, or explore different angles."
                )

            # Add specific thinking guidance based on progress
            if request.step_number == 1:
                guidance += (
                    " Consider: What are the key assumptions? What evidence supports or contradicts initial theories? "
                    "What alternative approaches exist?"
                )
            elif request.step_number >= request.total_steps // 2:
                guidance += (
                    " Consider: Synthesis of findings, validation of conclusions, identification of implementation "
                    "challenges, and preparation for expert analysis."
                )

            return guidance
        else:
            return "Thinking analysis is ready for expert validation and final recommendations."

    def format_final_response(self, assistant_response: str, request, **kwargs) -> dict:
        """
        Format the final response from the assistant for thinking analysis
        """
        response_data = {
            "thinking_analysis": assistant_response,
            "analysis_metadata": {
                "total_steps_completed": request.step_number,
                "final_confidence": request.confidence,
                "files_analyzed": len(request.relevant_files),
                "key_insights": len(request.relevant_context),
                "issues_identified": len(request.issues_found),
            },
        }

        # Add completion status
        if request.confidence == "certain":
            response_data["completion_status"] = "analysis_complete_with_certainty"
        else:
            response_data["completion_status"] = "analysis_complete_pending_validation"

        return response_data

    def format_step_response(
        self,
        assistant_response: str,
        request,
        status: str = "pause_for_thinkdeep",
        continuation_id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Format intermediate step responses for thinking workflow
        """
        response_data = super().format_step_response(assistant_response, request, status, continuation_id, **kwargs)

        # Add thinking-specific step guidance
        step_guidance = self.get_step_guidance_message(request)
        response_data["thinking_guidance"] = step_guidance

        # Add analysis progress indicators
        response_data["analysis_progress"] = {
            "step_completed": request.step_number,
            "remaining_steps": max(0, request.total_steps - request.step_number),
            "confidence_trend": request.confidence,
            "investigation_depth": "expanding" if request.next_step_required else "finalizing",
        }

        return response_data

    # Required abstract methods from BaseTool
    def get_request_model(self):
        """Return the thinkdeep workflow-specific request model."""
        return ThinkDeepWorkflowRequest

    async def prepare_prompt(self, request) -> str:
        """Not used - workflow tools use execute_workflow()."""
        return ""  # Workflow tools use execute_workflow() directly
