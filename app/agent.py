from __future__ import annotations

import os
import sys

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.workflow import Edge, Workflow, node
from pydantic import BaseModel, Field

current_dir = os.path.dirname(os.path.abspath(__file__))
shared_dir = os.path.abspath(os.path.join(current_dir, "..", "shared"))
if not os.path.exists(os.path.join(shared_dir, "firestore_client.py")):
    shared_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "shared"))
sys.path.append(shared_dir)
import firestore_client  # noqa: E402
from security_checkpoint import security_checkpoint  # noqa: E402


class VerificationResult(BaseModel):
    is_verified: bool = Field(description="True if image appears genuine")
    verification_score: float = Field(
        description="Score 0.0 to 1.0, above 0.7 is verified"
    )
    flags: list = Field(description="List of concerns found, empty if clean")
    action: str = Field(description="publish, reject, or human_review")


@node
def prepare_verification_input(ctx: Context, node_input: dict):
    """Prepares image + description for Gemini Vision."""
    ctx.state["issue_to_verify"] = node_input
    yield Event(
        output={
            "image_url": node_input.get("photo_url"),
            "description": node_input.get("description"),
            "ward_name": node_input.get("ward_name"),
            "category": node_input.get("category"),
        }
    )


verification_agent = LlmAgent(
    name="image_verifier",
    model="gemini-flash-latest",
    instruction="""You are a civic content verifier
for Hey Hood, a neighborhood accountability app in India.

Analyze the provided image and description.
Determine if this is a genuine civic issue report.

Check for:
1. Is this a real photo or AI generated?
2. Does the image match the description?
3. Does it show a genuine civic problem?
4. Are there signs of manipulation or deception?
5. Is this content appropriate for a civic platform?

Score above 0.7 = verified and safe to publish.
Score below 0.5 = reject or send to human review.
Between 0.5 and 0.7 = human_review.

Be lenient with low quality phone camera photos —
citizens use basic phones. Focus on authenticity
not professional quality.""",
    output_key="verification_result",
    output_schema=VerificationResult,
)


@node
def route_verification_result(ctx: Context, node_input):
    """Routes based on verification score."""
    result = ctx.state.get("verification_result", {})
    action = result.get("action", "human_review")
    score = result.get("verification_score", 0.5)
    issue = ctx.state.get("issue_to_verify", {})

    db = firestore_client.get_db()
    issue_id = issue.get("issue_id")

    if action == "publish" and score >= 0.7:
        db.collection("issues").document(issue_id).update(
            {"verified": True, "verification_score": score}
        )
        yield Event(output=issue, actions=EventActions(route="verified"))

    elif action == "reject" or score < 0.5:
        db.collection("issues").document(issue_id).update(
            {"verified": False, "status": "Rejected", "verification_score": score}
        )
        yield Event(output=issue, actions=EventActions(route="rejected"))

    else:
        db.collection("issues").document(issue_id).update(
            {"verified": False, "status": "Pending Review", "verification_score": score}
        )
        yield Event(output=issue, actions=EventActions(route="human_review"))


root_agent = Workflow(
    name="fake_news_detection_workflow",
    edges=[
        ("START", security_checkpoint),
        Edge(
            from_node=security_checkpoint,
            to_node=prepare_verification_input,
            route="clean",
        ),
        Edge(
            from_node=security_checkpoint,
            to_node=route_verification_result,
            route="human_review",
        ),
        (prepare_verification_input, verification_agent, route_verification_result),
    ],
)

app = App(
    name="fake_news_agent",
    root_agent=root_agent,
)
