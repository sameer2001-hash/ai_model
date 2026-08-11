"""
diet_pipeline.py
Core ML pipeline: blood work image -> structured lab values -> condition
classification -> diet recommendation. Kept separate from the web layer
(backend.py) so it can also be run/tested from the command line.
"""

import os
import re
import json
import base64
import logging

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE_MB = 20  # Groq's per-request image limit

VISION_MODEL = "qwen/qwen3.6-27b"
AGENT_MODEL = "openai/gpt-oss-20b"

vision_llm = ChatGroq(
    model=VISION_MODEL,
    temperature=0,
    reasoning_effort="none",
)
agent_llm = ChatGroq(model=AGENT_MODEL, temperature=0)



# Image handling
def encode_image(image_path: str) -> tuple[str, str]:
    """Validate and encode an image file to base64. Returns (base64_data, mime_type)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format '.{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}")

    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"Image is {size_mb:.1f}MB — exceeds {MAX_FILE_SIZE_MB}MB limit")

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return image_data, mime


def _strip_json_fence(text: str) -> str:
    # Remove Qwen's <think>...</think> reasoning block, if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text)



# 1: extract structured lab values from the image


EXTRACTION_PROMPT = """
This is a blood work report. Read every test result visible in the image and
return ONLY a JSON object (no markdown fences, no commentary) shaped exactly
like this:

{
  "tests": [
    {
      "name": "Total Cholesterol",
      "value": "220",
      "unit": "mg/dL",
      "normal_range": "125-200",
      "status": "high"
    }
  ],
  "overall_condition": "high_cholesterol"
}

Rules:
- "status" must be one of: "normal", "high", "low".
- "overall_condition" must be one of: "normal", "high_cholesterol", "high_sugar".
  Pick whichever single category best summarizes the most clinically significant
  flagged value. If nothing is flagged, use "normal".
- Include every test row you can read, even normal ones.
- If a value is illegible, omit that row rather than guessing.
"""


def analyze_blood_work(image_path: str) -> dict:
    """Send a blood work image to the vision model and return structured JSON."""
    image_data, mime = encode_image(image_path)

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
        {"type": "text", "text": EXTRACTION_PROMPT},
    ])

    logger.info(f"Sending {image_path} ({len(image_data)} b64 chars) to vision model")
    response = vision_llm.invoke([message])
    raw = _strip_json_fence(response.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Vision model did not return valid JSON: {raw[:300]}")
        raise ValueError("Could not read structured results from the report image.")

    parsed.setdefault("tests", [])
    parsed.setdefault("overall_condition", "normal")
    return parsed



# 2 diet recommendation tool (unchanged logic, used by the agent)


DIET_PLANS = {
    "high_cholesterol": {
        "eat":        ["fruits", "vegetables", "whole grains", "lean protein"],
        "do_not_eat": ["red meat", "fried food", "full-fat dairy", "processed snacks"],
    },
    "high_sugar": {
        "eat":        ["vegetables", "whole grains", "legumes", "nuts"],
        "do_not_eat": ["white rice", "white sugar", "junk food", "sugary drinks"],
    },
    "normal": {
        "eat":        ["vegetables", "fruits", "whole grains", "lean protein"],
        "do_not_eat": ["excessive sugar", "processed food", "trans fats"],
    },
}


@tool
def get_diet_recommendation(condition: str) -> dict:
    """Given a health condition, returns a diet plan. Condition must be one of: normal, high_cholesterol, high_sugar."""
    if condition not in DIET_PLANS:
        logger.warning(f"Unknown condition '{condition}', defaulting to 'normal'")
    return DIET_PLANS.get(condition, DIET_PLANS["normal"])


SYSTEM_PROMPT = """
You are a nutrition-education assistant, not a medical provider.
You will be given already-extracted blood work values and an overall condition
category. Call get_diet_recommendation with that condition, then present the
plan clearly. Always remind the user this is general dietary information, not
medical advice, and that they should consult a doctor for diagnosis or
treatment decisions.
"""

diet_agent = create_agent(
    agent_llm,
    tools=[get_diet_recommendation],
    system_prompt=SYSTEM_PROMPT,
)



# Func

def run_full_analysis(image_path: str) -> dict:
    """
    Full pipeline: image -> structured lab values -> diet plan.
    Returns a dict ready to hand straight to the frontend as JSON.
    """
    extraction = analyze_blood_work(image_path)
    condition = extraction["overall_condition"]

    agent_result = diet_agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    f"Extracted condition category: {condition}. "
                    f"Lab values: {json.dumps(extraction['tests'])}. "
                    "Retrieve and present the diet plan for this condition."
                )
            )
        ]
    })

    return {
        "tests": extraction["tests"],
        "overall_condition": condition,
        "diet_plan": DIET_PLANS.get(condition, DIET_PLANS["normal"]),
        "agent_summary": agent_result["messages"][-1].content,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "blood_work.png"
    try:
        result = run_full_analysis(path)
        print(json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError) as e:
        logger.error(e)
