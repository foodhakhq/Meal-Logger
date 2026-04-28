from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import anthropic
import os
import uvicorn
import json
from dotenv import load_dotenv
from fastapi.responses import Response
from fastapi.responses import PlainTextResponse
from openai import OpenAI
import logging
import re


# Load environment variables
load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Claude API key from environment variable
claude_api_key = os.getenv("CLAUDE_API_KEY")
if not claude_api_key:
    raise EnvironmentError("Claude API key not set in environment variable CLAUDE_API_KEY")

# Staging API key from environment variable
staging_api_key = os.getenv("STAGING_API_KEY")
if not staging_api_key:
    raise EnvironmentError("Staging API key not set in environment variable STAGING_API_KEY")


def get_openai_client():
    api_key = os.getenv("STAGING_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("STAGING_OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=api_key)
    return client


claude_client = anthropic.Anthropic(api_key=claude_api_key)

NUTRITIONAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "NutritionalArrayWrapped",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "food_title": {"type": "string"},
                            "servings": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Calories": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Protein": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Carbohydrates": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Total Fat": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Sodium": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Saturated Fat": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Cholesterol": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Sugar": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Calcium": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Iron": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Potassium": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Vitamin C": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Vitamin E": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"},
                            "Vitamin D": {"type": "string", "pattern": "^[0-9]+(\\.[0-9]+)?$"}
                        },
                        "required": [
                            "food_title", "servings", "Calories", "Protein", "Carbohydrates",
                            "Total Fat", "Sodium", "Saturated Fat", "Cholesterol", "Sugar",
                            "Calcium", "Iron", "Potassium", "Vitamin C", "Vitamin E", "Vitamin D"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["items"],
            "additionalProperties": False
        },
        "strict": True
    }
}

class QueryInput(BaseModel):
    query: str


def _extract_top_level_array(raw_text: str) -> str:
    """
    Accepts model JSON as string. Returns a stringified top-level JSON array.
    Handles:
      - {"items": [...]}
      - already-top-level arrays [...]
      - objects that contain a single array value
    Raises HTTPException 500 if no array can be found.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON.")

    # Case 1: already an array
    if isinstance(data, list):
        return json.dumps(data, ensure_ascii=False)

    # Case 2: {"items": [...]}
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return json.dumps(data["items"], ensure_ascii=False)

    # Case 3: object with exactly one array value
    if isinstance(data, dict):
        array_values = [v for v in data.values() if isinstance(v, list)]
        if len(array_values) == 1:
            return json.dumps(array_values[0], ensure_ascii=False)

    raise HTTPException(status_code=500, detail="No array payload found in model output.")


def format_response(final_adjusted_recipe: str) -> str:
    """Format the response as separate JSON objects"""
    # Split into individual JSON objects and filter empty strings
    cleaned = final_adjusted_recipe.replace("```json", "").replace("```", "").strip()
    json_strings = [s for s in cleaned.split('\n\n') if s.strip()]

    formatted_jsons = []
    for json_str in json_strings:
        try:
            # Parse and re-format each JSON object
            parsed = json.loads(json_str)
            formatted = json.dumps(parsed, indent=2)
            formatted_jsons.append(formatted)
        except json.JSONDecodeError:
            continue

    # Join with double newlines
    return '\n\n'.join(formatted_jsons)


def _strip_json_fences(t: str) -> str:
    return t.replace("```json", "").replace("```", "").strip()

def clean_food_title(title: str) -> str:
    patterns = [
        r"^\s*i\s+(just\s+)?(ate|had|have|am\s+eating|am\s+having|drank|consumed)\s+",
        r"^\s*(just\s+)?(ate|had)\s+",
        r"^\s*log\s+",
    ]
    cleaned = title
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else title
    
@app.post("/get_nutritional_info")
async def get_nutritional_info(query_input: QueryInput, authorization: str = Header(...)):
    """
    Validates the provided Authorization header against the PROD_API_KEY
    and uses the Claude API to process the nutritional info.
    """
    # Remove "Bearer " prefix from authorization
    provided_api_key = authorization.replace("Bearer ", "")

    # Compare with prod_api_key
    if provided_api_key != staging_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing credentials")

    example_nutritional_data = """
[   
    {
    "food_title": "Title of the recipe",
    "servings": "1",
    "Calories": "800",
    "Protein": "18.2",
    "Carbohydrates": "106.9",
    "Total Fat": "28.1",
    "Sodium": "380",
    "Saturated Fat": "5.9",
    "Cholesterol": "45",
    "Sugar": "4.8",
    "Calcium": "210",
    "Iron": "6.2",
    "Potassium": "890",
    "Vitamin C": "35",
    "Vitamin E": "3.8",
    "Vitamin D": "0.6"
    }
]
"""

    user_prompt = query_input.query
    system_prompt = f"""
            Accurately calculate the nutritional information of a recipe based on its title or description, defaulting to one serving. If the user specifies a different number of servings, adjust the nutritional values accordingly. Be precise, providing detailed breakdowns (e.g., calories, macronutrients, vitamins, and minerals) for clarity.

            Your Tasks:
            1. Process Multiple Recipes Individually 
              - If the user query contains multiple recipes, treat each one as a separate recipe.  
              - Based on a **detailed** ingredients list, provide nutritional breakdown, and JSON snippet for **each** recipes nested in JSON object array.

            2. Detailed Recipe Ingredients for Nutritional Breakdown
              - For each recipe titles in {user_prompt}, use a detailed list of ingredients.
              - Use exact measurements (cups, tablespoons, etc.) for each ingredient based on the number of servings.
              - Use this ingredient information to calculate the nutritional breakdown of **each** recipes.

            3. Accurate Nutritional Breakdown
              - For each recipe, include a comprehensive breakdown of the key nutrients by taking into account all ingredients:
                      **Calories**
                      **Protein**
                      **Carbohydrate**
                      **Total Fat**
                      **Sodium**
                      **Saturated Fat**
                      **Cholesterol**
                      **Sugar**
                      **Calcium**
                      **Iron**
                      **Potassium**
                      **Vitamin C**
                      **Vitamin E**
                      **Vitamin D**

            4. JSON Subsection as Dropdown
              - For each recipe in {user_prompt}, provide a JSON snippet strictly and exactly in the example format: {example_nutritional_data}
              - Violating this example format structure would lead to critical error.
              - Even for multiple recipe logging maintain the same format and do not add any extra keys.
              - Include all relevant data accurately calculated with detail.

            5. General Instructions
              - Do NOT ask any follow-up questions to the user.
              - Display the JSON response even if you have length constraints while providing JSON recipes.
              - Do not include any additional text or explanations
              - Return response in valid JSON format matching the example structure
              - The "servings" field must reflect the number of servings the user specified for that item. For example, if the user says "2 servings of coke", set "servings" to "2" and calculate all nutritional values for that many servings.
              - If the user does not specify a serving count for an item, default "servings" to "1".

              ###IMPORTANT: Violating this example recipe format structure would lead to critical error. {example_nutritional_data}


            """
    client = get_openai_client()

    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            reasoning_effort="minimal",
            max_completion_tokens=2500,
            response_format=NUTRITIONAL_SCHEMA,
            stream=False,
        )
        content = (resp.choices[0].message.content or "").strip()
        array_json = _extract_top_level_array(content)
        items = json.loads(array_json)
        for item in items:
            if "food_title" in item:
                item["food_title"] = clean_food_title(item["food_title"])
        array_json = json.dumps(items, ensure_ascii=False)
        return PlainTextResponse(content=array_json, status_code=200)

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        # Fallback: try with a different model if GPT-5 models aren't available
        try:
            logger.info("Attempting fallback to gpt-4o-mini...")
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=6000,
                response_format=NUTRITIONAL_SCHEMA,
                temperature=0.2,
                stream=False,
            )
            content = (resp.choices[0].message.content or "").strip()
            array_json = _extract_top_level_array(content)
            items = json.loads(array_json)
            for item in items:
                if "food_title" in item:
                    item["food_title"] = clean_food_title(item["food_title"])
            array_json = json.dumps(items, ensure_ascii=False)
            return PlainTextResponse(content=array_json, status_code=200)

        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate recipe with both GPT-5 and fallback model: {str(e)}"
            )
    # try:
    #     #raise Exception({'type': 'error', 'error': {'details': None, 'type': 'overloaded_error', 'message': 'Overloaded'}})
    #     with claude_client.messages.stream(
    #             model="claude-sonnet-4-6",
    #             max_tokens=8192,
    #             temperature=0,
    #             system=system_prompt,
    #             messages=[
    #                 {
    #                     "role": "user",
    #                     "content": user_prompt
    #                 }
    #             ]
    #     ) as stream:
    #         texts = []
    #         for text in stream.text_stream:
    #             texts.append(text)
    #         final_adjusted_recipe = ''.join(texts)

    #     formatted_response = format_response(final_adjusted_recipe)
    #     return Response(content=formatted_response, media_type="application/json")

    # except Exception as anthropic_error:
    #     # Log the error to see what args look like
    #     logging.error(f"Anthropic error args: {anthropic_error.args}")
    #     error_data = anthropic_error.args[0] if anthropic_error.args else {}

    #     # If error_data is not a dict, try to parse it as JSON
    #     if not isinstance(error_data, dict):
    #         try:
    #             error_data = json.loads(error_data)
    #         except Exception:
    #             error_data = {}

    #     # Also check the error message string as a fallback
    #     error_str = str(anthropic_error)

    #     if error_data.get("error", {}).get("type") == "overloaded_error" or "overloaded_error" in error_str:
    #         logging.warning("Anthropic Claude is overloaded. Falling back to OpenAI...")
    #     # If Claude API fails, fall back to OpenAI API
    #     try:
    #         openai_client = get_openai_client()
    #         # Use the same system and user prompts for OpenAI
    #         print("Openai recipe")
    #         response = openai_client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt}
    #             ],
    #             temperature = 0,
    #             stream=True
    #         )

    #         texts = []
    #         for chunk in response:
    #             delta = chunk.choices[0].delta
    #             content = getattr(delta, "content", "") or ""
    #             texts.append(content)
    #         final_adjusted_recipe = ''.join(texts)
    #         print(final_adjusted_recipe)
    #     except Exception as openai_exception:
    #         raise HTTPException(
    #             status_code=500,
    #             detail=f"Both Claude and OpenAI API calls failed. Claude error: {anthropic_error}; OpenAI error: {openai_exception}"

    #         )
    #     formatted_response = format_response(final_adjusted_recipe)
    #     return Response(content=formatted_response, media_type="application/json")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Meal Logger Service is up and running."}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
