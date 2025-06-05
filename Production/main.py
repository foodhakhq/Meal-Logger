from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import anthropic
import os
import uvicorn
import json
from dotenv import load_dotenv
from fastapi.responses import Response
from openai import OpenAI
import logging

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
api_key = os.getenv("API_KEY")
if not api_key:
    raise EnvironmentError("Staging API key not set in environment variable API_KEY")


def get_openai_client():
    api_key = os.getenv("STAGING_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("STAGING_OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=api_key)
    return client


claude_client = anthropic.Anthropic(api_key=claude_api_key)


class QueryInput(BaseModel):
    query: str


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



@app.post("/get_nutritional_info")
async def get_nutritional_info(query_input: QueryInput, authorization: str = Header(...)):
    """
    Validates the provided Authorization header against the PROD_API_KEY
    and uses the Claude API to process the nutritional info.
    """
    # Remove "Bearer " prefix from authorization
    provided_api_key = authorization.replace("Bearer ", "")

    # Compare with prod_api_key
    if provided_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing credentials")

    example_nutritional_data = """
    [
        {
        "food_title": "Chocolate Cake",
        "servings": "1",
        "Calories": "340",
        "Protein": "6.2",
        "Carbohydrates": "45.3",
        "Total Fat": "16.8",
        "Sodium": "230",
        "Saturated Fat": "3.5",
        "Cholesterol": "55",
        "Sugar": "28.4",
        "Calcium": "85",
        "Iron": "2.1",
        "Potassium": "185",
        "Vitamin C": "0.2",
        "Vitamin E": "2.1",
        "Vitamin D": "0.3"
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
              - For each recipe in {user_prompt}, provide a JSON snippet strictly in the format: {example_nutritional_data}
              - Include all relevant data accurately calculated with detail.

            5. General Instructions
              - Do NOT ask any follow-up questions to the user.
              - Display the JSON response even if you have length constraints while providing JSON recipes.
              - Do not include any additional text or explanations
              - Return response in valid JSON format matching the example structure
              - Default to a **single serving** unless otherwise specified by the user in the {user_prompt}.
            """
    try:
        with claude_client.messages.stream(
                model="claude-3-7-sonnet-20250219",
                max_tokens=8192,
                temperature=0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
        ) as stream:
            texts = []
            for text in stream.text_stream:
                texts.append(text)
            final_adjusted_recipe = ''.join(texts)

        formatted_response = format_response(final_adjusted_recipe)
        return Response(content=formatted_response, media_type="application/json")

    except Exception as e:
        error_message = str(e)
        logging.error(f"Error in Claude streaming: {e}")

        if "overloaded_error" in error_message or "529" in error_message:
            logging.warning("Claude is overloaded. Falling back to OpenAI...")
        # If Claude API fails, fall back to OpenAI API
        try:
            openai_client = get_openai_client()
            # Use the same system and user prompts for OpenAI

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature = 0,
                stream=True
            )

            texts = []
            for chunk in response:
                # Extract the content from each streamed chunk
                delta = chunk.choices[0].delta
                texts.append(delta.get("content", ""))
            final_adjusted_recipe = ''.join(texts)
        except Exception as openai_exception:
            raise HTTPException(
                status_code=500,
                detail=f"Both Claude and OpenAI API calls failed. Claude error: {e}; OpenAI error: {openai_exception}"
            )

        formatted_response = format_response(final_adjusted_recipe)
        return Response(content=formatted_response, media_type="application/json")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Meal Logger Service is up and running."}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
