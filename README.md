````markdown
# 🍽️ Meal Logger Service

**AI-powered nutritional analysis for recipes and meal descriptions.  
Returns accurate, per-serving breakdowns in JSON for easy UI or analytics use.  
Primary LLM: Claude (Anthropic) — fallback: OpenAI GPT-4o.**

---

## 🌎 Environments

- **Production:** `https://ai-foodhak.com`
- **Staging:**    `https://staging.ai-foodhak.com`

---

## 🧩 What does it do?

- Accepts any meal or recipe query.
- Returns detailed nutritional breakdown (macros, vitamins, minerals) for each recipe, per serving.
- Handles *multiple recipes in a single query*.
- Consistent, **clean JSON** output.  
- **No extra commentary, only structured data.**

---

## 🚦 Endpoints

| Method | Endpoint                  | Description                               |
|--------|--------------------------|-------------------------------------------|
| POST   | `/get_nutritional_info`  | Calculate nutrition for meals/recipes     |
| GET    | `/health`                | Service health check                      |

> **All POST endpoints require:**  
> `Authorization: Bearer <API_KEY>`

---

## 🛠️ Usage

### 1. Get Nutritional Info — POST `/get_nutritional_info`

#### Production Example

```bash
curl -X POST https://ai-foodhak.com/get_nutritional_info \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"query": "banana smoothie with almond milk, honey, and spinach. Also: grilled salmon with quinoa and asparagus, 2 servings."}'
````

#### Example JSON Response

```json
[
  {
    "food_title": "Banana Smoothie",
    "servings": "1",
    "Calories": "180",
    "Protein": "4.2",
    "Carbohydrates": "38.1",
    "Total Fat": "2.5",
    "Sodium": "34",
    "Saturated Fat": "0.2",
    "Cholesterol": "0",
    "Sugar": "23.8",
    "Calcium": "215",
    "Iron": "1.1",
    "Potassium": "422",
    "Vitamin C": "12.3",
    "Vitamin E": "1.2",
    "Vitamin D": "0"
  },
  {
    "food_title": "Grilled Salmon with Quinoa and Asparagus",
    "servings": "2",
    "Calories": "430",
    "Protein": "33.0",
    "Carbohydrates": "36.0",
    "Total Fat": "18.0",
    "Sodium": "320",
    "Saturated Fat": "3.1",
    "Cholesterol": "58",
    "Sugar": "3.2",
    "Calcium": "66",
    "Iron": "2.6",
    "Potassium": "590",
    "Vitamin C": "8.1",
    "Vitamin E": "2.7",
    "Vitamin D": "9.3"
  }
]
```

#### Example Error Response

```json
{
  "detail": "Invalid or missing credentials"
}
```

---

### 2. Health Check

```bash
curl https://ai-foodhak.com/health
```

Response:

```json
{
  "status": "healthy",
  "message": "Meal Logger Service is up and running."
}
```

---

## ⚡ Features

* **AI-powered** nutritional calculations, always up-to-date with science.
* Handles multiple recipes per request—returns a JSON array.
* Fallback logic: If Anthropic Claude is overloaded, automatically uses OpenAI GPT-4o.
* API-key protected for secure access.
* Consistent output: JSON objects for each recipe with all major nutrients.

---

## 📝 Notes

* Only **valid API keys** can access `/get_nutritional_info`.
* Output strictly follows a structured format for easy UI integration and post-processing.
* By default, assumes 1 serving per recipe unless user specifies otherwise.
* No user context/profile is required—just the meal description or recipe list.

---

