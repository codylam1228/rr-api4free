from openai import OpenAI

OPENROUTER_API_KEY = "sk-or-v1-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

completion = client.chat.completions.create(
  # extra_headers={
  #   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
  #   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  # },
  extra_body={},
  model="x-ai/grok-4-fast:free",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "How to print hello world in python please"
        }
      ]
    }
  ]
)
print(completion.choices[0].message.content)