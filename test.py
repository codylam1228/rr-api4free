from openai import OpenAI

# Option 1: Use the local FreeAPI proxy (recommended)
# The proxy will automatically rotate through your OpenRouter keys
# To get a proxy key, run: uv run generate_proxy_key.py
FREEAPI_PROXY_KEY = "freeapi-W3_8S1dmnCy4AGy8ba5XbhRiGsDKgj4aMMDmQH7V09RwReXWa2v9dPBUzfqXj3WB"

# Option 2: Direct OpenRouter access (original method)
# OPENROUTER_API_KEY = "sk-or-v1-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Initialize client with FreeAPI proxy
client = OpenAI(
  base_url="http://localhost:8964/v1",  # Point to local proxy server
  api_key=FREEAPI_PROXY_KEY,
)

# Uncomment below to use direct OpenRouter access instead:
# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key=OPENROUTER_API_KEY,
# )

completion = client.chat.completions.create(
  # extra_headers={
  #   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
  #   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  # },
  extra_body={},
  model="deepseek/deepseek-r1:free",
  # Alternative free models if this doesn't work:
  # - "nvidia/nemotron-nano-9b-v2:free"
  # - "openai/gpt-oss-20b:free"
  # - Try: https://openrouter.ai/settings/privacy to check data policy
  messages=[
    {
      "role": "user",
      "content": "How to print hello world in PHP please"
    }
  ]
)
print(completion.choices[0].message.content)