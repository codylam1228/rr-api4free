# Free OpenRouter API Key Pool Manager

![Status: Active](https://img.shields.io/badge/status-active-brightgreen)

FastAPI service that hands out OpenRouter API keys in a round-robin cycle. Drop multiple free-tier keys into `openrouter/keys.ini`, start the server, and hit `GET /api/v1/keys/next` to rotate through them evenly.

---

## Project Structure

```
FreeAPI/
├── openrouter.example/ (.exmaple need to be removed )
│   └── keys.ini
├── server.py
├── test.py
├── README.md
├── pyproject.toml
```

> Only `openrouter/keys.ini` is required at runtime. `openrouter.example/` is a template you can copy or rename into place.

---

## Why Use This?

- Rotate through throwaway keys so rate limits and bans are spread evenly.
- Single self-contained `server.py`—no extra modules to hunt down.
- Production-ready Python client with caching, retries, circuit breaker, and metrics helpers.

---

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### Setup
1. Clone the repo.
2. Copy `openrouter.example` to `openrouter` (or rename the folder).
3. Edit `openrouter/keys.ini` and add one key per line. Lines starting with `#` are ignored.
4. Install dependencies:
   ```bash
   uv sync
   ```
5. Run the API:
   ```bash
   uv run server.py
   ```
6. Open http://localhost:8964/docs for Swagger UI.

> **Note**: Keys must follow the OpenRouter prefix pattern `sk-or-v1-...`.

---

## API Overview

| Method | Path                  | Description                       |
|--------|-----------------------|-----------------------------------|
| GET    | `/`                   | Basic service metadata            |
| GET    | `/api/v1/keys/next`   | Fetch next key in rotation        |
| GET    | `/api/v1/health`      | Lightweight health indicator      |
| GET    | `/api/v1/stats`       | Pool statistics and usage counts  |

### Response Samples

`GET /api/v1/keys/next`
```json
{
  "key": "sk-or-v1-...",
  "index": 2
}
```

`GET /api/v1/stats`
```json
{
  "pool_stats": {
    "total_keys": 5,
    "active_keys": 5,
    "total_usage": 123,
    "current_index": 3
  },
  "keys_info": [
    {
      "id": 1,
      "key": "sk-or-v1-...",
      "usage_count": 50
    }
  ]
}
```

---

## Built-in Python Client

Import `freeapi_client` or instantiate `FreeAPIClient` for advanced usage.

```python
from server import freeapi_client

api_key = freeapi_client.get_api_key()
if api_key:
    print(api_key[:16])
```

### Highlights
- Connection pooling via `requests.Session`
- 5-minute cache with usage caps (50 per key by default)
- Exponential backoff retries
- 100 ms rate limiting between requests
- Circuit breaker after repeated failures
- Health/stats helpers: `health_check()`, `get_stats()`
- Decorator `@freeapi_client.with_api_key` for auto-injection

More examples live in `server.py` docstring (`Usage Examples`).

---

## Configuration

Key settings live in the `Config` class inside `server.py`:

| Setting       | Default   | Description                          |
|---------------|-----------|--------------------------------------|
| `SERVICE_HOST`| `0.0.0.0` | Bind address for FastAPI             |
| `SERVICE_PORT`| `8964`    | Port number                          |
| `LOG_LEVEL`   | `INFO`    | Logging level                        |
| `KEYS_FILE`   | `./openrouter/keys.ini` | Key storage location |

Adjust by editing the class or overriding environment variables before launch.

---

## Development

Run lint/tests (if you add them):
```bash
uv run python -m pytest
```

Reload keys without restarting by calling the `/api/v1/stats` endpoint—statistics update live. For fresh keys, restart or call `key_pool.reload_keys()`.

### Test Script

`test.py` provides a simple smoke test for the endpoints.

```bash
uv run python test.py
```

---

## Troubleshooting

- **404 on `/api/v1/keys/next`**: No valid keys loaded; check `openrouter/keys.ini`.
- **403 from OpenRouter**: Key likely banned; remove it from the list.
- **Port in use**: Change `SERVICE_PORT` or stop the conflicting service.
- **Slow responses**: Reduce `max_usage_per_key` or add more keys.

---

## License

MIT License. See `LICENSE` if included in your distribution.


