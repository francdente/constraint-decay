SERVER_CONFIG_TEMPLATE = """## Server Configuration

- The server MUST listen on **port {port}**.
- The server MUST expose a health-check endpoint at **GET /api/health-check** that returns HTTP 200.
- All API routes MUST be prefixed with **/api** (e.g., `/api/users`, `/api/articles`).
"""
