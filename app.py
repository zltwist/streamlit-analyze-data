async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    body = b"""
    <html>
      <head><meta charset='utf-8'><title>Analyzer Dashboard</title></head>
      <body style='font-family:Arial,sans-serif;padding:40px;max-width:720px;margin:auto;'>
        <h1>Analyzer Dashboard</h1>
        <p>This repository contains a Streamlit dashboard. Vercel needs a Python entrypoint to build, so this file acts as the deployment entrypoint.</p>
        <p>Run the dashboard locally with <code>streamlit run streamlit_app.py</code>.</p>
      </body>
    </html>
    """

    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/html; charset=utf-8")]})
    await send({"type": "http.response.body", "body": body})