# UI Debugging & Refactoring — Lessons Learned

## What Was Fixed

### 1. Send button greyed out, non-functional (multiple rounds)
**Root cause**: Gradio 6.x uses `<gradio-app>` web components that render over our custom page. Previous attempts to inject custom HTML via `gr.HTML()` were sanitized — Gradio strips `<script>` tags, `id` attributes, and inline event handlers.

**Solution**: Two-port architecture. Gradio runs on an internal port (8889) for API only. A custom FastAPI proxy serves our full HTML/CSS/JS on the user-facing port (8888), proxying `/gradio_api/*` to Gradio. This guarantees zero HTML sanitization since we control the root page completely.

### 2. Responses not streaming back
**Root cause**: The JS was calling the wrong Gradio API endpoint pattern. Gradio 6 requires a two-step flow:
1. `POST /gradio_api/call/chat` → returns `{"event_id": "..."}`
2. `GET /gradio_api/call/chat/{event_id}` → SSE stream with response data

Additionally, the response format changed — `content` is now an array of `{text, type}` objects, not a plain string.

**Solution**: Updated JS to use the two-step Gradio 6 flow and extract text from `last.content[0].text`. Updated the proxy to stream SSE responses using `httpx.AsyncClient.stream()`.

### 3. Send button broken again (f-string escaping bugs)
**Root cause**: The `_build_full_page()` function used a Python f-string (`return f"""..."""`) to build the HTML template. This caused three silent bugs:
- `{id}` in a JS comment (`// then GET /call/chat/{id} for SSE`) was interpolated as Python's `id()` builtin function, rendering as `then GET /call/chat/<built-in function id> for SSE`
- `\\n` for JavaScript's `\n` became a literal newline in the JS output, creating `buf.split('` followed by an actual newline — a JS syntax error
- Regex patterns like `\s`, `\S`, `\*` in the JS `_md()` function were Python escape sequences, not JS regex escapes, causing `SyntaxWarning` and wrong output

**Solution**: Rewrote `_build_full_page()` to use a regular triple-quoted string (no `f` prefix) with `$PLACEHOLDER` tokens and `.replace()` calls at the end:
```python
def _build_full_page(...):
    return """...$GREETING...$CHIPS_JSON...""".replace("$GREETING", greeting) ...
```
This eliminates all Python f-string escape interference with JavaScript code.

### 4. Inline event handlers not firing
**Root cause**: `document.addEventListener('input', ...)` with `e.target.closest()` worked in theory but was unreliable on mobile Safari. The `input` event sometimes doesn't fire predictably, especially with `autofocus` and fast typing.

**Solution**: Switched to inline event handlers directly on the HTML elements:
- `<textarea oninput="_handleInput(this)" onkeydown="_handleKeydown(event)">`
- `<div id="app" onclick="_handleClick(event)">`
- Handlers exposed on `window` so inline attributes can call them

### 5. Dropdown validation error — no response at all
**Root cause**: The Gradio `gr.Dropdown` was created without `choices`, so when the JS sent `"All folders"` as the folder filter value, Gradio rejected it with: `Value: All folders is not in the list of choices: []`. The SSE stream returned an error event, but the JS showed no error indicator — just an empty response.

**Solution**: Added `choices=folder_choices, value="All folders"` to the Dropdown.

### 6. Empty response for 40+ seconds before text appears
**Root cause**: The model `gemma4:latest` generates ~860 empty thinking tokens before producing actual text. The UI was yielding `history[-1]["content"] = full_text` where `full_text` was `""` for every thinking token, so the assistant message appeared as blank for ~40 seconds.

**Solution**: Keep "Thinking..." placeholder visible until actual text arrives:
```python
if full_text:
    history[-1]["content"] = full_text
```

### 7. Cursor blinking showing as "unicode box with cross and 8C"
**Root cause**: The CSS `content:"\258C"` (Unicode BLOCK CHARACTER) is not rendered correctly on all platforms/fonts, especially on iOS Safari.

**Solution**: Changed to a simple `|` pipe character with light gray color and font-weight styling.

### 8. Big gap between header and chat content after response
**Root cause**: The `#hero.hidden` CSS used `opacity:0; height:0; overflow:hidden` which still reserved space in some browsers due to flex layout behavior.

**Solution**: Changed to `display:none!important` which completely removes the element from the layout flow.

## What Made This Hard to Debug

### Gradio's HTML sanitization was silent
When you pass HTML to `gr.HTML()`, Gradio strips `<script>` tags and `id` attributes without any warning or error. The page renders, looks correct in DevTools, but the JS never executes. There's no console error, no 404, no network failure — it just silently doesn't work.

### Gradio 6 API changed without obvious version markers
The `/run/chat` endpoint (single JSON response) vs `/call/chat` + SSE (streaming) have similar URLs but completely different behavior. If you call the wrong one, you get an immediate JSON response with no error — just no streaming. The browser shows no error because the request succeeds.

### F-string interpolation of JavaScript is invisible
When Python's `{id}` gets interpolated in an f-string, the served page shows `<built-in function id>`. But you can only see this by curling the served page — the source file looks correct. The JS syntax error from `\\n` → literal newline is even worse because the file has correct-looking escape sequences, but Python renders them at string construction time.

### Gradio Dropdown validation fails with SSE error events
When the Dropdown has no choices and you pass an invalid value, Gradio returns a `{"error": "...", "visible": true}` SSE event. Most frontend code doesn't check for error events in SSE streams — they assume data events always contain valid content. The error was being silently swallowed by `catch(e) {}` in the JS.

### Thinking tokens produce empty content silently
Ollama with `gemma4:latest` returns 860 chunks with `content: ""` before actual text. Each chunk is a valid SSE event. The response stream is working. The connection is fine. The model is processing. But the UI shows nothing for 40 seconds because `full_text` is empty. No timeout, no loading indicator, no error — just silence.

### The Gradio web component overlays custom HTML
When Gradio registers its root route, it serves the `<gradio-app>` web component regardless of what routes you've previously registered. It doesn't throw an error, it doesn't log a warning — it just serves its own page. You can verify this by looking at the page source vs what's rendered — they're completely different.

### Browser caching masks fixes
The served HTML page has no cache-busting headers. Every fix required a hard refresh (Cmd+Shift+R) to see the new version. Without it, you're testing against stale JS, making you think your fix didn't work when it did.

## Debugging Tools That Helped
- **`curl` the served page**: The only way to see what the browser actually receives. Source code and served content can differ silently.
- **`curl` the Gradio API directly**: Bypasses the JS entirely to test if the backend is working.
- **Python `ast.parse()`**: Catches escape sequence issues that `python -c` misses.
- **Direct Ollama call in isolation**: Proved the model works, isolated the problem to the UI layer.
- **`grep` for patterns in served HTML**: `curl | grep 'buf.split'` showed the literal newline issue instantly.

## Current State
- Gradio API on port 8889 (internal, not exposed)
- Custom FastAPI proxy on port 8888 (exposed to users)
- All JS uses inline event handlers for reliability
- F-string replaced with `.replace()` template for correctness
- "Thinking..." placeholder visible during model processing
- Console logging enabled for browser-side debugging
- Server-side logging for query tracking
