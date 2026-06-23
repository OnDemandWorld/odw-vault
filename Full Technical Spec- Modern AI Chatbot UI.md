# Full Technical Spec: Modern AI Chatbot UI

I'll give you two paths — **(A) Gradio CSS override** for the fastest win, and **(B) a full custom React/Next.js spec** for a true Genspark-tier result. Pick based on how far you want to go.

---

## 1. Design System (shared by both paths)

### Color Tokens

```css
:root {
  /* Light mode */
  --bg-primary: #FFFFFF;
  --bg-secondary: #F7F7F8;
  --bg-tertiary: #ECECEC;
  --bg-composer: #FFFFFF;
  --bg-user-msg: #F4F4F4;

  --text-primary: #0D0D0D;
  --text-secondary: #5D5D5D;
  --text-tertiary: #8E8E8E;

  --border-subtle: rgba(0, 0, 0, 0.08);
  --border-default: rgba(0, 0, 0, 0.12);

  --accent: #000000;            /* or your brand color */
  --accent-hover: #1A1A1A;
  --accent-fg: #FFFFFF;

  --shadow-composer: 0 0 0 1px rgba(0,0,0,0.05), 0 2px 12px rgba(0,0,0,0.06);
  --shadow-composer-focus: 0 0 0 1px rgba(0,0,0,0.12), 0 4px 24px rgba(0,0,0,0.08);

  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
  --radius-xl: 28px;
  --radius-full: 9999px;
}

[data-theme="dark"] {
  --bg-primary: #1A1A1A;
  --bg-secondary: #212121;
  --bg-tertiary: #2A2A2A;
  --bg-composer: #2A2A2A;
  --bg-user-msg: #2F2F2F;

  --text-primary: #ECECEC;
  --text-secondary: #A8A8A8;
  --text-tertiary: #707070;

  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-default: rgba(255, 255, 255, 0.14);

  --accent: #FFFFFF;
  --accent-hover: #ECECEC;
  --accent-fg: #0D0D0D;
}
```

### Typography

```css
body {
  font-family: "Inter", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 15px;
  line-height: 1.6;
  color: var(--text-primary);
  background: var(--bg-primary);
  -webkit-font-smoothing: antialiased;
}

/* Hero greeting */
.hero-title {
  font-size: clamp(28px, 4vw, 40px);
  font-weight: 500;
  letter-spacing: -0.02em;
  line-height: 1.15;
}
```

### Spacing & Sizing

```
Composer max-width:     768px (desktop), 100% - 32px (mobile)
Chat canvas max-width:  768px
Sidebar width:          260px (expanded), 56px (collapsed)
Composer min-height:    56px
Composer max-height:    240px (then scroll)
Message vertical gap:   24px
Composer bottom gap:    24px from viewport bottom
```

---

## 2. Layout Specification

### Three-Zone Structure

```
┌─────────┬──────────────────────────────────────┐
│         │  Top Bar (48px)                      │
│ Sidebar ├──────────────────────────────────────┤
│ (260px) │                                      │
│         │   Main Canvas                        │
│ - New   │   ┌──────────────────────────────┐   │
│   chat  │   │                              │   │
│ - Hist. │   │   Empty: centered hero       │   │
│         │   │   Active: scrollable msgs    │   │
│ - User  │   │                              │   │
│         │   └──────────────────────────────┘   │
│         │   Composer (docked or centered)      │
└─────────┴──────────────────────────────────────┘
```

### State Transitions

**Empty state** — no messages yet:
- Hero greeting centered vertically (~40% from top)
- Composer directly below hero, centered
- Prompt chips below composer
- No scroll

**Active state** — first message sent:
- Hero fades out (200ms opacity transition)
- Composer animates to bottom-docked position (300ms ease-out, transform-based)
- Message list fades in, scrolls to bottom
- Composer stays sticky at `bottom: 24px`

---

## 3. Component Specifications

### 3.1 Composer (the hero element)

**Structure:**

```
┌──────────────────────────────────────────────────┐
│  [📎] [Tools ▾]                                  │
│                                                   │
│   Ask anything...                                 │
│                                                   │
│                              [🎤]  [↑]           │
└──────────────────────────────────────────────────┘
```

**Specs:**

- Container: `border-radius: 28px`, `padding: 12px 16px`, `background: var(--bg-composer)`, `box-shadow: var(--shadow-composer)`.
- On focus: shadow upgrades to `--shadow-composer-focus`, no border color change (subtler than a blue ring).
- Textarea: `border: none`, `outline: none`, `resize: none`, `background: transparent`, `font-size: 16px` (prevents iOS zoom), `min-height: 24px`, auto-grow up to 240px.
- Left actions row: attach button, mode/tool selector pill. Icon size 20px, button size 32px, border-radius 8px, hover background `var(--bg-tertiary)`.
- Right actions: mic button (32px), send button (32px circular).
- Send button states: disabled (gray, no pointer) when empty; enabled (filled `--accent`, white icon) when text present; loading (spinner) during streaming; stop (square icon) while streaming for cancellation.
- Keyboard: `Enter` submits, `Shift+Enter` newline, `Esc` clears focus, `↑` recalls last message when composer empty.

### 3.2 Empty State Hero

```html
<div class="hero">
  <h1 class="hero-title">Good evening, Alex</h1>
  <p class="hero-subtitle">How can I help you today?</p>
</div>
```

- Vertical position: flexbox `justify-content: center` on a container that takes the canvas height minus composer height.
- Greeting logic: time-based ("Good morning/afternoon/evening") + user name if available.
- Subtitle: `font-size: 18px`, `color: var(--text-secondary)`, `margin-top: 8px`.

### 3.3 Prompt Starter Chips

```
┌─────────────────────┐  ┌─────────────────────┐
│ 💡 Brainstorm ideas │  │ 📝 Draft an email   │
│ for a weekend trip  │  │ to my team          │
└─────────────────────┘  └─────────────────────┘
```

- Grid: 2 columns desktop, 1 column mobile, gap 12px.
- Card: `padding: 12px 16px`, `border: 1px solid var(--border-subtle)`, `border-radius: 12px`, `background: transparent`, `cursor: pointer`.
- Hover: `background: var(--bg-secondary)`, slight `transform: translateY(-1px)`, transition 150ms.
- Content: small icon (16px) + 2-line text, `font-size: 14px`.
- Refresh button: small `↻` icon below grid, regenerates set from a pool of 12+ prompts.

### 3.4 Message List

**User message:**
- Right-aligned, `max-width: 70%`, `background: var(--bg-user-msg)`, `border-radius: 20px`, `padding: 10px 16px`.

**Assistant message:**
- Full width within canvas, **no bubble**, no background.
- Optional 24px avatar/icon on the left (but most modern UIs skip this).
- Markdown-rendered: headings, lists, code blocks, tables.
- Code blocks: `background: var(--bg-secondary)`, `border-radius: 12px`, monospace font, syntax highlighting (Shiki or Prism), copy button top-right.
- Hover toolbar appears below the message: copy, regenerate, thumbs up/down. Fades in on `:hover` of the message row.

**Streaming:**
- Tokens appended as they arrive.
- Blinking cursor `▍` at end during stream.
- Pre-stream: 3-dot pulsing indicator or subtle shimmer for ~300–800ms.

### 3.5 Sidebar

```
┌──────────────────┐
│  ☰   [+ New chat]│
├──────────────────┤
│  Today           │
│  · Trip planning │
│  · Code review   │
│  Yesterday       │
│  · Email draft   │
│  Previous 7 days │
│  · ...           │
├──────────────────┤
│  👤 Alex         │
└──────────────────┘
```

- Width 260px expanded, 56px collapsed (icon-only).
- New chat button: full-width, `border: 1px solid var(--border-default)`, `border-radius: 10px`, prominent placement at top.
- History items: `padding: 8px 12px`, `border-radius: 8px`, `font-size: 14px`, truncate with ellipsis. Active item: `background: var(--bg-tertiary)`.
- Hover reveals a `⋯` menu (rename, delete, share).
- Grouped by date: Today / Yesterday / Previous 7 days / Previous 30 days / Older.
- User block at bottom: avatar + name, click opens settings menu.
- Mobile: sidebar becomes an off-canvas drawer triggered by hamburger.

### 3.6 Top Bar

- Height 48px, no background, no border-bottom (or 1px subtle).
- Left: model selector dropdown (e.g., "Claude Opus 4.7 ▾") OR current chat title.
- Right: share button, settings/profile menu.
- Sticky, `backdrop-filter: blur(12px)` if content scrolls under.

---

## 4. Path A: Gradio CSS Override

Fastest route. Add this as `custom.css` and pass to `gr.Blocks(css=...)`.

```python
import gradio as gr

CUSTOM_CSS = """
/* Reset Gradio's container */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: var(--bg-primary);
    font-family: "Inter", -apple-system, sans-serif;
}

/* Hide default Gradio footer/branding */
footer { display: none !important; }

/* Main chat container */
#chat-column {
    max-width: 768px;
    margin: 0 auto;
    padding: 24px 16px 120px 16px;
}

/* Chatbot area */
.chatbot, #chatbot {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Message styling */
.message-wrap {
    border: none !important;
    box-shadow: none !important;
}

.message.user {
    background: var(--bg-user-msg) !important;
    border-radius: 20px !important;
    padding: 10px 16px !important;
    max-width: 70% !important;
    margin-left: auto !important;
    border: none !important;
}

.message.bot {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    max-width: 100% !important;
}

/* Composer overhaul */
.input-container, #composer {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    width: calc(100% - 32px);
    max-width: 768px;
    background: var(--bg-composer);
    border-radius: 28px !important;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.05), 0 2px 12px rgba(0,0,0,0.06);
    padding: 12px 16px !important;
    border: none !important;
    transition: box-shadow 200ms ease;
    z-index: 10;
}

.input-container:focus-within {
    box-shadow: 0 0 0 1px rgba(0,0,0,0.12), 0 4px 24px rgba(0,0,0,0.08);
}

.input-container textarea {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    outline: none !important;
    font-size: 16px !important;
    resize: none !important;
    padding: 8px 0 !important;
}

/* Send button */
.input-container button[type="submit"], #send-btn {
    width: 32px !important;
    height: 32px !important;
    border-radius: 50% !important;
    background: var(--accent) !important;
    color: var(--accent-fg) !important;
    border: none !important;
    min-width: 32px !important;
}

/* Hero (empty state) */
#hero {
    text-align: center;
    padding: 20vh 16px 0;
}
#hero h1 {
    font-size: clamp(28px, 4vw, 40px);
    font-weight: 500;
    letter-spacing: -0.02em;
}

/* Prompt chips */
#starter-chips {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    max-width: 768px;
    margin: 24px auto 0;
}
@media (max-width: 640px) {
    #starter-chips { grid-template-columns: 1fr; }
}
.chip {
    padding: 12px 16px;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    cursor: pointer;
    font-size: 14px;
    transition: all 150ms;
    background: transparent;
    text-align: left;
}
.chip:hover {
    background: var(--bg-secondary);
    transform: translateY(-1px);
}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="chat-column"):
        hero = gr.HTML("""
            <div id="hero">
                <h1>How can I help you today?</h1>
            </div>
        """)
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            show_label=False,
            type="messages",
            avatar_images=None,
            bubble_full_width=False,
            height=600,
        )
        with gr.Row(elem_id="starter-chips"):
            chip1 = gr.Button("💡 Brainstorm weekend trip ideas", elem_classes="chip")
            chip2 = gr.Button("📝 Draft a follow-up email", elem_classes="chip")
            chip3 = gr.Button("🐍 Explain this Python code", elem_classes="chip")
            chip4 = gr.Button("📊 Analyze a dataset", elem_classes="chip")
        msg = gr.Textbox(
            elem_id="composer",
            placeholder="Ask anything...",
            show_label=False,
            container=False,
            lines=1,
            max_lines=10,
        )
```

**Limitations of the Gradio path:** you can get the visual ~80% of the way there, but the centered-to-docked composer animation, true sidebar history with grouping, and streaming markdown polish are awkward to implement. For a real Genspark-tier result, go to Path B.

---

## 5. Path B: Custom Next.js + Vercel AI SDK

This is the recommended stack and matches what most modern AI products use.

### Tech Stack

```
Framework:    Next.js 15 (App Router)
Styling:      Tailwind CSS v4 + CSS variables
Components:   shadcn/ui (Radix primitives)
Streaming:    Vercel AI SDK (ai + @ai-sdk/react)
Markdown:     react-markdown + remark-gfm + rehype-highlight (or Shiki)
Icons:        lucide-react
Animations:   framer-motion (composer transition, message fade-in)
State:        Zustand or React Context for chat list
Storage:      localStorage (MVP) → Postgres + Drizzle (production)
```

### File Structure

```
app/
├── layout.tsx
├── page.tsx                    # main chat page
├── api/
│   └── chat/route.ts           # streaming endpoint
components/
├── chat/
│   ├── Composer.tsx
│   ├── MessageList.tsx
│   ├── Message.tsx
│   ├── EmptyState.tsx
│   ├── PromptChips.tsx
│   └── StreamingIndicator.tsx
├── sidebar/
│   ├── Sidebar.tsx
│   ├── ChatHistoryItem.tsx
│   └── NewChatButton.tsx
├── topbar/
│   └── TopBar.tsx
└── ui/                         # shadcn components
lib/
├── store.ts                    # Zustand store
├── prompts.ts                  # starter prompt pool
└── utils.ts
```

### API Route (streaming)

```ts
// app/api/chat/route.ts
import { streamText } from "ai";
import { anthropic } from "@ai-sdk/anthropic";

export const runtime = "edge";

export async function POST(req: Request) {
  const { messages } = await req.json();
  const result = streamText({
    model: anthropic("claude-opus-4-7"),
    messages,
    system: "You are a helpful assistant.",
  });
  return result.toDataStreamResponse();
}
```

### Main Page Component

```tsx
// app/page.tsx
"use client";
import { useChat } from "@ai-sdk/react";
import { AnimatePresence, motion } from "framer-motion";
import EmptyState from "@/components/chat/EmptyState";
import PromptChips from "@/components/chat/PromptChips";
import MessageList from "@/components/chat/MessageList";
import Composer from "@/components/chat/Composer";
import Sidebar from "@/components/sidebar/Sidebar";
import TopBar from "@/components/topbar/TopBar";

export default function ChatPage() {
  const chat = useChat({ api: "/api/chat" });
  const isEmpty = chat.messages.length === 0;

  return (
    <div className="flex h-dvh bg-[var(--bg-primary)]">
      <Sidebar />
      <main className="flex-1 flex flex-col relative">
        <TopBar />
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 pb-40 pt-6 min-h-full flex flex-col">
            <AnimatePresence mode="wait">
              {isEmpty ? (
                <motion.div
                  key="empty"
                  className="flex-1 flex flex-col justify-center"
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.2 }}
                >
                  <EmptyState />
                  <div className="mt-8">
                    <Composer chat={chat} centered />
                  </div>
                  <PromptChips onSelect={(p) => chat.append({ role: "user", content: p })} />
                </motion.div>
              ) : (
                <motion.div key="active" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <MessageList messages={chat.messages} status={chat.status} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
        {!isEmpty && (
          <div className="absolute bottom-6 left-0 right-0 px-4">
            <div className="max-w-3xl mx-auto">
              <Composer chat={chat} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
```

### Composer Component

```tsx
// components/chat/Composer.tsx
"use client";
import { useRef, useEffect } from "react";
import { Paperclip, Mic, ArrowUp, Square } from "lucide-react";
import { cn } from "@/lib/utils";

export default function Composer({ chat, centered = false }) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isStreaming = chat.status === "streaming";
  const canSend = chat.input.trim().length > 0 && !isStreaming;

  // Auto-grow
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`;
  }, [chat.input]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) chat.handleSubmit();
    }
  };

  return (
    <form
      onSubmit={chat.handleSubmit}
      className={cn(
        "rounded-[28px] bg-[var(--bg-composer)] px-4 py-3",
        "shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_12px_rgba(0,0,0,0.06)]",
        "focus-within:shadow-[0_0_0_1px_rgba(0,0,0,0.12),0_4px_24px_rgba(0,0,0,0.08)]",
        "transition-shadow duration-200"
      )}
    >
      <textarea
        ref={textareaRef}
        value={chat.input}
        onChange={chat.handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything..."
        rows={1}
        className="w-full resize-none bg-transparent outline-none text-[16px] leading-6 placeholder:text-[var(--text-tertiary)] max-h-[240px]"
      />
      <div className="flex items-center justify-between mt-2">
        <div className="flex items-center gap-1">
          <button type="button" className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)]">
            <Paperclip size={18} />
          </button>
          <button type="button" className="px-3 py-1.5 rounded-full text-sm hover:bg-[var(--bg-tertiary)]">
            Tools
          </button>
        </div>
        <div className="flex items-center gap-1">
          <button type="button" className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)]">
            <Mic size={18} />
          </button>
          <button
            type={isStreaming ? "button" : "submit"}
            onClick={isStreaming ? () => chat.stop() : undefined}
            disabled={!isStreaming && !canSend}
            className={cn(
              "w-8 h-8 rounded-full flex items-center justify-center transition-colors",
              (canSend || isStreaming)
                ? "bg-[var(--accent)] text-[var(--accent-fg)]"
                : "bg-[var(--bg-tertiary)] text-[var(--text-tertiary)]"
            )}
          >
            {isStreaming ? <Square size={14} fill="currentColor" /> : <ArrowUp size={18} />}
          </button>
        </div>
      </div>
    </form>
  );
}
```

### Message Rendering

```tsx
// components/chat/Message.tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Copy, RefreshCw, ThumbsUp, ThumbsDown } from "lucide-react";

export default function Message({ message, isStreaming }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end my-6">
        <div className="max-w-[70%] bg-[var(--bg-user-msg)] rounded-[20px] px-4 py-2.5">
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="group my-6">
      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
          {message.content}
        </ReactMarkdown>
        {isStreaming && <span className="inline-block w-2 h-4 bg-current animate-pulse ml-0.5" />}
      </div>
      {!isStreaming && (
        <div className="flex gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <button className="p-1.5 rounded hover:bg-[var(--bg-secondary)]"><Copy size={14} /></button>
          <button className="p-1.5 rounded hover:bg-[var(--bg-secondary)]"><RefreshCw size={14} /></button>
          <button className="p-1.5 rounded hover:bg-[var(--bg-secondary)]"><ThumbsUp size={14} /></button>
          <button className="p-1.5 rounded hover:bg-[var(--bg-secondary)]"><ThumbsDown size={14} /></button>
        </div>
      )}
    </div>
  );
}
```

---

## 6. Animation Specs

| Element | Trigger | Animation | Duration | Easing |
|---|---|---|---|---|
| Empty → Active | First message sent | Hero opacity 1→0, translateY 0→-20px | 200ms | ease-out |
| Composer dock | First message sent | Position center → bottom (use `layoutId` in framer-motion) | 300ms | ease-out |
| Message in | New message appended | Opacity 0→1, translateY 8px→0 | 200ms | ease-out |
| Streaming cursor | While streaming | Opacity pulse 1→0.3→1 | 800ms loop | linear |
| Send button enable | Text entered | Background color transition | 150ms | ease |
| Composer focus | Focus | Box-shadow expand | 200ms | ease |
| Sidebar toggle | Click hamburger | Width 260 ↔ 56 | 200ms | ease-out |

For the centered-to-docked composer, the cleanest implementation is framer-motion's shared `layoutId`:

```tsx
<motion.div layoutId="composer" transition={{ type: "spring", stiffness: 300, damping: 30 }}>
  <Composer chat={chat} />
</motion.div>
```

Render it inside the empty-state container when `isEmpty`, and inside the bottom-fixed container when not. Framer handles the position interpolation automatically.

---

## 7. Accessibility Checklist

Keyboard navigation throughout (Tab order: sidebar → top bar → composer → message actions). All icon-only buttons need `aria-label`. The composer textarea needs an associated label (visually hidden). Focus rings must be visible — don't remove `:focus-visible` outlines, just style them: `outline: 2px solid var(--accent); outline-offset: 2px`. Live region (`aria-live="polite"`) on the message list so screen readers announce streaming responses. Color contrast minimum 4.5:1 for body text, 3:1 for large text. Respect `prefers-reduced-motion` — disable the composer dock animation and message fade-ins. Minimum tap target 44×44px on mobile.

---

## 8. Responsive Breakpoints

```
Mobile:    < 640px   — sidebar becomes drawer, composer full-width minus 16px padding, single-column chips
Tablet:    640–1024  — sidebar collapsible, composer max 720px
Desktop:   > 1024px  — sidebar expanded by default, composer max 768px
```

---

## 9. Performance Notes

Use React Server Components for the shell, client components only for `Composer`, `MessageList`, and `Sidebar`. Virtualize the message list (`@tanstack/react-virtual`) once it exceeds ~50 messages. Lazy-load `rehype-highlight` languages (it ships huge by default — use Shiki with a curated language set instead). Debounce localStorage writes for chat history. Use `content-visibility: auto` on off-screen messages.

---

## 10. Suggested Build Order

Start with the shell: Next.js + Tailwind + the color tokens and typography. Then build the static empty state with hero and chips. Then the composer (static, no streaming). Then wire up `useChat` and the streaming API. Then message rendering with markdown. Then the centered-to-docked animation. Then sidebar with localStorage. Then polish: hover toolbars, dark mode toggle, keyboard shortcuts, and finally accessibility pass.

---

Want me to go deeper on any specific piece — the streaming markdown rendering with proper code-block syntax highlighting, the Zustand store for chat history, the framer-motion shared-layout animation, or a complete starter repo structure with `package.json` and config files?