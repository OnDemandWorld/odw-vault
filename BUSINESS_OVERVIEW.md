# RAG Pre-Flight Pipeline — Business Overview

> Version: 1.0
> Date: 2026-05-04
> Audience: Business users, project managers, sales teams

---

## What We Built

A **document intelligence system** that reads your company's entire document library — thousands of files across dozens of folders — and turns it into a searchable knowledge base that can answer questions in natural language.

Think of it as building a "company brain" from your existing files. No manual tagging, no human re-reading, no data entry.

---

## The Problem It Solves

Companies accumulate massive amounts of documentation over years:
- Project plans, contracts, meeting minutes
- Training manuals and procedures
- Technical specifications and test reports
- Proposals, presentations, spreadsheets

When someone needs to find information — *"What's the maximum slope angle for the robot deployment?"* or *"What happened at Gleneagles Hospital in November?"* — they either ask a colleague who might know, or search through folders manually. This wastes time and loses institutional knowledge when people leave.

Our system reads every document once, understands what's in it, and lets anyone ask questions in plain language — with answers traced back to the actual source files.

---

## How It Works (Simplified)

### Step 1: Read — Ingest all documents
The system scans a folder (or hundreds of folders), identifies every file type, and extracts text from formats including:

| Format Type | Examples |
|---|---|
| PDFs (text and scanned) | Reports, contracts, manuals |
| Office documents | Word, Excel, PowerPoint |
| Images | Photos with embedded text/metadata |
| Video & audio | Recordings transcribed to text |
| CAD files | Technical drawings (filenames + metadata) |
| Archives | ZIP, RAR, 7z — automatically expanded |

**No data is lost.** Original files are never modified. Everything happens as a read-only scan.

### Step 2: Understand — Build context for every document piece
The system breaks large documents into manageable pieces and uses AI to:
- **Label folders** — understands what each folder is about (e.g., "project management," "training material")
- **Summarize documents** — creates a short summary of each file
- **Add context** — for every piece of text, writes a sentence explaining where it fits within its parent document

This last step is what makes the answers accurate. Instead of searching for isolated sentences, the system understands the full picture.

### Step 3: Index — Build a searchable knowledge base
All the processed content is stored in a vector database that can be searched semantically — meaning it finds answers even when the exact words don't match. Ask *"how steep can the ramp be?"* and it finds *"maximum slope ≤8°"* because it understands the meaning, not just the keywords.

### Step 4: Answer — Natural language Q&A
Users ask questions in plain English or Chinese. The system:
1. Searches its knowledge base for relevant content
2. Grounds its answer in the actual documents
3. Cites the source file so you can verify

---

## What the System Currently Knows

Our proof-of-concept processed a corpus of:

| Metric | Count |
|---|---|
| Unique documents | 384 |
| Duplicate files filtered out | 1,793 (removed automatically) |
| Total files scanned | 2,177 |
| Text pieces with AI context | 8,159 |
| Documents summarized | 45 |
| Videos transcribed | 36 |
| Audio files transcribed | 10 |
| Folders labelled | 71 |
| Languages | English (92.6%) + Chinese (7.4%) |

The knowledge spans: delivery robot deployment specs, hospital project details, UV disinfection science, training procedures, incident reports, and more.

---

## Extensibility — Adding & Removing Files

### Adding new documents
**Yes, it's designed for this.** Simply drop files into the source folder and re-run the pipeline. The system is **incremental** — it only processes what's new and skips what's already been done.

Typical re-run time for a small batch of files: **a few minutes**.

### Removing documents
Mark a file or folder as "excluded" via the CLI or a CSV list. The system removes it from search results without touching the original file. This is useful for:
- Removing sensitive documents from the knowledge base
- Excluding irrelevant folders (e.g., software installers, log files)

### Updating documents
Replace a file in the source folder. The system detects the change (via file hash comparison) and re-processes only that file on the next run.

**No manual database edits needed.** The folder is the source of truth — whatever's in the folder is what the system knows.

---

## How It Scales to Other Clients

The solution is **inherently multi-tenant by design** because:
1. Each client gets their own source folder + database
2. The system never hard-codes any specific document content
3. Configuration (models, thresholds, formats) lives in a single editable file

### What changes per client

| Item | What it means |
|---|---|
| **Source folder** | Point to the client's document directory |
| **Database** | Each client gets their own `corpus.db` |
| **Vector store** | Each client gets their own search index |
| **Config file** | Adjust AI models, language preferences, exclusions |

### What stays the same

Everything else — the pipeline code, the AI models, the search interface — is reusable as-is.

### Scaling considerations

| Scale Factor | Current (POC) | Production Estimate |
|---|---|---|
| Document count | 384 docs | 10,000-100,000+ |
| Processing time | ~25 hours (local AI) | 2-6 hours (API-based AI) |
| Infrastructure | Single laptop | Cloud server or on-premise |
| AI model cost | $0 (local GPU) | ~$5-50 per 10K documents |

For clients with large document libraries, we can offer two tiers:
- **Local deployment** — runs on their own hardware, zero ongoing AI cost, slower processing
- **Cloud deployment** — uses hosted AI APIs, faster processing, small per-document cost

---

## Technical Architecture (Simplified)

```
Source Folder
    │
    ▼
┌──────────────────────────────┐
│  Phase 0: Archive Expansion  │  Unzip/expand archives automatically
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 1: File Discovery     │  Walk folders, hash files, build inventory
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 2: Format ID          │  Identify every file type (PDF, DOCX, XLSX...)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 3: Content Triage     │  Detect OCR needs, media duration, language
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 4: Deduplication      │  Remove exact duplicates automatically
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 5: Folder Intelligence│  AI labels every folder by topic
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 6: Report Generation  │  Summary report of the entire corpus
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 7: Review & Approval  │  Exclude sensitive files, sign off
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 8: Text Extraction    │  Pull text from PDFs, Office docs, media
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 9: Document Summary   │  AI creates a short summary of each file
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 10: Chunking          │  Break long docs into searchable pieces
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 10.5: Context Build   │  AI writes context for every piece
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 11: Embedding         │  Convert text to searchable vectors
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Phase 12+: Q&A Interface    │  Users ask questions, get grounded answers
└──────────────────────────────┘
```

Each phase is independent and can be re-run without affecting the others.

---

## Key Advantages Over Competing Solutions

| Feature | Our Solution | Typical Alternatives |
|---|---|---|
| **File format support** | 15+ types including PDF, Office, CAD, media, archives | Usually PDF + text only |
| **Deduplication** | Automatic SHA-256 hash-based dedup | Manual or none |
| **Context augmentation** | AI adds document-level context to every search piece | Raw text search only |
| **Bilingual** | English + Chinese (Simplified & Traditional) | Usually English only |
| **Privacy** | Runs fully offline, no data leaves your network | Requires cloud upload |
| **Audit trail** | Every AI decision is logged with timestamps | Black box |
| **Source citation** | Every answer links back to the original file | Answers without proof |
| **Incremental updates** | Re-run only processes new/changed files | Full re-index every time |

---

## What's Next

- **Q&A interface** — A simple web page where users type questions and get answers with source links
- **Multi-client management** — A dashboard to manage multiple client deployments
- **Per-phase endpoint routing** — Allow different AI models/servers for different processing stages
- **Automated scheduling** — Periodic re-indexing (e.g., weekly) to keep the knowledge base fresh
- **Access controls** — Role-based permissions so different users see different document subsets

---

## Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation — finding relevant documents and using them to generate accurate answers |
| **Corpus** | The collection of documents the system knows about |
| **Chunk** | A small piece of a document (typically 1-2 paragraphs) that the system can search independently |
| **Embedding** | A numerical representation of text that allows semantic (meaning-based) search |
| **Vector DB** | A database optimized for storing and searching embeddings |
| **Deduplication** | Identifying and removing exact duplicate files |
| **POC** | Proof of Concept — the current demonstration version |
