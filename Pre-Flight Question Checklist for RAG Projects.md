# Pre-Flight Question Checklist for RAG Projects

Use this before designing any RAG system. The answers determine architecture, model choices, scope, and scheduling. If you can't answer a question, that's itself useful information — it tells you what to investigate first.

## 1. Purpose and users

What problem does this RAG solve that a search engine over the same corpus wouldn't? If the answer is "search would be fine," strongly consider building search first.

Who are the users — by role, technical level, and number? Five engineers asking deep technical questions is a very different system than fifty mixed-role staff asking high-level questions.

What are five real questions a user would ask in week one? Write them down before building anything. Test your finished system against them.

What's the cost of a wrong answer? Mildly annoying, embarrassing, or business-damaging? This sets your tolerance for hallucination and how aggressively you need citations, confidence indicators, and fallback-to-search behavior.

What does "good enough to ship" look like? Define this concretely: percentage of test questions answered correctly, latency, coverage. Without this, the project never ends.

## 2. Corpus scope and content

Where does the source material live — single folder, multiple folders, NAS, cloud drive, mixed?

How big is it, in GB and in file count? Order of magnitude is enough.

What file types are present, roughly? PDFs, Office docs, code, media, CAD, email exports, chat exports, databases, proprietary formats?

What languages? Single, bilingual, multilingual? Mixed within documents or segregated by file?

How is the corpus organized — by project, by client, by date, by type, by author, ad-hoc? Does the folder hierarchy carry semantic meaning that should be preserved as metadata?

How clean is it? One person's curated folders or twenty years of accumulated mess?

What proportion is duplicates and near-duplicates? Guess based on your knowledge of how the corpus accumulated.

How much of it is actually useful? Often only 10–30% of a corpus is worth indexing; the rest is noise that hurts retrieval.

## 3. Corpus dynamics

Is the corpus static or actively edited? If edited, how frequently and by whom?

If new files arrive, how should the system handle them — daily batch, real-time watch, manual trigger?

Are there files that change in place (live documents) versus append-only? Live documents need re-indexing logic; append-only is simpler.

Is there a retention policy? Should old documents be removed from the index after some time?

How do you handle deletions in the source — should the index reflect them?

## 4. Confidentiality, access, and compliance

What's the sensitivity classification of the corpus? Public, internal, confidential, regulated?

Are there client NDAs that restrict use of specific documents even for internal tooling?

Does the system need to enforce per-user access control, or is everyone with system access entitled to see everything?

If access control is needed, how is it modeled — by folder, by file, by client, by tag, by user role?

Are there jurisdictions involved that constrain where data and inference can run? GDPR, HIPAA, China cross-border data, sector-specific rules?

Must the system be fully air-gapped, or is outbound network for model API calls acceptable? This determines whether everything (models, signatures, language packs) must be pre-downloaded.

Is logging of queries acceptable, and if so for how long? Query logs are invaluable for tuning but are themselves sensitive data.

Who audits this system, and what evidence will they need?

## 5. Infrastructure and resources

What hardware will host the system — laptop, workstation, single server, multiple machines?

Apple Silicon, x86, GPU available?

Available RAM? This bounds which LLMs and embedding models you can run.

Is there a budget for hardware upgrades, or must it run on what's available?

Is there an existing vector store, search system, or database the RAG should plug into?

Who maintains this once it's built? Same person who built it, an ops team, or nobody?

## 6. Quality and evaluation

How will you evaluate retrieval quality? Eyeball test, structured eval set, user feedback?

Do you have or can you create a labeled set of (question, expected source documents) pairs? Even 30 of these is enormously valuable.

What's the acceptable latency for a query? Sub-second, a few seconds, can-wait-a-minute-for-a-deep-search?

Do answers need citations? At what granularity — file, page, paragraph?

How will you detect drift over time as the corpus grows?

What's your plan for handling questions the system answers wrong — log them, fix retrieval, fix data, fix prompts?

## 7. Scope discipline

What's explicitly *out of scope* for v1? Write this down.

What would tempt you to add in scope later, and what's the criterion for accepting it?

Is this a permanent system or an experiment? Build accordingly.

If this fails, what's the fallback for users? "Just use search," "ask the team," "open a ticket"?

## 8. Integration and surface

How will users interact with the system — chat UI, Slack bot, API, embedded in another tool, command line?

Do answers need to be exportable, citable in reports, copy-paste-friendly?

Does it need to integrate with existing systems — ticketing, CRM, project management, identity provider?

Does it need a mobile interface, or is desktop fine?

## 9. Specific to mixed-content corpora (engineering, legal, medical, etc.)

Are there file formats whose *content* matters but where text extraction is poor or impossible? CAD geometry, raw sensor data, binary blobs? How should those be represented — by metadata only, by associated documentation, by skipping?

Are there file formats where the *filename* carries critical information (`v3_final_revised_signed.pdf`)? Make sure your indexing captures filename as a first-class field, not just content.

Are there structured artifacts (spreadsheets, databases, configs) where retrieval-over-chunks is the wrong abstraction and a structured query would be better?

Are there time-sensitive documents (project status, deployment logs) where the *latest* version is what matters, not all versions?

## 10. Organizational

Who is the decision-maker on scope and trade-offs?

Who is the subject-matter expert on the corpus content? You'll need them during pre-flight review and again during evaluation.

What's the timeline, and is it realistic given the corpus size and complexity?

What happens to this project if the person building it leaves? Documentation, handover, code clarity matter accordingly.

How will success be communicated to the rest of the organization? A demo, a report, training, just emailing a URL?

---

## Using this checklist

Don't try to answer every question before starting — that's a recipe for analysis paralysis. The goal is to surface which questions you *can't* answer yet, so the pre-flight pipeline (and the pilot project I suggested earlier) becomes targeted at filling those gaps. Re-read the checklist after pre-flight is done; many questions will be easier to answer when you've actually looked at the data.

For your noveltebot situation specifically, the questions I'd push hardest on before writing any code are: corpus size and host machine, the access-control model across clients, and whether CAD/ROS/binary engineering content needs first-class handling or filename-only indexing. Those three answers shape the rest of the design.

Want me to write the Phase 0 + Phase 1 starter script next, or work up a robotics-services-tuned `format_policy.csv` seed file (PRONOM IDs and strategies for the formats you're most likely to encounter)?