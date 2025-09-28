# azure-multimodal-ai-agents
azure-multimodal-ai-agents

End-to-end, multimodal, multi-agent reference project for Azure AI Foundry Agent Service using the Microsoft Agent Framework SDK. It demonstrates sequential orchestration across agents that handle text, audio, and images/video frames, then create an actionable ticket (GitHub Issue or Azure DevOps), and log an audit trail.

Key Capabilities

Sequential multi-agent pipeline: Ingest ➜ Summarize ➜ Classify ➜ Recommend ➜ Act ➜ Audit

Multimodal ingestion:

Text feedback (chat/email form)

Audio (transcription via Azure AI Speech)

Images / screenshots (captioning & OCR via Azure AI Vision & Document Intelligence)

Action tools: create GitHub Issue (default) or Azure DevOps Work Item

Audit logger: JSONL log with hashes + timestamps to support compliance (inspired by the user's Azure Audit Logger idea)

Config-first via .env and appsettings.json

Tested locally with simple fixtures; CI via GitHub Actions (lint + tests)
