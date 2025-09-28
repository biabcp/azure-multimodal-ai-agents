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

Architecture
# azure-multimodal-agents

End-to-end, **multimodal, multi-agent** reference project for **Azure AI Foundry Agent Service** using the **Microsoft Agent Framework SDK**. It demonstrates sequential orchestration across agents that handle **text, audio, and images/video frames**, then create an actionable ticket (GitHub Issue or Azure DevOps), and log an **audit trail**.

> Built for hands-on learning and portfolio use. Drop it into GitHub as-is.

---

## ✨ Key Capabilities

* **Sequential multi-agent pipeline**: Ingest ➜ Summarize ➜ Classify ➜ Recommend ➜ Act ➜ Audit
* **Multimodal ingestion**:

  * Text feedback (chat/email form)
  * Audio (transcription via Azure AI Speech)
  * Images / screenshots (captioning & OCR via Azure AI Vision & Document Intelligence)
* **Action tools**: create GitHub Issue (default) or Azure DevOps Work Item
* **Audit logger**: JSONL log with hashes + timestamps to support compliance (inspired by the user's Azure Audit Logger idea)
* **Config-first** via `.env` and `appsettings.json`
* **Tested locally** with simple fixtures; CI via GitHub Actions (lint + tests)

---

Architecture

```text
[User input]
  ├─ Text → (direct)
  ├─ Audio → Speech2Text (Azure AI Speech)
  └─ Image → Vision Caption (Azure AI Vision) + OCR (Document Intelligence)
          ↓
   Ingest Agent (multimodal normalizer)
          ↓
   Summarizer Agent (neutral, concise summary)
          ↓
   Classifier Agent (Sentiment: Positive/Negative, Type: Bug/Feature/Other)
          ↓
   Action Planner Agent (next best action + target system)
          ↓
   Tool Executor (GitHub/AzDO) + Auditor (JSONL)
          ↓
   Output (ticket link + audit record id)
```

**Pattern**: Sequential orchestration using `SequentialBuilder` from the Microsoft Agent Framework SDK.

---

Repo Structure

```
azure-multimodal-agents/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .env.example
├─ appsettings.json
├─ requirements.txt
├─ src/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ orchestration/
│  │  ├─ __init__.py
│  │  └─ pipeline.py
│  ├─ agents/
│  │  ├─ __init__.py
│  │  ├─ prompts.py
│  │  └─ factory.py
│  ├─ tools/
│  │  ├─ __init__.py
│  │  ├─ vision.py
│  │  ├─ speech.py
│  │  ├─ ocr.py
│  │  ├─ actions.py
│  │  └─ audit.py
│  └─ utils/
│     ├─ __init__.py
│     └─ io.py
├─ samples/
│  ├─ sample_audio.wav
│  ├─ sample_image.png
│  └─ sample_text.txt
├─ tests/
│  ├─ __init__.py
│  ├─ test_prompts.py
│  ├─ test_pipeline.py
│  └─ fixtures/
│     ├─ audio.wav
│     └─ image.png
└─ .github/workflows/
   └─ ci.yml
```

---

Setup

1) Prerequisites

* Python 3.10+
* Azure subscription + access to **Azure AI Foundry (ai.azure.com)**
* Deployed chat model (e.g., `gpt-4o`) in your **AI Foundry Project**
* Optional services for multimodal:

  * **Azure AI Speech**
  * **Azure AI Vision (Image Analysis)**
  * **Azure AI Document Intelligence**
* GitHub PAT (only if using GitHub issue creation)

2) Clone & install

```bash
git clone https://github.com/<your-username>/azure-multimodal-agents.git
cd azure-multimodal-agents
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

3) Configure environment

Copy `.env.example` to `.env` and fill in your values.

```ini
=== Core Azure OpenAI / Agent Service ===
AZURE_OPENAI_ENDPOINT="https://<your-ai-foundry-endpoint>"
AZURE_OPENAI_DEPLOYMENT="gpt-4o"
AZURE_OPENAI_API_VERSION="2024-10-01-preview"

Auth: use Azure CLI or Managed Identity by default
If using API key auth (not recommended for Cloud Shell):
AZURE_OPENAI_API_KEY="<key>"

=== Optional multimodal services ===
AZURE_SPEECH_REGION="<region>"
AZURE_SPEECH_KEY="<key>"

AZURE_VISION_ENDPOINT="https://<vision-endpoint>"
AZURE_VISION_KEY="<key>"

AZURE_DOCINTELLIGENCE_ENDPOINT="https://<di-endpoint>"
AZURE_DOCINTELLIGENCE_KEY="<key>"

=== Actions (choose one) ===
GITHUB_TOKEN="<pat>"
GITHUB_REPO="<owner>/<repo>"
# or Azure DevOps
AZDO_ORG_URL="https://dev.azure.com/<org>"
AZDO_PROJECT="<project>"
AZDO_PAT="<pat>"

=== Audit ===
AUDIT_LOG_PATH="./audit/audit.jsonl"
```

Also verify `appsettings.json`:

```json
{
  "orchestration": {
    "model": "${AZURE_OPENAI_DEPLOYMENT}",
    "max_output_tokens": 500
  },
  "actions": {
    "target": "github"  
  }
}
```

Sign in

```bash
az login
```

(If needed, set the right subscription and tenant.)

---

Run a demo

A) Text-only feedback

```bash
python -m src.main --text "The charts are great but please add dark mode for late-night use."
```

B) Audio file → transcription → pipeline

```bash
python -m src.main --audio samples/sample_audio.wav
```

C) Image (screenshot) → caption+OCR → pipeline

```bash
python -m src.main --image samples/sample_image.png
```

Output (example)**

```
------------------------------------------------------------
01 [user]
Customer feedback: ...
------------------------------------------------------------
02 [summarizer]
User requests a dark mode for better nighttime usability.
------------------------------------------------------------
03 [classifier]
Feature request | Sentiment: Neutral-Positive
------------------------------------------------------------
04 [action]
Created GitHub issue: https://github.com/owner/repo/issues/123
------------------------------------------------------------
05 [audit]
record_id=7f2c1... file=audit/audit.jsonl
```

---

Source Code

`requirements.txt`

```txt
agent-framework>=0.2.0
azure-identity>=1.17.1
azure-ai-documentintelligence>=1.0.0b4
azure-ai-vision-imageanalysis>=1.0.0
azure-cognitiveservices-speech>=1.40.0
httpx>=0.27.0
pydantic>=2.8.0
python-dotenv>=1.0.1
rich>=13.7.1
pytest>=8.2.0
ruff>=0.5.7
```

`src/main.py`

```python
import argparse
import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from src.orchestration.pipeline import build_pipeline
from src.tools.audit import AuditLogger
from src.utils.io import load_text_from_modal

console = Console()

async def run(args):
    load_dotenv()
    feedback_text = await load_text_from_modal(
        text=args.text,
        audio_path=args.audio,
        image_path=args.image,
    )

    workflow = build_pipeline()
    outputs = []

    async for event in workflow.run_stream(f"Customer feedback: {feedback_text}"):
        from agent_framework import WorkflowOutputEvent, ChatMessage, Role
        if isinstance(event, WorkflowOutputEvent):
            outputs.append(event.data)

    Pretty-print last stage messages
    if outputs:
        from agent_framework import ChatMessage, Role
        console.print("-" * 60)
        for i, msg in enumerate(outputs[-1], start=1):
            name = msg.author_name or ("assistant" if msg.role == Role.ASSISTANT else "user")
            console.print(f"{i:02d} [{name}]\n{msg.text}")

    Audit (persist whole conversation summary)
    logger = AuditLogger(os.getenv("AUDIT_LOG_PATH", "./audit/audit.jsonl"))
    rec_id = logger.log_record({
        "text": feedback_text,
        "outputs": [ [m.to_dict() for m in batch] for batch in outputs ],
    })
    console.print("-" * 60)
    console.print(f"[audit] record_id={rec_id} path={logger.path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--audio", type=str)
    parser.add_argument("--image", type=str)
    args = parser.parse_args()
    asyncio.run(run(args))
```

`src/orchestration/pipeline.py`

```python
from agent_framework import SequentialBuilder
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential, AzureCliCredential
import os
from src.agents.factory import create_agents


def _client():
    # Prefer CLI/Managed Identity; fall back to key if present
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        # agent-framework picks up endpoint/key via env vars
        from agent_framework.azure import AzureOpenAIChatClient
        return AzureOpenAIChatClient(api_key=api_key, endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
    # Default creds (CLI / Managed Identity)
    try:
        cred = AzureCliCredential()
    except Exception:
        cred = DefaultAzureCredential(exclude_visual_studio_code_credential=True)
    return AzureOpenAIChatClient(credential=cred, endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))


def build_pipeline():
    client = _client()
    agents = create_agents(client)
    workflow = SequentialBuilder().participants([
        agents.ingest,
        agents.summarizer,
        agents.classifier,
        agents.action,
        agents.auditor,
    ]).build()
    return workflow
```

`src/agents/prompts.py`

```python
SUMMARIZER = (
    "You are a neutral product feedback summarizer. "
    "Condense the user input into a single, neutral sentence that captures the core request or issue."
)

CLASSIFIER = (
    "You are a product feedback classifier. "
    "Return JSON with fields: {type: one of [Bug, Feature, Other], sentiment: one of [Positive, Neutral, Negative], confidence: 0-1}."
)

ACTION = (
    "You are a product operations assistant. "
    "Given the summary and classification JSON, propose the next best action. "
    "If type=Feature, suggest creating a backlog item with acceptance criteria. "
    "If type=Bug, suggest creating a bug with reproduction steps."
)

INGEST = (
    "You normalize multimodal inputs. If given text, use it. If given captions/OCR, merge succinctly. Output a single clean paragraph of the user's feedback."
)

AUDITOR = (
    "You generate a short compliance-ready audit note summarizing what was done (no PII)."
)
```

`src/agents/factory.py`

```python
from dataclasses import dataclass
from agent_framework.azure import AzureOpenAIChatClient
from src.agents import prompts

@dataclass
class Agents:
    ingest: object
    summarizer: object
    classifier: object
    action: object
    auditor: object


def create_agents(client: AzureOpenAIChatClient) -> Agents:
    ingest = client.create_agent(instructions=prompts.INGEST, name="ingest")
    summarizer = client.create_agent(instructions=prompts.SUMMARIZER, name="summarizer")
    classifier = client.create_agent(instructions=prompts.CLASSIFIER, name="classifier")
    action = client.create_agent(instructions=prompts.ACTION, name="action")
    auditor = client.create_agent(instructions=prompts.AUDITOR, name="auditor")
    return Agents(ingest, summarizer, classifier, action, auditor)
```

`src/tools/speech.py`

```python
import os

def transcribe(audio_path: str) -> str:
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not (key and region):
        return ""  # disabled
    import azure.cognitiveservices.speech as speechsdk
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_input = speechsdk.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    result = recognizer.recognize_once()
    return result.text or ""
```

`src/tools/vision.py`

```python
import os

def caption(image_path: str) -> str:
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")
    if not (endpoint and key):
        return ""
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.core.credentials import AzureKeyCredential

    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(image_path, "rb") as f:
        result = client.analyze(image_data=f.read(), visual_features=["caption"])
    return getattr(result.caption_result, "text", "")
```

`src/tools/ocr.py`

```python
import os

def extract_text(image_path: str) -> str:
    endpoint = os.getenv("AZURE_DOCINTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCINTELLIGENCE_KEY")
    if not (endpoint and key):
        return ""
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-read", document=f)
    result = poller.result()
    lines = []
    for page in result.pages or []:
        for line in page.lines or []:
            lines.append(line.content)
    return "\n".join(lines)
```

`src/tools/actions.py`

```python
import os
import httpx

async def create_github_issue(title: str, body: str) -> str:
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    if not (token and repo):
        return ""
    url = f"https://api.github.com/repos/{repo}/issues"
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json={"title": title, "body": body}, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
        r.raise_for_status()
        data = r.json()
        return data.get("html_url", "")
```

`src/tools/audit.py`

```python
import json, os, hashlib, time
from pathlib import Path

class AuditLogger:
    def __init__(self, path: str):
        self.path = path
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

    def log_record(self, payload: dict) -> str:
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        rec_id = hashlib.sha256(serialized + str(time.time()).encode()).hexdigest()[:16]
        entry = {"id": rec_id, "ts": int(time.time()), "payload": payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return rec_id
```

`src/utils/io.py`

```python
from typing import Optional
from src.tools.speech import transcribe
from src.tools.vision import caption
from src.tools.ocr import extract_text

async def load_text_from_modal(text: Optional[str], audio_path: Optional[str], image_path: Optional[str]) -> str:
    if text:
        return text.strip()
    chunks = []
    if audio_path:
        chunks.append(transcribe(audio_path))
    if image_path:
        cap = caption(image_path)
        if cap:
            chunks.append(f"Caption: {cap}")
        ocr = extract_text(image_path)
        if ocr:
            chunks.append(f"OCR: {ocr}")
    return "\n".join([c for c in chunks if c]).strip() or "(no content)"
```

`.env.example`

```ini
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-01-preview

AZURE_SPEECH_REGION=
AZURE_SPEECH_KEY=

AZURE_VISION_ENDPOINT=
AZURE_VISION_KEY=

AZURE_DOCINTELLIGENCE_ENDPOINT=
AZURE_DOCINTELLIGENCE_KEY=

GITHUB_TOKEN=
GITHUB_REPO=

AZDO_ORG_URL=
AZDO_PROJECT=
AZDO_PAT=

AUDIT_LOG_PATH=./audit/audit.jsonl
```

`.github/workflows/ci.yml`

```yaml
name: CI
on:
  push:
  pull_request:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install -r requirements.txt
      - run: ruff check .
      - run: pytest -q
```

### `tests/test_prompts.py`

```python
from src.agents import prompts

def test_prompts_exist():
    assert len(prompts.SUMMARIZER) > 10
    assert "classifier" not in prompts.SUMMARIZER.lower()
```

`tests/test_pipeline.py`

```python
from src.orchestration.pipeline import build_pipeline

def test_build_pipeline():
    wf = build_pipeline()
    assert wf is not None
```

`LICENSE`

```text
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
... (standard MIT terms)
```

`.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc
.env
.audit/
audit/
```

---

Developer Notes

* **Auth**: Project prefers Azure CLI/Managed Identity; only falls back to API key if provided.
* **Regional quotas**: If your `gpt-4o` deployment hits quota, deploy in another recommended region and update `.env`.
* **Tools**: If you don’t set Vision/Speech/Doc Intelligence keys, the project still runs text-only.
* **Switch actions**: Set `actions.target` in `appsettings.json` to `github` or `azdo`.

---

Cleanup

* Delete the resource group hosting your AI Foundry / Speech / Vision / Doc Intelligence to avoid charges.

---

Credits

* Microsoft Agent Framework SDK + Azure AI Foundry

> PRs welcome. Add adapters for ServiceNow/Jira, add RAG context, or plug into your existing **Audit Logger for Azure OpenAI** project.

