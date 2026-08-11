# Panel — Blood Work to Diet Plan

A small Flask app (no Streamlit) that reads an uploaded blood work report
image, extracts test values with a vision model, and turns the flagged
values into a plain-English diet plan.

## Structure

```
health_diet_app/
├── backend.py          Flask app: serves the UI, exposes POST /analyze
├── diet_pipeline.py    Core ML logic: vision extraction + diet agent (importable/testable on its own)
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── uploads/             temp storage, files are deleted right after analysis
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd health_diet_app
pip install -r requirements.txt
cp .env.example .env      # then paste your real GROQ_API_KEY into .env
python backend.py
```

Open http://localhost:5000

## Testing the pipeline alone (no web server)

```bash
python diet_pipeline.py path/to/report.png
```

Prints the structured JSON result straight to the terminal — useful for
checking that extraction is working before debugging the UI.

## Notes

- `/analyze` deletes the uploaded image immediately after processing —
  nothing is stored on disk longer than one request.
- Vision extraction uses `meta-llama/llama-4-scout-17b-16e-instruct`;
  the diet-plan agent uses `openai/gpt-oss-20b`. Swap either model string
  in `diet_pipeline.py` if Groq changes what's available on your account —
  run `client.models.list()` to check current model IDs.
- This is general dietary information, not a diagnosis — the UI and the
  agent's system prompt both say so on purpose. Keep that disclaimer if
  you extend this further.
