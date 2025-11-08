# Freedom Assignment

## Prerequisites

- Python 3.11+
- `pip` for dependency installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Running the FastAPI service

Launch the development server:

```bash
uvicorn src.api.server:app --reload
```

Open your browser at http://127.0.0.1:8000 to access the UI. Enter an industry name and click **Generate**; the backend will run the news → idea → image pipeline and stream the results back to the page. Generated assets are saved under the configured `output` directory and are served directly from the `/static` route.

## CLI pipeline (optional)

The existing CLI entry point still works:

```bash
python -m src.main
```

Export the `INDUSTRY` environment variable before running the script to select a different segment.