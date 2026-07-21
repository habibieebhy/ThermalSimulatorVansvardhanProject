# Vansvardhan Mattress Research Lab

Version **1.0.0** is a standalone evidence, engineering-inference, and thermal
simulation platform built from the uploaded `ThermalSimulatorVansvardhanProject`
baseline. The original five-prototype thermal engine is preserved and now sits
beside a complete mattress-market research workflow.

It does not depend on BRIXTA, a vector database, Redis, Celery, or any other
project.

## What the platform does

```text
Company, market, aliases, known URLs
                 ↓
Tavily basic web search (free Researcher tier)
                 ↓
Bounded crawler + sitemap + articles + text PDFs
                 ↓
Content-addressed raw evidence storage
                 ↓
Deterministic extraction, then optional Gemini extraction
                 ↓
Normalized products, variants, layers, claims, and provenance
                 ↓
Collection-only Excel workbook and review queue
                 ↓
Optional constraints + similarity + Bayesian ranking
                 ↓
Candidate constructions + preliminary thermal screening
                 ↓
Independent five-prototype six-hour thermal simulator
```

The recommended research order is **evidence first, algorithms second,
simulation third**.

## Trust boundary

- Tavily finds URLs. It does not decide mattress construction.
- Gemini reads captured documents and extracts explicitly stated facts.
- Missing collection values remain missing.
- The constraint solver rejects physically inconsistent candidates.
- Bayesian scores are provisional until calibrated on verified teardowns.
- The thermal simulator is comparative, not CFD or certification.
- Every observed claim retains a source reference and raw artifact.

## System requirements

- macOS, Linux, or Windows
- Python 3.11–3.13; Python 3.12 is recommended
- Internet access for live collection
- A free Tavily API key for web discovery
- A Gemini API key for optional LLM extraction

Gemini Google Search grounding is not required. Tavily documents 1,000 free
credits per month on its Researcher tier without a credit card:

- <https://docs.tavily.com/documentation/api-credits>
- <https://docs.tavily.com/documentation/api-reference/endpoint/search>

## Clean installation

```bash
cd VansvardhanMattressResearchLab-v1.0.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install . --no-deps
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Confirm the correct build:

```bash
cat VERSION
python -c "import mattress_intelligence; print(mattress_intelligence.__version__)"
mattress-lab --help
mattress-sim --help
```

All version checks should report `1.0.0`.

## Private configuration

Never place a real key in `.env.example`. Create a private `.env`:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env`:

```dotenv
MATTRESS_INTEL_LLM_PROVIDER=gemini
GEMINI_API_KEY=your_private_gemini_key
GEMINI_MODEL=gemini-3.5-flash

MATTRESS_INTEL_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=your_private_tavily_key
MATTRESS_INTEL_SEARCH_QUERIES=2

MATTRESS_INTEL_DATA_DIR=./data
MATTRESS_INTEL_OUTPUT_DIR=./outputs
MATTRESS_INTEL_ARTIFACT_DIR=./artifacts
MATTRESS_INTEL_DATABASE_PATH=./data/mattress_intelligence.sqlite3

MATTRESS_INTEL_USER_AGENT=Vansvardhan-Mattress-Research/1.0
MATTRESS_INTEL_REQUEST_TIMEOUT_SECONDS=30
MATTRESS_INTEL_REQUEST_DELAY_SECONDS=1.0
MATTRESS_INTEL_MAX_DOWNLOAD_BYTES=15000000
MATTRESS_INTEL_RENDER_JAVASCRIPT=false
```

Use two searches for the first smoke test. Increase to six only after reviewing
search quality and usage.

Validate both providers without printing either key:

```bash
mattress-lab tavily-check
mattress-lab gemini-check
mattress-lab doctor
```

## Graphical application

```bash
python -m streamlit run app.py
```

Open <http://localhost:8501>. The app has three pages:

1. **Research Workflow** — orientation and trust boundary.
2. **Evidence Collection Lab** — web collection, extraction, review, analysis,
   and Excel download.
3. **Thermal Prototype Lab** — the uploaded five-prototype simulator with live
   graph playback and parameter controls.

Stop Streamlit with `Control+C`.

## First live collection

Start with collection only, two searches, and 20 pages:

```bash
mattress-lab collect \
  --company "Sleepwell" \
  --domain "https://replace-with-verified-official-domain.example" \
  --market "India" \
  --max-pages 20 \
  --llm gemini \
  --search-provider tavily \
  --search \
  --external \
  --output outputs/sleepwell_smoke_collection.xlsx
```

`collect` deliberately skips similarity, Bayesian inference, configuration
generation, and thermal screening. Its purpose is to build and audit the
evidence layer first.

### Unknown brands

Add aliases, known sources, and precise custom searches:

```bash
mattress-lab collect \
  --company "Example Bedding Industries" \
  --domain "https://verified-domain.example" \
  --market "India" \
  --alias "Example Sleep" \
  --alias "EBI Mattress" \
  --seed-url "https://retailer.example/old-model" \
  --seed-url "https://archive.example/catalogue.pdf" \
  --query '"Example Sleep" mattress foam density' \
  --query '"Example Bedding" catalogue filetype:pdf' \
  --max-pages 100 \
  --llm gemini \
  --search-provider tavily \
  --search --external \
  --output outputs/example_bedding_collection.xlsx
```

Custom queries run before the built-in search plan. The plan covers current
catalogues, construction details, brochures, old archives, retailer evidence,
patents, trademarks, and named material technologies.

## What a collection run produces

The Excel workbook contains:

| Sheet | Purpose |
| --- | --- |
| Discovery Log | Query, result rank, URL, title, score, provider usage |
| Evidence Sources | Source type, official status, URL, hash, artifact path |
| Products | Deduplicated catalogue records |
| Variants | Size, thickness, weight, and price variants |
| Layers | Observed layer names, normalized materials, thickness, density |
| Observed Claims | Field value, status, confidence, and evidence |
| Review Queue | Missing, uncertain, and low-confidence records |
| Run Metadata | Warnings, provider mode, and limitations |
| Configurations | Empty in collection-only mode; populated in analysis mode |
| Thermal Screening | Empty in collection-only mode; populated in analysis mode |

Raw evidence is stored by SHA-256 below `artifacts/`. Research runs are stored
in `data/mattress_intelligence.sqlite3`.

List and re-export persisted runs:

```bash
mattress-lab runs
mattress-lab export --run-id RUN_ID --output outputs/regenerated.xlsx
```

## Full intelligence run

After the collection workbook has been reviewed:

```bash
mattress-lab research \
  --company "Sleepwell" \
  --domain "https://replace-with-verified-official-domain.example" \
  --market "India" \
  --max-pages 250 \
  --max-configurations 10 \
  --simulate-top 3 \
  --llm gemini \
  --search-provider tavily \
  --search --external \
  --output outputs/sleepwell_intelligence.xlsx
```

The analysis path uses independent algorithms:

1. **Constraint satisfaction** — exact total thickness and optional mass
   tolerance; OR-Tools CP-SAT when installed, bounded Python fallback otherwise.
2. **Similarity** — TF-IDF unigrams/bigrams and cosine similarity over known
   product properties.
3. **Bayesian ranking** — material-density priors plus weight, firmness, and
   comparable-product likelihoods.
4. **Confidence** — posterior adjusted by evidence completeness; explicitly
   provisional and uncalibrated.
5. **Knowledge graph** — company, brand, product, layer, material, claim,
   source, and possible-configuration edges with traversal.
6. **Thermal screening** — fast comparative resistance and thermal-inertia
   screen for candidate stacks.

Gemini does not participate in those calculations.

## Five-prototype thermal simulator

Run the original uploaded simulation directly:

```bash
mattress-sim \
  --output outputs/mattress_investor_dashboard.png \
  --csv outputs/mattress_simulation.csv
```

Or:

```bash
python run_simulation.py
```

The model compares:

| Prototype | Mechanism | Electrical profile |
| --- | --- | ---: |
| P1 Aero-Natural | Open-cell latex plus finite PCM | 0 W |
| P2 Eco-Battery | Ambient-water loop and passive radiator | Constant 5 W |
| P3 Core-Chiller | Controlled Peltier and water block | 0–60 W |
| P4 Hyper-Conductive | Graphite spreading to rejecting edges | 0 W |
| P5 Dual-Zone | Turbo followed by pulsed eco maintenance | 40 W then pulsed 10 W |

The occupied mattress zone uses:

```text
C = mass × specific heat
q = conductivity × area × temperature difference / path length
ΔT = net heat flow × Δt / C
```

At each time step the engine calculates body heat input, room-side rejection,
architecture-specific removal, electrical power, and accumulated Wh. Final
stabilized temperature is the last-15-minute mean.

The Streamlit thermal page provides the original controls for room/body/foam,
P1 PCM, P2 radiator, P3 controller, P4 graphite spreader, and P5 hybrid duty
cycle, plus optional graph playback.

## PDF and browser behavior

- HTML, external articles, plain text, and text-based PDFs are supported.
- Image-only scanned PDFs are preserved but need a future OCR integration.
- The crawler obeys robots.txt by default and does not bypass authentication,
  paywalls, CAPTCHAs, or access controls.
- Raw HTTP is tried first. For JavaScript shells, install Chromium:

```bash
python -m playwright install chromium
```

Then set:

```dotenv
MATTRESS_INTEL_RENDER_JAVASCRIPT=true
```

## Offline demo

```bash
mattress-lab demo
```

This requires no web search or API key and creates:

```text
outputs/demo_mattress_intelligence.xlsx
data/mattress_intelligence.sqlite3
```

## API

Start the optional FastAPI service:

```bash
uvicorn mattress_intelligence.api:app --host 0.0.0.0 --port 8000 --reload
```

Open <http://localhost:8000/docs>.

Key endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Runtime/provider health |
| GET | `/v1/runs` | Persisted runs |
| GET | `/v1/runs/{run_id}` | Complete result |
| GET | `/v1/runs/{run_id}/excel` | Excel artifact |
| POST | `/v1/collect` | Collection without inference |
| POST | `/v1/research` | Collection plus algorithms |

The API is synchronous for local development. Move long jobs to a background
worker before multi-user deployment.

## Docker

Create `.env`, then:

```bash
docker compose up --build
```

```text
UI:  http://localhost:8501
API: http://localhost:8000/docs
```

Host-mounted `data/`, `outputs/`, and `artifacts/` preserve results.

## Tests

```bash
python -m unittest discover -s tests -v
```

Optional developer tooling:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Project structure

```text
VansvardhanMattressResearchLab-v1.0.0/
├── app.py
├── pages/
│   ├── 0_Home.py
│   ├── 1_Evidence_Collection_Lab.py
│   └── 2_Thermal_Prototype_Lab.py
├── src/
│   ├── mattress_intelligence/
│   │   ├── search.py
│   │   ├── crawler.py
│   │   ├── llm.py
│   │   ├── extraction.py
│   │   ├── entities.py
│   │   ├── configurations.py
│   │   ├── similarity.py
│   │   ├── inference.py
│   │   ├── graph.py
│   │   ├── thermal.py
│   │   ├── exporter.py
│   │   ├── storage.py
│   │   ├── pipeline.py
│   │   ├── api.py
│   │   └── cli.py
│   └── mattress_thermal/
│       ├── simulation.py
│       └── cli.py
├── examples/
├── tests/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── compose.yaml
└── VERSION
```

## Troubleshooting

### `mattress-lab: command not found`

```bash
source .venv/bin/activate
python -m pip install . --no-deps --force-reinstall
```

Fallback:

```bash
python -m mattress_intelligence.cli --help
```

### `ModuleNotFoundError`

Run commands from the project root with the virtual environment active, then
reinstall the project.

### Provider key missing

Confirm that `.env` exists in the project root. Do not print or share its
contents. Use `tavily-check` and `gemini-check` to validate providers safely.

### No products extracted

Review Discovery Log and Warnings, verify the official domain, enable external
evidence, add aliases/seed URLs/custom queries, then enable browser rendering
only if the site is a JavaScript shell.

### `429` provider error

Reduce `MATTRESS_INTEL_SEARCH_QUERIES`, reduce page count, and wait for rate
limits to recover. Automatic retries cannot bypass an exhausted quota.

## Engineering limitations

This platform cannot recover information that was never published. It does not
turn marketing names into confirmed chemical composition. Candidate densities
and layer stacks are hypotheses until validated by teardown, mass measurement,
coupon testing, guarded-hot-plate testing, thermal manikin testing, or another
appropriate physical method.

The five-prototype model is a transparent lumped-parameter comparator, not
OpenFOAM, CFD, FEA, or certification evidence. Replace effective conductivity,
COP, coupling, path dimensions, PCM capacity, and controller parameters with
measured values before making product claims.
