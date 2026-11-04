# Duplicate interface and file cleanup

The React frontend is the only dashboard. Removed the retired Streamlit source/configuration and its eight UI tests, four command wrapper scripts, the forwarding requirements.txt file, obsolete development handoff reports, and ten duplicate or superseded generated report/table/figure copies. The authoritative study artifacts remain under artifacts/experiments/.

Removed ten unused locked and installed dependencies: Streamlit, Altair, PyDeck, attrs, itsdangerous, jsonschema, jsonschema-specifications, python-multipart, referencing and rpds-py. The lockfile was regenerated from the dependency closure using existing installed versions. Requests is now an explicit acquisition dependency; FastAPI/Uvicorn are primary application dependencies; HTTPX is declared for API tests.

The user requested retaining the flattened src/ layout. Explicit setuptools package mapping preserves the fabfederate import namespace without a second source tree. API artifact discovery and study provenance now resolve the repository root from the new physical paths.

Validation after cleanup: 100 Python tests pass (the eight removed tests exercised only the retired interface); Ruff passes; TypeScript checks and the production frontend build pass. The CLI help loads, the API discovers five complete studies and defaults to nasa-final, and the existing saved FedAvg model still predicts exactly 0.2428964078426361 mm for the recorded feature example. Historical verification logs describe the implementation as it existed at their recorded dates.
