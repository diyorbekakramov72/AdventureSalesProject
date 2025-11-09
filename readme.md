# AdventureSalesProject

This project is a ready-to-run example for the AdventureWorks Sales assignment.
It includes ETL, preprocessing, EDA, RFM segmentation and a simple FastAPI service.

## How to run

1. Create and activate a Python virtual environment:
   - `python -m venv venv`
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

2. Install requirements:
   - `pip install -r requirements.txt`

3. Run the ETL & analysis notebook:
   - Open `main.ipynb` in Jupyter or VS Code and run the cells. This will create `sales.db`.

4. Run the FastAPI server:
   - `uvicorn app:app --reload`
   - Open http://127.0.0.1:8000/docs

Files included:
- data/*.csv : dummy Sales, Customers, Products
- main.ipynb  : ETL, EDA, RFM cells
- app.py      : FastAPI endpoints
- init_db.py  : script to create SQLite db from CSVs
- requirements.txt
