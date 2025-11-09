"""
FastAPI app: AdventureWorks Sales Dashboard (Fixed Version)
Author: ChatGPT (2025)
Description:
- Loads Excel → SQLite
- Exploratory, RFM, ML & Dashboard endpoints
Run:
    pip install fastapi uvicorn pandas scikit-learn plotly jinja2 python-multipart openpyxl requests
    python fastapi_adventureworks_dashboard_fixed.py
Then open: http://localhost:5000/dashboard
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import sqlite3
import os
import numpy as np
import json
import plotly.graph_objects as go
from jinja2 import Template
import uvicorn
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from requests import post

# ---------- CONFIG ----------
DB = 'C:/Users/diyor/Downloads/AdventureSalesProject/AdventureSalesProject/adventureworks.db'
EXCEL_PATH = 'C:/Users/diyor/Downloads/AdventureSalesProject/AdventureSalesProject/data/AdventureWorks Sales.xlsx'

app = FastAPI(title='AdventureWorks Sales Dashboard')

# ---------- DB INIT ----------
def init_db_from_excel(excel_path=EXCEL_PATH, db_path=DB):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found at {excel_path}")
    xls = pd.ExcelFile(excel_path)
    con = sqlite3.connect(db_path)
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            df.columns = [c.strip().replace(' ', '_') for c in df.columns]
            df.to_sql(sheet.lower(), con, if_exists='replace', index=False)
            print(f"Loaded sheet: {sheet}")
        except Exception as e:
            print(f"Failed to parse sheet {sheet}: {e}")
    con.close()
    return True


@app.get('/init-db')
async def api_init_db():
    try:
        init_db_from_excel()
        return {'status': 'ok', 'message': 'Database initialized from Excel'}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


# ---------- DB HELPER ----------
def load_table(name):
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{name}"', con)
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df


# ---------- EDA ----------
@app.get('/eda')
async def eda():
    sales = load_table('sales_data')
    if sales.empty:
        return JSONResponse({'error': 'sales_data table not found'}, status_code=400)

    stats = {
        'rows': len(sales),
        'columns': list(sales.columns),
        'total_sales_amount': float(sales['SalesAmount'].sum()) if 'SalesAmount' in sales.columns else None,
        'unique_customers': int(sales['CustomerKey'].nunique()) if 'CustomerKey' in sales.columns else None,
        'date_range': None
    }

    if 'OrderDateKey' in sales.columns:
        sales['OrderDate'] = pd.to_datetime(sales['OrderDateKey'].astype(str), format='%Y%m%d', errors='coerce')
        stats['date_range'] = [
            str(sales['OrderDate'].min().date()),
            str(sales['OrderDate'].max().date())
        ]
    return stats


# ---------- RFM ----------
@app.get('/rfm')
async def rfm(k_segments: int = 4):
    sales = load_table('sales_data')
    if sales.empty:
        return JSONResponse({'error': 'sales_data not found'}, status_code=400)

    for col in ['CustomerKey', 'OrderDateKey']:
        if col not in sales.columns:
            return JSONResponse({'error': f'Missing {col} column'}, status_code=400)

    sales['OrderDate'] = pd.to_datetime(sales['OrderDateKey'].astype(str), format='%Y%m%d', errors='coerce')
    sales['SalesAmount'] = sales.get('SalesAmount', pd.Series(1, index=sales.index))
    snapshot_date = sales['OrderDate'].max() + pd.Timedelta(days=1)

    rfm = sales.groupby('CustomerKey').agg(
        Recency=('OrderDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('OrderDate', 'count'),
        Monetary=('SalesAmount', 'sum')
    )

    r_labels = range(4, 0, -1)
    rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=r_labels)
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=range(1,5))
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=range(1,5))
    rfm['RFM_Score'] = rfm[['R_score','F_score','M_score']].astype(int).sum(axis=1)

    def label(score):
        if score >= 9: return 'Best'
        elif score >= 6: return 'Loyal'
        elif score >= 4: return 'At Risk'
        else: return 'Lost'

    rfm['Segment'] = rfm['RFM_Score'].apply(label)
    out = rfm.reset_index().head(100).to_dict(orient='records')
    return {'snapshot_date': str(snapshot_date.date()), 'sample': out}


# ---------- ML MODELS ----------
@app.post('/train-models')
async def train_models():
    sales = load_table('sales_data')
    if sales.empty:
        return JSONResponse({'error': 'sales_data not found'}, status_code=400)

    sales['OrderDate'] = pd.to_datetime(sales['OrderDateKey'].astype(str), format='%Y%m%d', errors='coerce')
    sales['SalesAmount'] = sales.get('SalesAmount', pd.Series(1, index=sales.index))

    cust = sales.groupby('CustomerKey').agg(
        last_order=('OrderDate', 'max'),
        freq=('OrderDate', 'count'),
        monetary=('SalesAmount', 'sum')
    ).reset_index()

    cust['recency'] = (sales['OrderDate'].max() - cust['last_order']).dt.days
    cust['churn'] = (cust['recency'] > 180).astype(int)

    X = cust[['recency', 'freq', 'monetary']].fillna(0)
    y = cust['churn']

    if len(cust) < 10:
        return {'error': 'Not enough customers to train models (need >=10)'}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

    reg = GradientBoostingRegressor(n_estimators=50, random_state=42)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, cust['monetary'], test_size=0.2, random_state=42)
    reg.fit(Xr_train, yr_train)
    rmse = float(mean_squared_error(yr_test, reg.predict(Xr_test), squared=False))

    pickle.dump(clf, open('churn_model.pkl', 'wb'))
    pickle.dump(reg, open('clv_model.pkl', 'wb'))

    return {'churn_auc': auc, 'clv_rmse': rmse}


# ---------- DASHBOARD ----------
@app.get('/dashboard', response_class=HTMLResponse)
async def dashboard():
    sales = load_table('sales_data')
    if sales.empty:
        return HTMLResponse('<h3>sales_data not found. Run <a href="/init-db">/init-db</a>.</h3>')

    if 'OrderDateKey' in sales.columns:
        sales['OrderDate'] = pd.to_datetime(sales['OrderDateKey'].astype(str), format='%Y%m%d', errors='coerce')
        if 'SalesAmount' in sales.columns:
            sales_ts = sales.groupby(pd.Grouper(key='OrderDate', freq='M')).agg(total_sales=('SalesAmount', 'sum')).reset_index()
        else:
            sales_ts = sales.groupby(pd.Grouper(key='OrderDate', freq='M')).size().reset_index(name='total_sales')
    else:
        sales_ts = pd.DataFrame({'OrderDate':[], 'total_sales':[]})

    sales['SalesAmount'] = sales.get('SalesAmount', pd.Series(1, index=sales.index))
    top_customers = sales.groupby('CustomerKey').agg(total=('SalesAmount','sum')).reset_index().nlargest(10, 'total')

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=sales_ts['OrderDate'].astype(str).tolist(),
        y=sales_ts['total_sales'].tolist(),
        name='Monthly Sales'
    ))

    fig_top = go.Figure([go.Bar(
        x=top_customers['CustomerKey'].astype(str).tolist(),
        y=top_customers['total'].tolist()
    )])

    template = Template('''
    <html><head>
      <title>AdventureWorks Dashboard</title>
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
      <style>body{font-family:Arial;padding:20px}</style>
    </head><body>
      <h1>AdventureWorks Sales Dashboard</h1>
      <div id='ts' style='width:100%;height:400px;'></div>
      <div id='top' style='width:100%;height:400px;'></div>
      <script>
        var fig1 = {{fig1}};
        var fig2 = {{fig2}};
        Plotly.newPlot('ts', fig1.data, fig1.layout || {});
        Plotly.newPlot('top', fig2.data, fig2.layout || {});
      </script>
    </body></html>
    ''')

    html = template.render(fig1=json.dumps(fig_ts.to_plotly_json()), fig2=json.dumps(fig_top.to_plotly_json()))
    return HTMLResponse(content=html)


if __name__ == '__main__':
    if os.path.exists(EXCEL_PATH) and not os.path.exists(DB):
        init_db_from_excel()
    uvicorn.run(app, port=5000)
