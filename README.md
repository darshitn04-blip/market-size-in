# MarketSize.in — Quick Commerce Expansion Model

Decision-support dashboard for prioritising Indian Tier-2 expansion cities for a quick-commerce dark-store network.

## What it does

- Creates a transparent bottom-up TAM from projected households × basket size × monthly order frequency × adoption.
- Cross-checks it against a top-down allocation of national quick-commerce GMV.
- Ranks 45 cities using income, density, internet access, demand, competitive intensity, and a logistics-cost proxy.
- Uses scikit-learn K-Means to group similar city opportunity profiles, maps candidates, and writes a Power BI/Tableau-ready SQLite warehouse (`market_size.db`).

## Start the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

## Data integrity

Population is based on Census 2011 urban-agglomeration data and projected using the stated growth-rate model. Income and internet access are state-proxy inputs, deliberately labelled as such. Competitor stores start at zero and must be uploaded as a verified `city,competitor_stores` CSV; no fictional scraping data is included.

Before a real investment decision, replace proxy fields with licensed/current city-level data and validate each selected micro-market's economics, rider availability, real estate, and competitor coverage.
