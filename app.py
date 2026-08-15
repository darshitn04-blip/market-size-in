from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from analytics import default_cities, run_model, save_model_to_sqlite

st.set_page_config(page_title="MarketSize.in | Quick Commerce", page_icon="⚡", layout="wide")

@st.cache_data
def load_cities(): return default_cities()

st.title("MarketSize.in  ⚡")
st.caption("Quick-commerce dark-store expansion model for Tier-2 Indian cities")

with st.sidebar:
    st.header("Model assumptions")
    basket = st.number_input("Average basket size (₹)", 200, 1000, 420, 10)
    frequency = st.slider("Orders per adopting household / month", .5, 6.0, 2.5, .1)
    adoption = st.slider("Addressable household adoption", .02, .40, .12, .01)
    national_gmv = st.number_input("India quick-commerce GMV (₹ crore)", 10000, 300000, 70000, 1000)
    som_share = st.slider("Target first-year share of city TAM", .02, .30, .12, .01)
    st.divider()
    st.caption("Competition counts default to 0 until manually verified or scraped. They are deliberately not fabricated.")

cities = load_cities()
competition_upload = st.sidebar.file_uploader("Upload verified competitor counts CSV", type="csv", help="Columns: city, competitor_stores")
if competition_upload:
    counts = pd.read_csv(competition_upload)
    cities = cities.drop(columns="competitor_stores").merge(counts[["city", "competitor_stores"]], on="city", how="left").fillna({"competitor_stores": 0})
output = run_model(cities, basket, frequency, adoption, national_gmv, som_share)
scores = output.cities
save_model_to_sqlite(output)

tabs = st.tabs(["Recommendation", "Market sizing", "City map", "Evidence & handoff"])
with tabs[0]:
    best = scores.iloc[0]
    a,b,c,d = st.columns(4)
    a.metric("Recommended launch city", best.city, f"Score {best.attractiveness_score:.0f}/100")
    b.metric("Bottom-up TAM", f"₹{best.bottom_up_tam_cr:,.0f} Cr")
    c.metric("Year-1 SOM", f"₹{best.som_cr:,.0f} Cr")
    d.metric("Target city cohort", f"Cluster {best.city_cluster}")
    st.subheader("S-C-R: a decision-ready recommendation")
    st.markdown(f"**Situation.** Quick commerce needs dense, digitally reachable demand outside the largest metros.  \
**Complication.** Population alone overstates demand and ignores household spend, serviceability, and competitive intensity.  \
**Resolution.** Prioritize **{best.city}**, then validate micro-market coverage and competitor availability before committing dark-store capex.")
    st.plotly_chart(px.bar(scores.head(12).sort_values("attractiveness_score"), x="attractiveness_score", y="city", orientation="h", color="som_cr", color_continuous_scale="Oranges", title="Top expansion candidates"), use_container_width=True)

with tabs[1]:
    left,right = st.columns(2)
    left.plotly_chart(px.scatter(scores, x="bottom_up_tam_cr", y="top_down_tam_cr", size="population_2026_lakh", color="attractiveness_score", hover_name="city", title="Bottom-up vs top-down TAM cross-check"), use_container_width=True)
    right.plotly_chart(px.scatter(scores, x="density_proxy", y="income_index", size="bottom_up_tam_cr", color="city_cluster", hover_name="city", title="City opportunity clusters"), use_container_width=True)
    st.dataframe(scores[["rank","city","state","bottom_up_tam_cr","top_down_tam_cr","tam_crosscheck_gap_pct","attractiveness_score","som_cr"]].round(1), hide_index=True, use_container_width=True)

with tabs[2]:
    st.plotly_chart(px.scatter_map(scores, lat="lat", lon="lon", size="som_cr", color="attractiveness_score", hover_name="city", hover_data=["rank","bottom_up_tam_cr","competitor_stores"], zoom=3.4, center={"lat":22.5,"lon":79}, color_continuous_scale="YlOrRd", title="India expansion attractiveness map"), use_container_width=True)
    st.info("For a true choropleth, join this score table to a city-boundary GeoJSON using GeoPandas. Point-level mapping is used here because city boundaries are not bundled with the project.")

with tabs[3]:
    st.subheader("Evidence, limitations, and Power BI/Tableau handoff")
    st.markdown("""**Public-source baseline**
    - City population/growth: [data.gov.in urban-agglomeration Census 2011 resource](https://www.data.gov.in/resource/state-wise-details-top-fifteen-cities-terms-decadal-growth-population-2001-2011-ministry)
    - Household consumption: [MoSPI HCES 2022–23](https://mospi.gov.in/themes/product/71-household-consumption-expenditure-survey-hces)
    - State macroeconomic cross-check: [RBI Database on Indian Economy](https://data.rbi.org.in/)

    Population is a Census 2011 baseline projected with the displayed growth rate; income and internet fields are state-level proxy inputs. Replace them with city-level downloads before an investment decision. Competitor counts require a lawful, terms-compliant manual audit or approved data partner—this app does not claim to have scraped them.
    """)
    st.download_button("Download city scores CSV", scores.to_csv(index=False).encode(), "city_expansion_scores.csv", "text/csv")
    st.code("SELECT city, rank, attractiveness_score, bottom_up_tam_cr, top_down_tam_cr, som_cr\nFROM city_expansion_scores\nORDER BY rank;", language="sql")
    st.caption("A local Power BI/Tableau-ready SQLite table is regenerated at market_size.db whenever the model changes.")
