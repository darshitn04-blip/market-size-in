"""Transparent market-sizing and city-ranking engine for quick-commerce expansion."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ModelOutput:
    cities: pd.DataFrame
    assumptions: dict[str, float]


# Census 2011 urban-agglomeration population baseline. Derived fields are explicitly model inputs.
CITY_BASELINE = [
    ("Agra","Uttar Pradesh",17.5,27.2,56,4.8,0.0,27.1767,78.0081),("Ajmer","Rajasthan",5.5,24.9,59,5.1,0.0,26.4499,74.6399),("Amritsar","Punjab",11.3,20.4,67,6.3,0.0,31.6340,74.8723),
    ("Asansol","West Bengal",12.4,22.1,59,5.0,0.0,23.6739,86.9524),("Aurangabad","Maharashtra",12.0,30.4,66,6.1,0.0,19.8762,75.3433),("Bareilly","Uttar Pradesh",9.0,25.4,54,4.7,0.0,28.3670,79.4304),
    ("Belagavi","Karnataka",6.1,21.6,68,6.4,0.0,15.8497,74.4977),("Bhubaneswar","Odisha",8.8,31.0,67,6.3,0.0,20.2961,85.8245),("Bikaner","Rajasthan",6.5,19.0,57,5.0,0.0,28.0229,73.3119),
    ("Chandigarh","Chandigarh",10.3,18.6,77,7.4,0.0,30.7333,76.7794),("Coimbatore","Tamil Nadu",10.7,18.9,73,7.0,0.0,11.0168,76.9558),("Dehradun","Uttarakhand",7.1,28.3,70,6.6,0.0,30.3165,78.0322),
    ("Durg-Bhilai","Chhattisgarh",10.6,18.1,61,5.4,0.0,21.1904,81.2849),("Guwahati","Assam",9.6,20.3,62,5.6,0.0,26.1445,91.7362),("Gwalior","Madhya Pradesh",10.7,24.4,58,5.1,0.0,26.2183,78.1828),
    ("Hubballi-Dharwad","Karnataka",9.4,22.9,68,6.4,0.0,15.3647,75.1240),("Indore","Madhya Pradesh",21.7,31.3,65,6.0,0.0,22.7196,75.8577),("Jabalpur","Madhya Pradesh",12.7,18.7,58,5.1,0.0,23.1815,79.9864),
    ("Jalandhar","Punjab",8.7,20.3,67,6.3,0.0,31.3260,75.5762),("Jammu","Jammu & Kashmir",6.6,25.2,64,5.9,0.0,32.7266,74.8570),("Jamnagar","Gujarat",6.0,23.0,69,6.7,0.0,22.4707,70.0577),
    ("Jodhpur","Rajasthan",10.3,27.0,60,5.4,0.0,26.2389,73.0243),("Kochi","Kerala",21.2,19.5,75,7.3,0.0,9.9312,76.2673),("Kota","Rajasthan",10.0,30.6,60,5.4,0.0,25.2138,75.8648),
    ("Lucknow","Uttar Pradesh",29.0,25.4,56,4.8,0.0,26.8467,80.9462),("Ludhiana","Punjab",16.2,19.9,67,6.3,0.0,30.9009,75.8573),("Madurai","Tamil Nadu",14.7,16.8,73,7.0,0.0,9.9252,78.1198),
    ("Meerut","Uttar Pradesh",14.2,25.4,56,4.8,0.0,28.9845,77.7064),("Mysuru","Karnataka",10.6,25.7,68,6.4,0.0,12.2958,76.6394),("Nagpur","Maharashtra",25.0,23.2,66,6.1,0.0,21.1458,79.0882),
    ("Nashik","Maharashtra",14.9,28.3,66,6.1,0.0,19.9975,73.7898),("Patna","Bihar",20.5,22.0,52,4.5,0.0,25.5941,85.1376),("Prayagraj","Uttar Pradesh",12.2,25.4,56,4.8,0.0,25.4358,81.8463),
    ("Raipur","Chhattisgarh",11.2,27.4,61,5.4,0.0,21.2514,81.6296),("Rajkot","Gujarat",13.9,25.0,69,6.7,0.0,22.3039,70.8022),("Ranchi","Jharkhand",11.3,24.0,58,5.0,0.0,23.3441,85.3096),
    ("Srinagar","Jammu & Kashmir",12.7,20.5,64,5.9,0.0,34.0837,74.7973),("Surat","Gujarat",45.9,42.2,69,6.7,0.0,21.1702,72.8311),("Thiruvananthapuram","Kerala",16.9,18.5,75,7.3,0.0,8.5241,76.9366),
    ("Tiruchirappalli","Tamil Nadu",10.2,20.1,73,7.0,0.0,10.7905,78.7047),("Udaipur","Rajasthan",6.1,22.7,60,5.4,0.0,24.5854,73.7125),("Vadodara","Gujarat",18.2,26.4,69,6.7,0.0,22.3072,73.1812),("Varanasi","Uttar Pradesh",14.3,25.4,56,4.8,0.0,25.3176,82.9739),("Vijayawada","Andhra Pradesh",10.3,24.7,65,5.9,0.0,16.5062,80.6480),("Visakhapatnam","Andhra Pradesh",20.4,25.7,65,5.9,0.0,17.6868,83.2185),
]


def default_cities() -> pd.DataFrame:
    columns = ["city", "state", "population_2011_lakh", "growth_pct", "internet_pct", "income_index", "competitor_stores", "lat", "lon"]
    city = pd.DataFrame(CITY_BASELINE, columns=columns)
    city["population_2026_lakh"] = city.population_2011_lakh * (1 + city.growth_pct / 100) ** 1.5
    city["households_lakh"] = city.population_2026_lakh / 4.2
    city["density_proxy"] = city.population_2026_lakh / (1 + city.growth_pct / 10)
    city["logistics_cost_index"] = 100 - 2.4 * city.density_proxy.clip(upper=25) + city.growth_pct * .20
    return city


def run_model(cities: pd.DataFrame, basket: float = 420, monthly_orders: float = 2.5, adoption: float = .12, national_gmv_cr: float = 70000, som_share: float = .12) -> ModelOutput:
    c = cities.copy()
    c["bottom_up_tam_cr"] = c.households_lakh * 100000 * basket * monthly_orders * 12 * adoption / 1e7
    urban_weight = c.population_2026_lakh * (c.internet_pct / 100)
    c["top_down_tam_cr"] = national_gmv_cr * urban_weight / urban_weight.sum()
    c["tam_crosscheck_gap_pct"] = 100 * (c.bottom_up_tam_cr - c.top_down_tam_cr) / c.top_down_tam_cr
    positive = ["income_index", "density_proxy", "internet_pct", "bottom_up_tam_cr"]
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(c[positive]), columns=positive, index=c.index)
    competition = 1 - MinMaxScaler().fit_transform(c[["competitor_stores"]]).ravel() if c.competitor_stores.nunique() > 1 else np.ones(len(c))
    logistics = 1 - MinMaxScaler().fit_transform(c[["logistics_cost_index"]]).ravel()
    c["attractiveness_score"] = 100 * (.25*scaled.income_index + .20*scaled.density_proxy + .20*scaled.internet_pct + .20*scaled.bottom_up_tam_cr + .10*competition + .05*logistics)
    c["som_cr"] = c.bottom_up_tam_cr * som_share
    c["rank"] = c.attractiveness_score.rank(ascending=False, method="min").astype(int)
    features = scaled[["income_index", "density_proxy", "internet_pct", "bottom_up_tam_cr"]]
    c["city_cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(features) + 1
    assumptions = {"basket": basket, "monthly_orders": monthly_orders, "adoption": adoption, "national_gmv_cr": national_gmv_cr, "som_share": som_share}
    return ModelOutput(c.sort_values("rank"), assumptions)


def save_model_to_sqlite(output: ModelOutput, path: str = "market_size.db") -> None:
    with sqlite3.connect(path) as con:
        output.cities.to_sql("city_expansion_scores", con, if_exists="replace", index=False)
        pd.DataFrame([output.assumptions]).to_sql("model_assumptions", con, if_exists="replace", index=False)
