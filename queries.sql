-- Connect Tableau or Power BI to market_size.db and use this decision-ready view.
CREATE VIEW ranked_city_recommendations AS
SELECT city, state, rank, attractiveness_score, bottom_up_tam_cr, top_down_tam_cr,
       tam_crosscheck_gap_pct, som_cr, competitor_stores, city_cluster
FROM city_expansion_scores
ORDER BY rank;
