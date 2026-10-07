-- PostgreSQL / DBeaver examples for model validation
-- Assume table: credit_risk_scored
-- Columns include default_12m, predicted_pd, model_version, scoring_date, sector_risk

-- 1. Basic portfolio / default-rate check
SELECT
    COUNT(*) AS n_obs,
    AVG(default_12m::numeric) AS observed_default_rate,
    AVG(predicted_pd) AS average_predicted_pd
FROM credit_risk_scored;

-- 2. Calibration by PD decile
WITH scored AS (
    SELECT
        default_12m,
        predicted_pd,
        NTILE(10) OVER (ORDER BY predicted_pd) AS pd_decile
    FROM credit_risk_scored
)
SELECT
    pd_decile,
    COUNT(*) AS n_obs,
    AVG(predicted_pd) AS avg_predicted_pd,
    AVG(default_12m::numeric) AS observed_default_rate
FROM scored
GROUP BY pd_decile
ORDER BY pd_decile;

-- 3. Compare versions / monitoring periods
SELECT
    model_version,
    DATE_TRUNC('month', scoring_date) AS month,
    COUNT(*) AS n_obs,
    AVG(predicted_pd) AS avg_pd,
    AVG(default_12m::numeric) AS observed_default_rate
FROM credit_risk_scored
GROUP BY model_version, DATE_TRUNC('month', scoring_date)
ORDER BY month, model_version;

-- 4. Sector-level diagnostics
SELECT
    sector_risk,
    COUNT(*) AS n_obs,
    AVG(predicted_pd) AS avg_pd,
    AVG(default_12m::numeric) AS observed_default_rate
FROM credit_risk_scored
GROUP BY sector_risk
HAVING COUNT(*) >= 30
ORDER BY sector_risk;

-- 5. Data-quality checks
SELECT
    SUM(CASE WHEN predicted_pd IS NULL THEN 1 ELSE 0 END) AS null_pd,
    SUM(CASE WHEN predicted_pd < 0 OR predicted_pd > 1 THEN 1 ELSE 0 END) AS invalid_pd,
    SUM(CASE WHEN default_12m NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_target
FROM credit_risk_scored;
