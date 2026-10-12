-- PostgreSQL example; example schema ONLY. Do not run against a live bank database.
-- Input: credit_applications(application_id,application_date,loan_amount,default_12m,model_version,pd_score).
SELECT application_id,COUNT(*) AS copies
FROM credit_applications GROUP BY application_id HAVING COUNT(*)>1;

SELECT date_trunc('month',application_date) AS vintage,model_version,
       COUNT(*) AS applications,
       AVG(default_12m::numeric) AS observed_bad_rate,
       AVG(pd_score) AS predicted_pd,
       SUM(pd_score*loan_amount*0.45) AS hypothetical_expected_loss
FROM credit_applications
GROUP BY 1,2 ORDER BY 1,2;

-- Detect non-finite / invalid ranges, missing fields and mismatched scores.
SELECT COUNT(*) FILTER(WHERE pd_score IS NULL OR pd_score<0 OR pd_score>1) AS bad_scores,
       COUNT(*) FILTER(WHERE loan_amount<=0 OR loan_amount IS NULL) AS bad_amounts,
       COUNT(*) FILTER(WHERE model_version IS NULL) AS version_missing
FROM credit_applications;
