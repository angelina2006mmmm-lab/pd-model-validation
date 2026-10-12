-- Illustrative PostgreSQL table: credit_applications, no production bank schema.
SELECT application_id, COUNT(*) FROM credit_applications GROUP BY 1 HAVING COUNT(*)>1;
SELECT date_trunc('month',application_date) AS vintage, product_type,
       COUNT(*) AS loan_count, ROUND(AVG(default_12m::numeric),4) AS default_rate
FROM credit_applications GROUP BY 1,2 ORDER BY 1,2;
SELECT COUNT(*) FILTER (WHERE monthly_income IS NULL OR monthly_income<=0) AS bad_income,
       COUNT(*) FILTER (WHERE dti<0 OR dti>1) AS bad_dti,
       COUNT(*) FILTER (WHERE default_12m NOT IN (0,1)) AS invalid_target
FROM credit_applications;
