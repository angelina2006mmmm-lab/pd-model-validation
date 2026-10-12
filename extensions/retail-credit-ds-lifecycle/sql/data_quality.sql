-- PostgreSQL exploratory SQL (illustrative schema).
SELECT application_id, COUNT(*) FROM credit_applications GROUP BY 1 HAVING COUNT(*)>1;
SELECT application_month, COUNT(*) FROM macro_monthly GROUP BY 1 HAVING COUNT(*)>1;
SELECT COUNT(*) FILTER (WHERE monthly_income<=0 OR loan_amount<=0) AS bad_amounts,
 COUNT(*) FILTER (WHERE default_12m NOT IN (0,1)) AS invalid_target FROM credit_applications;
SELECT a.application_month,COUNT(*) AS applications,
 ROUND(AVG(a.default_12m::numeric),4) AS observed_default_rate,
 MAX(m.macro_stress) AS macro_stress
FROM credit_applications a LEFT JOIN macro_monthly m USING(application_month)
GROUP BY a.application_month ORDER BY a.application_month;
