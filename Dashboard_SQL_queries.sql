-- ===================================================================
-- DATABRICKS DASHBOARD SQL QUERIES
-- Flight Delay Prediction Analytics
-- ===================================================================
-- 
-- Table: default.flight_delay_predictions
-- 
-- Use these queries to create visualizations in your Databricks Dashboard
-- 
-- Dashboard Creation:
--   1. Go to Workspace → Create → Dashboard
--   2. Add visualizations (click + Visualization)
--   3. Paste queries below and configure chart types
-- ===================================================================

-- ===================================================================
-- 1. OVERVIEW METRICS
-- ===================================================================

-- Total Flights & High Risk Count
SELECT 
    COUNT(*) as total_flights,
    SUM(CASE WHEN delay_risk_in_flight = 'High' THEN 1 ELSE 0 END) as high_risk_flights,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_delay_probability,
    COUNT(DISTINCT airline_code) as total_airlines,
    COUNT(DISTINCT route) as total_routes
FROM default.flight_delay_predictions
WHERE DATE(prediction_timestamp) = CURRENT_DATE();

-- ===================================================================
-- 2. HIGH RISK FLIGHTS (Real-time Alert List)
-- ===================================================================

-- High Risk Flights - In-Flight Model
SELECT 
    airline_name,
    CONCAT(airline_code, fl_number) as flight_number,
    route,
    flight_date,
    probability_delay_in_pct as delay_probability,
    delay_risk_in_flight as risk_level,
    dep_delay as departure_delay_mins,
    prediction_timestamp
FROM default.flight_delay_predictions
WHERE delay_risk_in_flight = 'High'
ORDER BY probability_delay_in_pct DESC
LIMIT 20;

-- High Risk Flights - Pre-Departure Model (for scheduling)
SELECT 
    airline_name,
    CONCAT(airline_code, fl_number) as flight_number,
    route,
    flight_date,
    probability_delay_pre_pct as delay_probability,
    delay_risk_pre_dep as risk_level,
    prediction_timestamp
FROM default.flight_delay_predictions
WHERE delay_risk_pre_dep = 'High'
AND flight_date >= CURRENT_DATE()
ORDER BY probability_delay_pre_pct DESC
LIMIT 20;

-- ===================================================================
-- 3. AIRLINE PERFORMANCE
-- ===================================================================

-- Airline Delay Risk Ranking
SELECT 
    airline_name,
    COUNT(*) as total_flights,
    ROUND(AVG(probability_delay_pre_pct), 2) as avg_pre_delay_prob,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_in_delay_prob,
    SUM(CASE WHEN delay_risk_in_flight = 'High' THEN 1 ELSE 0 END) as high_risk_count,
    ROUND(SUM(CASE WHEN delay_risk_in_flight = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as high_risk_pct
FROM default.flight_delay_predictions
GROUP BY airline_name
HAVING COUNT(*) >= 3
ORDER BY avg_in_delay_prob DESC;

-- Airline Risk Distribution (for Pie Chart)
SELECT 
    airline_name,
    delay_risk_in_flight as risk_level,
    COUNT(*) as flight_count
FROM default.flight_delay_predictions
GROUP BY airline_name, delay_risk_in_flight
ORDER BY airline_name, risk_level;

-- ===================================================================
-- 4. ROUTE ANALYSIS
-- ===================================================================

-- Top 20 Routes by Delay Risk
SELECT 
    route,
    origin_airport_code,
    destination_airport_code,
    COUNT(*) as flight_count,
    ROUND(AVG(probability_delay_pre_pct), 2) as avg_pre_delay_prob,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_in_delay_prob,
    ROUND(AVG(dep_delay), 1) as avg_departure_delay_mins
FROM default.flight_delay_predictions
GROUP BY route, origin_airport_code, destination_airport_code
HAVING COUNT(*) >= 2
ORDER BY avg_in_delay_prob DESC
LIMIT 20;

-- Route Risk Heatmap Data
SELECT 
    origin_airport_code,
    destination_airport_code,
    COUNT(*) as flight_count,
    ROUND(AVG(probability_delay_in_pct), 1) as avg_delay_prob
FROM default.flight_delay_predictions
GROUP BY origin_airport_code, destination_airport_code
HAVING COUNT(*) >= 1;

-- ===================================================================
-- 5. TIME-BASED TRENDS
-- ===================================================================

-- Daily Delay Trends (Last 30 Days)
SELECT 
    DATE(flight_date) as date,
    COUNT(*) as total_flights,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_delay_prob,
    SUM(CASE WHEN delay_risk_in_flight = 'High' THEN 1 ELSE 0 END) as high_risk_flights
FROM default.flight_delay_predictions
WHERE flight_date >= DATE_SUB(CURRENT_DATE(), 30)
GROUP BY DATE(flight_date)
ORDER BY date;

-- Hourly Prediction Pattern (When predictions were made)
SELECT 
    HOUR(prediction_timestamp) as hour_of_day,
    COUNT(*) as prediction_count,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_delay_prob
FROM default.flight_delay_predictions
GROUP BY HOUR(prediction_timestamp)
ORDER BY hour_of_day;

-- ===================================================================
-- 6. MODEL COMPARISON
-- ===================================================================

-- Pre-Departure vs In-Flight Model Comparison
SELECT 
    'Pre-Departure' as model,
    ROUND(AVG(probability_delay_pre_pct), 2) as avg_delay_prob,
    SUM(CASE WHEN delay_risk_pre_dep = 'High' THEN 1 ELSE 0 END) as high_risk_count,
    SUM(CASE WHEN delay_risk_pre_dep = 'Medium' THEN 1 ELSE 0 END) as medium_risk_count,
    SUM(CASE WHEN delay_risk_pre_dep = 'Low' THEN 1 ELSE 0 END) as low_risk_count
FROM default.flight_delay_predictions

UNION ALL

SELECT 
    'In-Flight' as model,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_delay_prob,
    SUM(CASE WHEN delay_risk_in_flight = 'High' THEN 1 ELSE 0 END) as high_risk_count,
    SUM(CASE WHEN delay_risk_in_flight = 'Medium' THEN 1 ELSE 0 END) as medium_risk_count,
    SUM(CASE WHEN delay_risk_in_flight = 'Low' THEN 1 ELSE 0 END) as low_risk_count
FROM default.flight_delay_predictions;

-- Prediction Accuracy (when actual delays are known)
SELECT 
    delay_risk_in_flight as predicted_risk,
    actual_delayed,
    COUNT(*) as count
FROM default.flight_delay_predictions
WHERE actual_delayed IN ('Yes', 'No')
GROUP BY delay_risk_in_flight, actual_delayed
ORDER BY predicted_risk, actual_delayed;

-- ===================================================================
-- 7. DEPARTURE DELAY IMPACT
-- ===================================================================

-- Departure Delay Correlation with Arrival Delay Risk
SELECT 
    CASE 
        WHEN dep_delay <= 0 THEN 'On Time / Early'
        WHEN dep_delay <= 15 THEN '1-15 mins late'
        WHEN dep_delay <= 30 THEN '16-30 mins late'
        WHEN dep_delay <= 60 THEN '31-60 mins late'
        ELSE '60+ mins late'
    END as departure_status,
    COUNT(*) as flight_count,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_arrival_delay_prob,
    SUM(CASE WHEN delay_risk_in_flight = 'High' THEN 1 ELSE 0 END) as high_risk_arrivals
FROM default.flight_delay_predictions
GROUP BY CASE 
    WHEN dep_delay <= 0 THEN 'On Time / Early'
    WHEN dep_delay <= 15 THEN '1-15 mins late'
    WHEN dep_delay <= 30 THEN '16-30 mins late'
    WHEN dep_delay <= 60 THEN '31-60 mins late'
    ELSE '60+ mins late'
END
ORDER BY MIN(dep_delay);

-- ===================================================================
-- 8. RISK DISTRIBUTION
-- ===================================================================

-- Overall Risk Distribution (for Pie Chart)
SELECT 
    delay_risk_in_flight as risk_level,
    COUNT(*) as flight_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
FROM default.flight_delay_predictions
GROUP BY delay_risk_in_flight
ORDER BY FIELD(risk_level, 'High', 'Medium', 'Low');

-- Risk Distribution by Airline (for Stacked Bar Chart)
SELECT 
    airline_name,
    delay_risk_in_flight as risk_level,
    COUNT(*) as count
FROM default.flight_delay_predictions
GROUP BY airline_name, delay_risk_in_flight
HAVING SUM(COUNT(*)) OVER (PARTITION BY airline_name) >= 3
ORDER BY airline_name, FIELD(risk_level, 'High', 'Medium', 'Low');

-- ===================================================================
-- 9. SPECIFIC FLIGHT LOOKUP
-- ===================================================================

-- Search for Specific Flight (use parameters in dashboard)
-- Replace '{{airline_code}}' and '{{flight_number}}' with dashboard parameters
SELECT 
    prediction_timestamp,
    airline_name,
    CONCAT(airline_code, fl_number) as flight_number,
    route,
    flight_date,
    dep_delay as departure_delay_mins,
    probability_delay_pre_pct as pre_delay_prob,
    delay_risk_pre_dep as pre_risk,
    probability_delay_in_pct as in_delay_prob,
    delay_risk_in_flight as in_risk,
    actual_delayed
FROM default.flight_delay_predictions
WHERE airline_code = 'DL'  -- Replace with parameter: {{airline_code}}
AND fl_number = 1234        -- Replace with parameter: {{flight_number}}
ORDER BY prediction_timestamp DESC
LIMIT 10;

-- ===================================================================
-- 10. LATEST PREDICTIONS
-- ===================================================================

-- Most Recent Predictions (Last Hour)
SELECT 
    prediction_timestamp,
    airline_name,
    CONCAT(airline_code, fl_number) as flight_number,
    route,
    probability_delay_in_pct as delay_probability,
    delay_risk_in_flight as risk_level
FROM default.flight_delay_predictions
WHERE prediction_timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
ORDER BY prediction_timestamp DESC
LIMIT 50;

-- Latest Predictions by Risk Level
SELECT 
    delay_risk_in_flight as risk_level,
    COUNT(*) as recent_predictions,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_delay_prob
FROM default.flight_delay_predictions
WHERE prediction_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY delay_risk_in_flight
ORDER BY FIELD(risk_level, 'High', 'Medium', 'Low');

-- ===================================================================
-- 11. ENSEMBLE MODEL ANALYSIS (NEW)
-- ===================================================================

-- Model Agreement Analysis
SELECT 
    delay_risk_in_flight as risk_level,
    COUNT(*) as flight_count,
    ROUND(AVG(rf_in_pct), 2) as avg_rf_prediction,
    ROUND(AVG(gbt_in_pct), 2) as avg_gbt_prediction,
    ROUND(AVG(probability_delay_in_pct), 2) as avg_ensemble,
    ROUND(AVG(ABS(rf_in_pct - gbt_in_pct)), 2) as avg_model_disagreement
FROM default.flight_delay_predictions
GROUP BY delay_risk_in_flight
ORDER BY FIELD(risk_level, 'High', 'Medium', 'Low');

-- Flights with High Model Disagreement (uncertainty indicator)
SELECT 
    airline_name,
    CONCAT(airline_code, fl_number) as flight_number,
    route,
    rf_in_pct as rf_prediction,
    gbt_in_pct as gbt_prediction,
    probability_delay_in_pct as ensemble_prediction,
    ABS(rf_in_pct - gbt_in_pct) as model_disagreement,
    delay_risk_in_flight as risk_level
FROM default.flight_delay_predictions
WHERE ABS(rf_in_pct - gbt_in_pct) > 15  -- More than 15% disagreement
ORDER BY model_disagreement DESC
LIMIT 20;

-- Model Performance Comparison (Pre-Departure)
SELECT 
    'Random Forest' as model,
    ROUND(AVG(rf_pre_pct), 2) as avg_delay_prob,
    ROUND(STDDEV(rf_pre_pct), 2) as std_dev
FROM default.flight_delay_predictions

UNION ALL

SELECT 
    'GBT' as model,
    ROUND(AVG(gbt_pre_pct), 2) as avg_delay_prob,
    ROUND(STDDEV(gbt_pre_pct), 2) as std_dev
FROM default.flight_delay_predictions

UNION ALL

SELECT 
    'Ensemble' as model,
    ROUND(AVG(probability_delay_pre_pct), 2) as avg_delay_prob,
    ROUND(STDDEV(probability_delay_pre_pct), 2) as std_dev
FROM default.flight_delay_predictions;

-- ===================================================================
-- 12. ALTERNATIVE FLIGHT RECOMMENDATIONS (NEW)
-- ===================================================================

-- Top Alternative Recommendations
SELECT 
    CONCAT(original_airline, original_flight) as delayed_flight,
    CONCAT(origin, ' → ', destination) as route,
    original_delay_prob,
    CONCAT(alternative_airline, ' ', alternative_flight) as recommended_flight,
    alternative_delay_prob,
    alternative_risk_level,
    improvement_percentage,
    recommendation_rank
FROM default.alternative_flight_recommendations
WHERE recommendation_rank <= 3
ORDER BY improvement_percentage DESC
LIMIT 20;

-- Alternative Recommendations Summary by Route
SELECT 
    CONCAT(origin, ' → ', destination) as route,
    COUNT(DISTINCT CONCAT(original_airline, original_flight)) as delayed_flights,
    COUNT(*) as total_alternatives,
    ROUND(AVG(improvement_percentage), 1) as avg_improvement_pct,
    ROUND(AVG(original_delay_prob), 1) as avg_original_delay,
    ROUND(AVG(alternative_delay_prob), 1) as avg_alternative_delay
FROM default.alternative_flight_recommendations
GROUP BY origin, destination
ORDER BY delayed_flights DESC;

-- Best Alternative Airlines (by improvement)
SELECT 
    alternative_airline,
    COUNT(*) as times_recommended,
    ROUND(AVG(improvement_percentage), 1) as avg_improvement,
    ROUND(AVG(alternative_delay_prob), 1) as avg_delay_prob
FROM default.alternative_flight_recommendations
WHERE recommendation_rank = 1
GROUP BY alternative_airline
HAVING COUNT(*) >= 2
ORDER BY avg_improvement DESC;

-- Flights Needing Alternatives (No Good Options)
SELECT 
    original_airline,
    original_flight,
    CONCAT(origin, ' → ', destination) as route,
    original_delay_prob,
    flight_date
FROM default.flight_delay_predictions fp
LEFT JOIN default.alternative_flight_recommendations afr
    ON fp.airline_code = afr.original_airline
    AND fp.fl_number = afr.original_flight
    AND fp.flight_date = afr.flight_date
WHERE fp.delay_risk_in_flight = 'High'
    AND afr.original_airline IS NULL
ORDER BY original_delay_prob DESC;

-- Alternative Recommendation Details (for customer service)
SELECT 
    CONCAT(original_airline, original_flight) as your_flight,
    flight_date,
    CONCAT(origin, ' → ', destination) as route,
    original_delay_prob as your_delay_risk_pct,
    recommendation_rank as option_number,
    CONCAT(alternative_airline, ' Flight ', alternative_flight) as alternative,
    alternative_delay_prob as alternative_delay_risk_pct,
    alternative_risk_level,
    improvement_percentage as you_save_pct,
    CASE 
        WHEN improvement_percentage >= 30 THEN 'Highly Recommended'
        WHEN improvement_percentage >= 20 THEN 'Recommended'
        ELSE 'Consider'
    END as recommendation_level
FROM default.alternative_flight_recommendations
WHERE recommendation_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY flight_date, your_flight, option_number;

-- ===================================================================
-- 13. COMBINED PREDICTIONS + ALTERNATIVES VIEW
-- ===================================================================

-- Comprehensive Flight Status with Alternatives
SELECT 
    fp.airline_name,
    CONCAT(fp.airline_code, fp.fl_number) as flight,
    fp.route,
    fp.flight_date,
    fp.probability_delay_in_pct as delay_prob,
    fp.delay_risk_in_flight as risk,
    CASE 
        WHEN afr.original_airline IS NOT NULL THEN 'Yes'
        ELSE 'No'
    END as has_alternatives,
    COUNT(afr.recommendation_rank) as num_alternatives,
    MIN(afr.alternative_delay_prob) as best_alternative_prob
FROM default.flight_delay_predictions fp
LEFT JOIN default.alternative_flight_recommendations afr
    ON fp.airline_code = afr.original_airline
    AND fp.fl_number = afr.original_flight
    AND fp.flight_date = afr.flight_date
GROUP BY 
    fp.airline_name, fp.airline_code, fp.fl_number, 
    fp.route, fp.flight_date, fp.probability_delay_in_pct,
    fp.delay_risk_in_flight, afr.original_airline
ORDER BY fp.probability_delay_in_pct DESC;

-- ===================================================================
-- RECOMMENDED DASHBOARD LAYOUT
-- ===================================================================
-- 
-- ROW 1: Key Metrics (Cards)
--   - Total Flights Today
--   - High Risk Flight Count
--   - Average Delay Probability
--   - Total Airlines Monitored
--
-- ROW 2: High Risk Alerts (Table)
--   - Query: "High Risk Flights - In-Flight Model"
--   - Auto-refresh: Every 5 minutes
--
-- ROW 3: Airline & Route Analysis (Side by Side)
--   - Left: Airline Delay Risk Ranking (Bar Chart)
--   - Right: Top Routes by Delay Risk (Bar Chart)
--
-- ROW 4: Trends (Time Series)
--   - Daily Delay Trends (Line Chart)
--
-- ROW 5: Risk Distribution (Pie Charts)
--   - Left: Overall Risk Distribution
--   - Right: Model Comparison
--
-- ROW 6: Departure Impact (Bar Chart)
--   - Departure Delay Correlation with Arrival Risk
--
-- ===================================================================
-- DASHBOARD TIPS
-- ===================================================================
--
-- 1. Add Parameters:
--    - Date range filter
--    - Airline selector
--    - Risk level filter
--
-- 2. Set Auto-Refresh:
--    - High priority queries: 5-15 minutes
--    - Overview metrics: 15-30 minutes
--
-- 3. Visualizations:
--    - Use red/yellow/green colors for risk levels
--    - Show probability as percentage (0-100%)
--    - Use tables for detailed flight info
--    - Use charts for trends and comparisons
--
-- 4. Alerts:
--    - Set up email alerts for high risk flights
--    - Configure thresholds (e.g., >70% delay probability)
--
-- ===================================================================