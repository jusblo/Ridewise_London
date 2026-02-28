# Data Dictionary - RideWise Customer Analytics

> **Version:** 1.0  
> **Last Updated:** February 2026  
> **Data Period:** 12 months (2025)  
> **Geographic Coverage:** 5 European cities (London, Berlin, Amsterdam, Barcelona, Milan)

## Overview

This data dictionary provides comprehensive documentation for the RideWise Customer Segmentation & Churn Prediction dataset. The dataset contains synthetic but realistic mobility data representing 200,000+ customers and 800,000+ trips across European markets.

---

## Table of Contents

1. [Dataset Summary](#dataset-summary)
2. [Data Tables](#data-tables)
3. [Data Quality Specifications](#data-quality-specifications)
4. [Business Rules & Constraints](#business-rules--constraints)
5. [Data Lineage](#data-lineage)
6. [Usage Guidelines](#usage-guidelines)

---

## Dataset Summary

| Metric               | Value                                                    |
| -------------------- | -------------------------------------------------------- |
| **Total Customers**  | 200,000+                                                 |
| **Total Trips**      | 800,000+                                                 |
| **Time Period**      | 12 months (Jan 2025 - Dec 2025)                          |
| **Cities**           | 5 (London, Berlin, Amsterdam, Barcelona, Milan)          |
| **Data Format**      | CSV (UTF-8 encoded)                                      |
| **Storage Location** | `data/raw/` (immutable), `data/processed/` (transformed) |
| **Update Frequency** | Static (synthetic dataset for training)                  |

---

## Data Tables

### 1. Riders Table

**File:** `data/raw/riders.csv`  
**Description:** Customer profile and demographic information  
**Primary Key:** `user_id`  
**Row Count:** ~200,000

| Column Name        | Data Type | Description                     | Valid Values                                | Nullable | Business Logic                  |
| ------------------ | --------- | ------------------------------- | ------------------------------------------- | -------- | ------------------------------- |
| `user_id`          | String    | Unique customer identifier      | Format: `USR_XXXXXX`                        | No       | Primary key, immutable          |
| `signup_date`      | Date      | Account registration date       | YYYY-MM-DD, 2024-01-01 to 2025-12-31        | No       | Must be <= current date         |
| `loyalty_status`   | String    | Customer tier membership        | Bronze, Silver, Gold, Platinum              | No       | Determines pricing & benefits   |
| `age`              | Integer   | Customer age in years           | 18-75                                       | No       | Legal age requirement: ≥18      |
| `city`             | String    | Primary operational city        | London, Berlin, Amsterdam, Barcelona, Milan | No       | Determines service availability |
| `avg_rating_given` | Float     | Average rating given to drivers | 1.0-5.0                                     | Yes      | Null if no trips completed      |
| `churn_prob`       | Float     | Estimated churn probability     | 0.0-1.0                                     | Yes      | Model-generated, for validation |
| `referred_by`      | String    | Referrer's user_id              | Valid `user_id` or NULL                     | Yes      | Referral program tracking       |

**Sample Data:**

```csv
user_id,signup_date,loyalty_status,age,city,avg_rating_given,churn_prob,referred_by
USR_000001,2025-01-15,Gold,34,London,4.7,0.12,USR_000045
USR_000002,2025-02-20,Bronze,28,Berlin,4.2,0.45,NULL
```

---

### 2. Trips Table

**File:** `data/raw/trips.csv`  
**Description:** Detailed transaction records for all completed rides  
**Primary Key:** `trip_id`  
**Foreign Keys:** `user_id` → riders.user_id, `driver_id` → drivers.driver_id  
**Row Count:** ~800,000

| Column Name        | Data Type | Description                 | Valid Values                                | Nullable | Business Logic             |
| ------------------ | --------- | --------------------------- | ------------------------------------------- | -------- | -------------------------- |
| `trip_id`          | String    | Unique trip identifier      | Format: `TRP_XXXXXXXX`                      | No       | Primary key, immutable     |
| `user_id`          | String    | Customer identifier         | Valid `user_id` from riders                 | No       | Foreign key constraint     |
| `driver_id`        | String    | Driver identifier           | Valid `driver_id` from drivers              | No       | Foreign key constraint     |
| `fare`             | Float     | Base trip cost (EUR)        | 3.50-150.00                                 | No       | Excludes surge & tip       |
| `surge_multiplier` | Float     | Demand-based pricing factor | 1.0-3.0                                     | No       | 1.0 = no surge             |
| `tip`              | Float     | Gratuity amount (EUR)       | 0.00-50.00                                  | Yes      | Optional, 0 if NULL        |
| `payment_type`     | String    | Payment method              | Card, Cash, Wallet, Corporate               | No       | Determines settlement flow |
| `pickup_time`      | Timestamp | Trip start datetime         | YYYY-MM-DD HH:MM:SS                         | No       | Must be < dropoff_time     |
| `dropoff_time`     | Timestamp | Trip end datetime           | YYYY-MM-DD HH:MM:SS                         | No       | Must be > pickup_time      |
| `pickup_lat`       | Float     | Pickup latitude             | -90.0 to 90.0                               | No       | Geospatial coordinate      |
| `pickup_lng`       | Float     | Pickup longitude            | -180.0 to 180.0                             | No       | Geospatial coordinate      |
| `dropoff_lat`      | Float     | Dropoff latitude            | -90.0 to 90.0                               | No       | Geospatial coordinate      |
| `dropoff_lng`      | Float     | Dropoff longitude           | -180.0 to 180.0                             | No       | Geospatial coordinate      |
| `weather`          | String    | Weather condition           | Sunny, Rainy, Cloudy, Snowy, Foggy          | Yes      | External data integration  |
| `city`             | String    | Trip city                   | London, Berlin, Amsterdam, Barcelona, Milan | No       | Must match user's city     |
| `loyalty_status`   | String    | User tier at trip time      | Bronze, Silver, Gold, Platinum              | No       | Snapshot for pricing       |

**Derived Metrics:**

- **Trip Duration:** `dropoff_time - pickup_time` (minutes)
- **Total Revenue:** `fare * surge_multiplier + tip` (EUR)
- **Distance:** Calculated using Haversine formula on lat/lng coordinates

---

### 3. Drivers Table

**File:** `data/raw/drivers.csv`  
**Description:** Driver profiles and performance metrics  
**Primary Key:** `driver_id`  
**Row Count:** ~50,000

| Column Name       | Data Type | Description               | Valid Values                                | Nullable | Business Logic             |
| ----------------- | --------- | ------------------------- | ------------------------------------------- | -------- | -------------------------- |
| `driver_id`       | String    | Unique driver identifier  | Format: `DRV_XXXXXX`                        | No       | Primary key, immutable     |
| `rating`          | Float     | Average customer rating   | 1.0-5.0                                     | Yes      | NULL if no completed trips |
| `vehicle_type`    | String    | Vehicle category          | Economy, Comfort, Premium, XL               | No       | Determines pricing tier    |
| `signup_date`     | Date      | Driver onboarding date    | YYYY-MM-DD                                  | No       | Must be <= current date    |
| `last_active`     | Date      | Most recent activity date | YYYY-MM-DD                                  | Yes      | For churn detection        |
| `city`            | String    | Primary operating city    | London, Berlin, Amsterdam, Barcelona, Milan | No       | Service area constraint    |
| `acceptance_rate` | Float     | % of requests accepted    | 0.0-1.0 (0-100%)                            | Yes      | Performance KPI            |

**Business Rules:**

- Drivers with `rating < 4.0` flagged for quality review
- `acceptance_rate < 0.70` triggers performance warning
- Inactive drivers: `last_active > 30 days ago`

---

### 4. Sessions Table

**File:** `data/raw/sessions.csv`  
**Description:** App engagement and user behavior tracking  
**Primary Key:** `session_id`  
**Foreign Key:** `rider_id` → riders.user_id  
**Row Count:** ~1,500,000

| Column Name      | Data Type | Description                | Valid Values                                | Nullable | Business Logic         |
| ---------------- | --------- | -------------------------- | ------------------------------------------- | -------- | ---------------------- |
| `session_id`     | String    | Unique session identifier  | Format: `SES_XXXXXXXXXX`                    | No       | Primary key, immutable |
| `rider_id`       | String    | Customer identifier        | Valid `user_id` from riders                 | No       | Foreign key constraint |
| `session_time`   | Timestamp | Session start datetime     | YYYY-MM-DD HH:MM:SS                         | No       | UTC timezone           |
| `time_on_app`    | Float     | Session duration (minutes) | 0.1-120.0                                   | No       | Auto-logout at 120 min |
| `pages_visited`  | Integer   | Number of screens viewed   | 1-50                                        | No       | Engagement metric      |
| `converted`      | Boolean   | Trip booked in session     | 0 (False), 1 (True)                         | No       | Conversion tracking    |
| `city`           | String    | Session origin city        | London, Berlin, Amsterdam, Barcelona, Milan | No       | Geolocation-based      |
| `loyalty_status` | String    | User tier during session   | Bronze, Silver, Gold, Platinum              | No       | Snapshot for analytics |

**Conversion Metrics:**

- **Conversion Rate:** `SUM(converted) / COUNT(session_id)`
- **Avg. Session Duration:** `AVG(time_on_app)` by loyalty tier
- **Engagement Score:** `pages_visited * time_on_app`

---

### 5. Promotions Table

**File:** `data/raw/promotions.csv`  
**Description:** Marketing campaigns and A/B testing data  
**Primary Key:** `promo_id`  
**Row Count:** ~500

| Column Name       | Data Type | Description                 | Valid Values                                     | Nullable | Business Logic              |
| ----------------- | --------- | --------------------------- | ------------------------------------------------ | -------- | --------------------------- |
| `promo_id`        | String    | Unique promotion identifier | Format: `PROMO_XXXX`                             | No       | Primary key, immutable      |
| `promo_name`      | String    | Campaign name               | Free text (max 100 chars)                        | No       | Human-readable label        |
| `promo_type`      | String    | Discount mechanism          | Percentage, Fixed_Amount, BOGO, Free_Ride        | No       | Determines calculation      |
| `promo_value`     | Float     | Discount value              | 0.05-50.00                                       | No       | % (0.05-0.50) or EUR (5-50) |
| `start_date`      | Date      | Campaign start date         | YYYY-MM-DD                                       | No       | Must be < end_date          |
| `end_date`        | Date      | Campaign end date           | YYYY-MM-DD                                       | No       | Must be > start_date        |
| `target_segment`  | String    | Target customer group       | Bronze, Silver, Gold, Platinum, All, New_Users   | No       | Eligibility filter          |
| `city_scope`      | String    | Geographic scope            | City name or "All"                               | No       | Service area constraint     |
| `ab_test_groups`  | String    | Experiment groups           | Control, Variant_A, Variant_B                    | Yes      | NULL if not A/B test        |
| `test_allocation` | Float     | % users in test group       | 0.0-1.0                                          | Yes      | NULL if not A/B test        |
| `success_metric`  | String    | Primary KPI                 | Conversion_Rate, Revenue, Trips_Count, Retention | No       | Evaluation criterion        |

**Campaign Performance Metrics:**

- **Redemption Rate:** Users who used promo / Total eligible users
- **Incremental Revenue:** Revenue during campaign vs. baseline
- **ROI:** (Revenue - Discount Cost) / Discount Cost

---

## Data Quality Specifications

### Completeness Requirements

| Table          | Critical Columns (0% NULL)                                   | Optional Columns (NULL Allowed)           |
| -------------- | ------------------------------------------------------------ | ----------------------------------------- |
| **Riders**     | user_id, signup_date, loyalty_status, age, city              | avg_rating_given, churn_prob, referred_by |
| **Trips**      | trip_id, user_id, driver_id, fare, pickup_time, dropoff_time | tip, weather                              |
| **Drivers**    | driver_id, vehicle_type, signup_date, city                   | rating, last_active, acceptance_rate      |
| **Sessions**   | session_id, rider_id, session_time, time_on_app, converted   | None                                      |
| **Promotions** | promo_id, promo_name, promo_type, start_date, end_date       | ab_test_groups, test_allocation           |

### Data Validation Rules

1. **Referential Integrity:**
   - All `user_id` in trips must exist in riders table
   - All `driver_id` in trips must exist in drivers table
   - All `rider_id` in sessions must exist in riders table

2. **Temporal Consistency:**
   - `signup_date` ≤ first trip `pickup_time`
   - `pickup_time` < `dropoff_time` (all trips)
   - `start_date` < `end_date` (all promotions)

3. **Logical Constraints:**
   - `age` ≥ 18 (legal requirement)
   - `fare` > 0 (no free rides except promotions)
   - `surge_multiplier` ≥ 1.0 (cannot be discount)
   - `rating` between 1.0 and 5.0

4. **Uniqueness:**
   - No duplicate `user_id`, `trip_id`, `driver_id`, `session_id`, `promo_id`

---

## Business Rules & Constraints

### Loyalty Tier Definitions

| Tier         | Criteria          | Benefits                                |
| ------------ | ----------------- | --------------------------------------- |
| **Bronze**   | 0-10 trips/month  | Standard pricing                        |
| **Silver**   | 11-25 trips/month | 5% discount, priority support           |
| **Gold**     | 26-50 trips/month | 10% discount, free cancellations        |
| **Platinum** | 51+ trips/month   | 15% discount, dedicated account manager |

### Churn Definition

A customer is considered **churned** if:

- No trips in the last 60 days
- No app sessions in the last 45 days
- Account status = "Inactive"

### Pricing Logic

```
Total Trip Cost = (Base Fare × Surge Multiplier) + Tip - Promo Discount
```

---

## Data Lineage

```
Raw Data Sources → Data Validation → Feature Engineering → Model Training
     ↓                    ↓                  ↓                  ↓
  CSV Files         Quality Checks      RFM Analysis      Segmentation
  (Immutable)       (Logging)           Behavioral        Churn Prediction
                                        Metrics
```

### Processing Pipeline

1. **Ingestion:** Raw CSV files loaded into SQLite database
2. **Validation:** Schema checks, NULL validation, referential integrity
3. **Transformation:** Feature engineering (RFM, temporal, behavioral)
4. **Storage:** Processed data saved to `data/processed/`
5. **Modeling:** ML pipelines consume processed data

---

## Usage Guidelines

### For Data Scientists

1. **Always use `data/raw/` for source data** - Never modify raw files
2. **Save transformed data to `data/processed/`** - Use versioned filenames
3. **Document feature engineering** - Add comments in transformation scripts
4. **Validate assumptions** - Check data distributions before modeling
5. **Handle missing values appropriately** - Document imputation strategies

### For Analysts

1. **Use processed data for reporting** - Raw data may contain quality issues
2. **Filter by date ranges** - Ensure temporal consistency in analyses
3. **Segment by city** - Markets have different characteristics
4. **Account for seasonality** - Trip patterns vary by month/day

### Data Access Patterns

```python
# Example: Loading riders data
import pandas as pd

riders = pd.read_csv('data/raw/riders.csv', parse_dates=['signup_date'])
trips = pd.read_csv('data/raw/trips.csv', parse_dates=['pickup_time', 'dropoff_time'])

# Merge for customer-level analysis
customer_trips = trips.merge(riders, on='user_id', how='left')
```

---

## Contact & Support

For questions about data definitions, quality issues, or access requests:

- **Data Owner:** RideWise Analytics Team
- **Documentation:** See `docs/data_schema.md` for technical specifications
- **Issues:** Report data quality problems via GitHub Issues

---

**Document Version History:**

- v1.0 (Feb 2026): Initial data dictionary creation
# 🚀 FastAPI Churn Prediction Service

## Run the API

```bash
uvicorn src.api.main:app --reload