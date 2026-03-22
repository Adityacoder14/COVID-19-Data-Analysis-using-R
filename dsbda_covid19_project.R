# =============================================================================
#  DSBDA PROJECT — COVID-19 Data Analysis
#  Subject : Data Science & Big Data Analytics (DSBDA)
#  Language : R
#  Dataset  : COVID-19 (simulated realistic dataset — no download needed)
#  Practicals Covered:
#    1. Data Wrangling & Cleaning
#    2. Descriptive Statistics
#    3. Data Visualization
#    4. Linear & Logistic Regression
# =============================================================================

# ── Install packages if not already installed ─────────────────────────────────
packages <- c("dplyr", "tidyr", "ggplot2", "scales", "corrplot", "caret")
install_if_missing <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(install_if_missing)) install.packages(install_if_missing, dependencies = TRUE)

# ── Load libraries ────────────────────────────────────────────────────────────
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(corrplot)
library(caret)

cat("\n============================================================\n")
cat("   DSBDA PROJECT : COVID-19 Data Analysis in R\n")
cat("============================================================\n\n")


# =============================================================================
# SECTION 0 : CREATE DATASET
# Description: We simulate a realistic COVID-19 country-level dataset so the
#              project runs without any external file download.
# =============================================================================

set.seed(42)
n <- 200   # 200 records (countries × time periods)

countries <- c("India","USA","Brazil","UK","Germany","France","Italy",
               "Russia","Spain","Argentina","Colombia","Mexico","South Africa",
               "Japan","South Korea","Australia","Canada","Turkey","Poland","Iran")

covid_raw <- data.frame(
  country         = sample(countries, n, replace = TRUE),
  date            = sample(seq(as.Date("2020-03-01"), as.Date("2022-12-01"), by="month"), n, replace=TRUE),
  total_cases     = sample(1000:5000000, n, replace = TRUE),
  total_deaths    = NA,
  total_recovered = NA,
  total_tests     = sample(50000:20000000, n, replace=TRUE),
  population      = sample(1000000:1400000000, n, replace=TRUE),
  gdp_per_capita  = sample(2000:65000, n, replace=TRUE),
  hospital_beds   = round(runif(n, 0.5, 14), 2),   # per 1000 people
  vaccination_pct = round(runif(n, 0, 95), 1),
  stringency_index= round(runif(n, 10, 100), 1),   # govt. response 0–100
  high_mortality  = NA,                              # target for logistic regression
  stringsAsFactors = FALSE
)

# Derive realistic deaths and recovered
covid_raw$total_deaths    <- round(covid_raw$total_cases * runif(n, 0.005, 0.04))
covid_raw$total_recovered <- round(covid_raw$total_cases * runif(n, 0.75, 0.95))

# Inject missing values (simulate real-world dirty data)
covid_raw$total_cases[sample(n, 10)]      <- NA
covid_raw$total_deaths[sample(n, 8)]      <- NA
covid_raw$vaccination_pct[sample(n, 12)]  <- NA
covid_raw$hospital_beds[sample(n, 6)]     <- NA
covid_raw$gdp_per_capita[sample(n, 5)]    <- NA

# Inject duplicate rows
covid_raw <- rbind(covid_raw, covid_raw[sample(n, 10), ])

cat(">> Dataset created with", nrow(covid_raw), "rows and", ncol(covid_raw), "columns.\n\n")


# =============================================================================
# PRACTICAL 1 : DATA WRANGLING & CLEANING
# Objectives :
#   a) Identify & remove duplicates
#   b) Handle missing values
#   c) Create derived / new features
#   d) Filter and sort data
#   e) Reshape data (wide ↔ long)
# =============================================================================

cat("------------------------------------------------------------\n")
cat(" PRACTICAL 1 : DATA WRANGLING & CLEANING\n")
cat("------------------------------------------------------------\n\n")

# ── 1a. Initial inspection ────────────────────────────────────────────────────
cat("-- Structure of raw dataset --\n")
str(covid_raw)

cat("\n-- First 6 rows --\n")
print(head(covid_raw))

cat("\n-- Missing values per column --\n")
missing_summary <- colSums(is.na(covid_raw))
print(missing_summary)

cat("\n-- Duplicate rows --\n")
cat("Number of duplicate rows:", sum(duplicated(covid_raw)), "\n")

# ── 1b. Remove duplicates ─────────────────────────────────────────────────────
covid_clean <- covid_raw[!duplicated(covid_raw), ]
cat("Rows after removing duplicates:", nrow(covid_clean), "\n")

# ── 1c. Handle missing values ─────────────────────────────────────────────────
# Numeric columns: replace NA with column median (robust to outliers)
numeric_cols <- c("total_cases","total_deaths","vaccination_pct",
                  "hospital_beds","gdp_per_capita")

for (col in numeric_cols) {
  med_val <- median(covid_clean[[col]], na.rm = TRUE)
  covid_clean[[col]][is.na(covid_clean[[col]])] <- med_val
  cat(sprintf("  NA in %-20s replaced with median = %.2f\n", col, med_val))
}

cat("\n-- Missing values after cleaning --\n")
print(colSums(is.na(covid_clean)))

# ── 1d. Feature engineering ───────────────────────────────────────────────────
covid_clean <- covid_clean %>%
  mutate(
    # Case fatality rate (%)
    cfr               = round((total_deaths / total_cases) * 100, 3),

    # Cases per million population
    cases_per_million = round((total_cases / population) * 1e6, 2),

    # Test positivity rate (%)
    positivity_rate   = round((total_cases / total_tests) * 100, 2),

    # Recovery rate (%)
    recovery_rate     = round((total_recovered / total_cases) * 100, 2),

    # Active cases
    active_cases      = total_cases - total_deaths - total_recovered,

    # Month and Year extracted from date
    year              = as.integer(format(date, "%Y")),
    month             = as.integer(format(date, "%m")),

    # Binary target: 1 if CFR > 2% (high mortality), 0 otherwise
    high_mortality    = ifelse(cfr > 2, 1, 0)
  )

cat("\n-- Derived features added --\n")
cat("  cfr, cases_per_million, positivity_rate, recovery_rate,\n")
cat("  active_cases, year, month, high_mortality\n")

# ── 1e. Filter & sort ─────────────────────────────────────────────────────────
# Countries with CFR > 1% sorted descending
high_cfr <- covid_clean %>%
  filter(cfr > 1) %>%
  arrange(desc(cfr)) %>%
  select(country, date, total_cases, total_deaths, cfr)

cat("\n-- Top 10 records with CFR > 1% --\n")
print(head(high_cfr, 10))

# ── 1f. Reshape (wide → long) ─────────────────────────────────────────────────
covid_long <- covid_clean %>%
  select(country, date, total_cases, total_deaths, total_recovered) %>%
  pivot_longer(cols = c(total_cases, total_deaths, total_recovered),
               names_to  = "metric",
               values_to = "value")

cat("\n-- Long format (first 6 rows) --\n")
print(head(covid_long))

cat("\n>> Practical 1 COMPLETE. Clean dataset has",
    nrow(covid_clean), "rows.\n\n")


# =============================================================================
# PRACTICAL 2 : DESCRIPTIVE STATISTICS
# Objectives :
#   a) Measures of central tendency (mean, median, mode)
#   b) Measures of dispersion (SD, variance, range, IQR)
#   c) Frequency distribution
#   d) Correlation matrix
#   e) Summary statistics by group
# =============================================================================

cat("------------------------------------------------------------\n")
cat(" PRACTICAL 2 : DESCRIPTIVE STATISTICS\n")
cat("------------------------------------------------------------\n\n")

# ── 2a. Summary of entire dataset ────────────────────────────────────────────
cat("-- Full summary statistics --\n")
print(summary(covid_clean %>% select(total_cases, total_deaths, cfr,
                                      vaccination_pct, hospital_beds,
                                      gdp_per_capita, positivity_rate)))

# ── 2b. Central tendency & dispersion (manual) ───────────────────────────────
cat("\n-- Detailed statistics for key numeric variables --\n")

stat_vars <- c("total_cases","total_deaths","cfr",
               "vaccination_pct","gdp_per_capita","positivity_rate")

# Mode helper
get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

stats_table <- data.frame(
  Variable  = stat_vars,
  Mean      = sapply(stat_vars, function(v) round(mean(covid_clean[[v]], na.rm=TRUE),  2)),
  Median    = sapply(stat_vars, function(v) round(median(covid_clean[[v]], na.rm=TRUE),2)),
  Mode      = sapply(stat_vars, function(v) round(get_mode(covid_clean[[v]]),           2)),
  Std_Dev   = sapply(stat_vars, function(v) round(sd(covid_clean[[v]], na.rm=TRUE),     2)),
  Variance  = sapply(stat_vars, function(v) round(var(covid_clean[[v]], na.rm=TRUE),    2)),
  Min       = sapply(stat_vars, function(v) round(min(covid_clean[[v]], na.rm=TRUE),    2)),
  Max       = sapply(stat_vars, function(v) round(max(covid_clean[[v]], na.rm=TRUE),    2)),
  IQR       = sapply(stat_vars, function(v) round(IQR(covid_clean[[v]], na.rm=TRUE),    2)),
  row.names = NULL
)
print(stats_table)

# ── 2c. Frequency distribution of high_mortality ─────────────────────────────
cat("\n-- Frequency distribution: high_mortality (0=Low, 1=High) --\n")
freq_table <- table(covid_clean$high_mortality)
rel_freq   <- prop.table(freq_table) * 100
print(data.frame(
  Category         = c("Low Mortality (CFR ≤ 2%)", "High Mortality (CFR > 2%)"),
  Frequency        = as.integer(freq_table),
  Relative_Freq_pct= round(as.numeric(rel_freq), 2)
))

# ── 2d. Correlation matrix ────────────────────────────────────────────────────
cat("\n-- Correlation matrix --\n")
corr_vars <- covid_clean %>%
  select(total_cases, total_deaths, cfr, vaccination_pct,
         hospital_beds, gdp_per_capita, stringency_index, positivity_rate)

cor_matrix <- round(cor(corr_vars, use = "complete.obs"), 3)
print(cor_matrix)

# ── 2e. Group-wise statistics ─────────────────────────────────────────────────
cat("\n-- Mean CFR and vaccination % by country (top 10) --\n")
country_stats <- covid_clean %>%
  group_by(country) %>%
  summarise(
    records        = n(),
    mean_cfr       = round(mean(cfr, na.rm=TRUE), 3),
    mean_vacc      = round(mean(vaccination_pct, na.rm=TRUE), 1),
    mean_cases_M   = round(mean(cases_per_million, na.rm=TRUE), 0),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_cfr))

print(head(country_stats, 10))

cat("\n>> Practical 2 COMPLETE.\n\n")


# =============================================================================
# PRACTICAL 3 : DATA VISUALIZATION
# Objectives :
#   a) Bar chart  — cases by country
#   b) Histogram  — distribution of CFR
#   c) Boxplot    — CFR by high/low mortality
#   d) Line chart — monthly trend
#   e) Scatter plot with regression line — vaccination vs CFR
#   f) Correlation heatmap
#   g) Pie chart  — mortality share
# =============================================================================

cat("------------------------------------------------------------\n")
cat(" PRACTICAL 3 : DATA VISUALIZATION\n")
cat("------------------------------------------------------------\n\n")
cat(">> Generating 7 plots...\n")

# ── 3a. Bar Chart — Average Cases per Million by Country ─────────────────────
p1 <- covid_clean %>%
  group_by(country) %>%
  summarise(avg_cases = mean(cases_per_million, na.rm=TRUE), .groups="drop") %>%
  arrange(desc(avg_cases)) %>%
  head(12) %>%
  ggplot(aes(x = reorder(country, avg_cases), y = avg_cases, fill = avg_cases)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#56B4E9", high = "#D55E00") +
  coord_flip() +
  scale_y_continuous(labels = comma) +
  labs(title    = "Practical 3a: Average COVID-19 Cases per Million by Country",
       subtitle = "Top 12 countries ranked by average cases per million population",
       x = "Country", y = "Cases per Million") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face="bold"))

print(p1)
cat("   Plot 1 (Bar Chart) displayed.\n")

# ── 3b. Histogram — Distribution of Case Fatality Rate ───────────────────────
p2 <- ggplot(covid_clean, aes(x = cfr)) +
  geom_histogram(binwidth = 0.3, fill = "#E69F00", color = "white", alpha = 0.85) +
  geom_vline(aes(xintercept = mean(cfr, na.rm=TRUE)),
             color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = mean(covid_clean$cfr)+0.3,
           y = 20, label = paste("Mean =", round(mean(covid_clean$cfr),2)),
           color = "red", size = 3.5) +
  labs(title    = "Practical 3b: Distribution of Case Fatality Rate (CFR)",
       subtitle = "Dashed red line = mean CFR",
       x = "CFR (%)", y = "Frequency") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face="bold"))

print(p2)
cat("   Plot 2 (Histogram) displayed.\n")

# ── 3c. Boxplot — CFR by Mortality Class ─────────────────────────────────────
p3 <- covid_clean %>%
  mutate(mortality_class = ifelse(high_mortality == 1, "High Mortality", "Low Mortality")) %>%
  ggplot(aes(x = mortality_class, y = cfr, fill = mortality_class)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 16, alpha = 0.8) +
  scale_fill_manual(values = c("High Mortality" = "#CC79A7",
                                "Low Mortality"  = "#009E73")) +
  labs(title    = "Practical 3c: Boxplot of CFR by Mortality Class",
       subtitle = "Spread and outliers of Case Fatality Rate",
       x = "", y = "CFR (%)", fill = "") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face="bold"), legend.position = "none")

print(p3)
cat("   Plot 3 (Boxplot) displayed.\n")

# ── 3d. Line Chart — Monthly Total Cases Trend ───────────────────────────────
monthly_trend <- covid_clean %>%
  mutate(month_date = as.Date(paste(year, month, "01", sep="-"))) %>%
  group_by(month_date) %>%
  summarise(total = sum(total_cases, na.rm=TRUE), .groups="drop") %>%
  arrange(month_date)

p4 <- ggplot(monthly_trend, aes(x = month_date, y = total)) +
  geom_line(color = "#0072B2", linewidth = 1.2) +
  geom_point(color = "#D55E00", size = 2) +
  scale_y_continuous(labels = comma) +
  scale_x_date(date_labels = "%b %Y", date_breaks = "3 months") +
  labs(title    = "Practical 3d: Monthly COVID-19 Cases Trend",
       subtitle = "Aggregated total cases across all countries per month",
       x = "Month", y = "Total Cases") +
  theme_minimal(base_size = 12) +
  theme(plot.title   = element_text(face="bold"),
        axis.text.x  = element_text(angle=45, hjust=1))

print(p4)
cat("   Plot 4 (Line Chart) displayed.\n")

# ── 3e. Scatter Plot — Vaccination % vs CFR ───────────────────────────────────
p5 <- ggplot(covid_clean, aes(x = vaccination_pct, y = cfr, color = country)) +
  geom_point(alpha = 0.6, size = 2, show.legend = FALSE) +
  geom_smooth(method = "lm", color = "black", se = TRUE, linewidth = 1) +
  labs(title    = "Practical 3e: Vaccination Rate vs Case Fatality Rate",
       subtitle = "Black line = linear regression fit; shaded area = 95% CI",
       x = "Vaccination Coverage (%)", y = "CFR (%)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face="bold"))

print(p5)
cat("   Plot 5 (Scatter + Regression) displayed.\n")

# ── 3f. Correlation Heatmap ───────────────────────────────────────────────────
cat("   Plot 6 (Correlation Heatmap) displayed.\n")
corrplot(cor_matrix,
         method   = "color",
         type     = "upper",
         tl.col   = "black",
         tl.cex   = 0.75,
         col      = colorRampPalette(c("#D55E00","white","#0072B2"))(200),
         addCoef.col = "black",
         number.cex  = 0.65,
         title    = "Practical 3f: Correlation Heatmap of COVID-19 Variables",
         mar      = c(0,0,2,0))

# ── 3g. Pie Chart — High vs Low Mortality Share ───────────────────────────────
pie_data <- covid_clean %>%
  count(high_mortality) %>%
  mutate(
    label   = ifelse(high_mortality == 1, "High Mortality", "Low Mortality"),
    percent = round(n / sum(n) * 100, 1),
    pct_lab = paste0(label, "\n", percent, "%")
  )

p7 <- ggplot(pie_data, aes(x="", y=n, fill=label)) +
  geom_bar(stat="identity", width=1, color="white") +
  coord_polar("y") +
  scale_fill_manual(values = c("High Mortality"="#CC79A7","Low Mortality"="#009E73")) +
  geom_text(aes(label = pct_lab), position = position_stack(vjust = 0.5), size=4.5) +
  labs(title = "Practical 3g: Proportion of High vs Low Mortality Records",
       fill  = "") +
  theme_void() +
  theme(plot.title    = element_text(face="bold", hjust=0.5),
        legend.position = "none")

print(p7)
cat("   Plot 7 (Pie Chart) displayed.\n")

cat("\n>> Practical 3 COMPLETE. All 7 plots generated.\n\n")


# =============================================================================
# PRACTICAL 4 : LINEAR & LOGISTIC REGRESSION
# Part A — Linear Regression  : Predict total_deaths from total_cases,
#                                vaccination_pct, hospital_beds, gdp_per_capita
# Part B — Logistic Regression: Predict high_mortality (0/1) using same features
# =============================================================================

cat("------------------------------------------------------------\n")
cat(" PRACTICAL 4 : REGRESSION ANALYSIS\n")
cat("------------------------------------------------------------\n\n")

# ── Prepare clean regression dataset ─────────────────────────────────────────
reg_data <- covid_clean %>%
  select(total_deaths, total_cases, vaccination_pct,
         hospital_beds, gdp_per_capita, stringency_index,
         positivity_rate, high_mortality) %>%
  drop_na()

cat("Regression dataset size:", nrow(reg_data), "rows\n\n")

# ── Train / Test split (80:20) ────────────────────────────────────────────────
set.seed(123)
train_idx   <- createDataPartition(reg_data$high_mortality, p=0.80, list=FALSE)
train_data  <- reg_data[ train_idx, ]
test_data   <- reg_data[-train_idx, ]
cat(sprintf("Train: %d rows  |  Test: %d rows\n\n", nrow(train_data), nrow(test_data)))


# ════════════════════════════════════════════════════════════════════════════
# PART A : LINEAR REGRESSION — Predicting total_deaths
# ════════════════════════════════════════════════════════════════════════════
cat("== PART A : LINEAR REGRESSION ==\n\n")
cat("Dependent variable  : total_deaths\n")
cat("Independent variables: total_cases, vaccination_pct, hospital_beds,\n")
cat("                       gdp_per_capita, stringency_index\n\n")

lm_model <- lm(total_deaths ~ total_cases + vaccination_pct +
                  hospital_beds + gdp_per_capita + stringency_index,
                data = train_data)

cat("-- Model Summary --\n")
print(summary(lm_model))

# Predictions on test set
lm_preds   <- predict(lm_model, newdata = test_data)
lm_actual  <- test_data$total_deaths

# Performance metrics
lm_mae     <- mean(abs(lm_preds - lm_actual))
lm_mse     <- mean((lm_preds - lm_actual)^2)
lm_rmse    <- sqrt(lm_mse)
ss_res     <- sum((lm_actual - lm_preds)^2)
ss_tot     <- sum((lm_actual - mean(lm_actual))^2)
lm_r2      <- 1 - ss_res/ss_tot

cat("\n-- Linear Regression Performance on Test Set --\n")
cat(sprintf("  MAE  (Mean Absolute Error)   : %.2f\n",  lm_mae))
cat(sprintf("  MSE  (Mean Squared Error)    : %.2f\n",  lm_mse))
cat(sprintf("  RMSE (Root MSE)              : %.2f\n",  lm_rmse))
cat(sprintf("  R²   (Coefficient of Det.)   : %.4f\n",  lm_r2))

# ── Residual plot ─────────────────────────────────────────────────────────────
residual_df <- data.frame(
  Fitted    = fitted(lm_model),
  Residuals = residuals(lm_model)
)

p_resid <- ggplot(residual_df, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5, color = "#0072B2") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth=1) +
  labs(title    = "Practical 4A: Residuals vs Fitted (Linear Regression)",
       subtitle = "Random scatter around zero = good fit",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face="bold"))
print(p_resid)

# ── Actual vs Predicted plot ───────────────────────────────────────────────────
pred_df <- data.frame(Actual = lm_actual, Predicted = lm_preds)
p_ap <- ggplot(pred_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "#E69F00") +
  geom_abline(slope=1, intercept=0, color="red", linetype="dashed", linewidth=1) +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(title    = "Practical 4A: Actual vs Predicted Deaths (Linear Regression)",
       subtitle = "Red dashed line = perfect prediction",
       x = "Actual Deaths", y = "Predicted Deaths") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face="bold"))
print(p_ap)

cat("\n")

# ════════════════════════════════════════════════════════════════════════════
# PART B : LOGISTIC REGRESSION — Predicting high_mortality (0/1)
# ════════════════════════════════════════════════════════════════════════════
cat("== PART B : LOGISTIC REGRESSION ==\n\n")
cat("Dependent variable   : high_mortality (1 = CFR > 2%, 0 = CFR ≤ 2%)\n")
cat("Independent variables: vaccination_pct, hospital_beds, gdp_per_capita,\n")
cat("                       stringency_index, positivity_rate\n\n")

log_model <- glm(high_mortality ~ vaccination_pct + hospital_beds +
                   gdp_per_capita + stringency_index + positivity_rate,
                 data   = train_data,
                 family = binomial(link = "logit"))

cat("-- Model Summary --\n")
print(summary(log_model))

# Odds ratios
cat("\n-- Odds Ratios (exp of coefficients) --\n")
odds <- exp(coef(log_model))
print(round(odds, 4))

# Predictions on test set
log_probs <- predict(log_model, newdata = test_data, type = "response")
log_preds <- ifelse(log_probs > 0.5, 1, 0)
log_actual <- test_data$high_mortality

# Confusion matrix
cat("\n-- Confusion Matrix --\n")
cm <- confusionMatrix(
  factor(log_preds,  levels=c(0,1)),
  factor(log_actual, levels=c(0,1)),
  positive = "1"
)
print(cm)

# Manual metrics display
acc  <- round(cm$overall["Accuracy"]  * 100, 2)
sens <- round(cm$byClass["Sensitivity"]* 100, 2)
spec <- round(cm$byClass["Specificity"]* 100, 2)
prec <- round(cm$byClass["Precision"]  * 100, 2)

cat("\n-- Logistic Regression Performance Summary --\n")
cat(sprintf("  Accuracy    : %.2f%%\n", acc))
cat(sprintf("  Sensitivity : %.2f%%  (True Positive Rate / Recall)\n", sens))
cat(sprintf("  Specificity : %.2f%%  (True Negative Rate)\n", spec))
cat(sprintf("  Precision   : %.2f%%\n", prec))

# ── Probability distribution plot ─────────────────────────────────────────────
prob_df <- data.frame(
  Probability    = log_probs,
  Actual_Class   = factor(log_actual,
                           levels=c(0,1),
                           labels=c("Low Mortality","High Mortality"))
)

p_prob <- ggplot(prob_df, aes(x = Probability, fill = Actual_Class)) +
  geom_histogram(binwidth = 0.05, alpha = 0.75, position = "identity", color="white") +
  geom_vline(xintercept = 0.5, color="black", linetype="dashed", linewidth=1) +
  scale_fill_manual(values = c("Low Mortality"="#009E73","High Mortality"="#CC79A7")) +
  labs(title    = "Practical 4B: Predicted Probability Distribution (Logistic Regression)",
       subtitle = "Dashed line = 0.5 decision boundary",
       x = "Predicted Probability of High Mortality",
       y = "Count", fill = "Actual Class") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face="bold"))
print(p_prob)

cat("\n>> Practical 4 COMPLETE.\n\n")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("============================================================\n")
cat("   DSBDA COVID-19 PROJECT — SUMMARY\n")
cat("============================================================\n")
cat("\n Practical 1 — Data Wrangling & Cleaning\n")
cat("   • Removed duplicates, imputed missing values (median)\n")
cat("   • Engineered: CFR, cases_per_million, positivity_rate,\n")
cat("     recovery_rate, active_cases, high_mortality\n")
cat("   • Reshaped data from wide to long format\n")

cat("\n Practical 2 — Descriptive Statistics\n")
cat("   • Computed mean, median, mode, SD, variance, IQR\n")
cat("   • Frequency distribution of mortality classes\n")
cat("   • Correlation matrix for 8 key variables\n")
cat("   • Group-wise statistics by country\n")

cat("\n Practical 3 — Data Visualization (7 plots)\n")
cat("   • Bar chart, Histogram, Boxplot, Line chart\n")
cat("   • Scatter + regression line, Correlation heatmap, Pie chart\n")

cat("\n Practical 4 — Regression\n")
cat(sprintf("   • Linear Regression  → RMSE = %.2f  |  R² = %.4f\n", lm_rmse, lm_r2))
cat(sprintf("   • Logistic Regression → Accuracy = %.2f%%\n", acc))

cat("\n============================================================\n")
cat("   END OF PROJECT\n")
cat("============================================================\n")
