import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---- LOAD DATA ----
df = pd.read_csv(r"C:\Users\yesas\Desktop\retail-sales-forecasting\data\raw_sales.csv",
                 encoding='latin1')
print(f"Raw data shape: {df.shape}")

# ============================================================
# STEP 1: CLEAN DATA
# ============================================================

# Convert dates
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# Extract date parts
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Month_Name'] = df['Order Date'].dt.strftime('%b')
df['Year_Month'] = df['Order Date'].dt.to_period('M')

# Drop unnecessary columns
df = df.drop(columns=['Row ID', 'Postal Code', 'Customer ID', 'Product ID'])

# Round sales
df['Sales'] = df['Sales'].round(2)

print(f"✓ Dates cleaned")
print(f"✓ Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
print(f"✓ Years in data: {sorted(df['Year'].unique())}")

# ============================================================
# STEP 2: SAVE CLEAN DATA FOR TABLEAU
# ============================================================
clean_path = r"C:\Users\yesas\Desktop\retail-sales-forecasting\data\clean_sales.csv"
df.to_csv(clean_path, index=False)
print(f"✓ Clean data saved for Tableau!")

# ============================================================
# STEP 3: CREATE MONTHLY SALES TIME SERIES
# ============================================================
monthly_sales = df.groupby('Year_Month')['Sales'].sum().reset_index()
monthly_sales['Year_Month'] = monthly_sales['Year_Month'].astype(str)
monthly_sales['Order Date'] = pd.to_datetime(monthly_sales['Year_Month'])
monthly_sales = monthly_sales.sort_values('Order Date')
monthly_sales['Sales'] = monthly_sales['Sales'].round(2)

print(f"\n✓ Monthly sales data:")
print(monthly_sales[['Year_Month', 'Sales']].to_string())

# Save monthly data
monthly_path = r"C:\Users\yesas\Desktop\retail-sales-forecasting\data\monthly_sales.csv"
monthly_sales.to_csv(monthly_path, index=False)
print(f"\n✓ Monthly sales saved!")

# ============================================================
# STEP 4: ARIMA FORECASTING
# ============================================================
print("\n--- Running ARIMA Forecasting ---")

# Install statsmodels if needed:
# pip install statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Use Sales as time series
ts = monthly_sales.set_index('Order Date')['Sales']

# Check if series is stationary
adf_result = adfuller(ts)
print(f"ADF Test p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("✓ Series is stationary — good for ARIMA!")
else:
    print("Series is not stationary — will use differencing in ARIMA")

# Fit ARIMA model
# (1,1,1) is a good starting point for retail sales
model = ARIMA(ts, order=(1, 1, 1))
fitted_model = model.fit()
print(f"\n✓ ARIMA model fitted successfully!")
print(f"AIC Score: {fitted_model.aic:.2f}")

# Forecast next 6 months
forecast = fitted_model.forecast(steps=6)
last_date = ts.index[-1]
forecast_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=6, freq='MS'
)
forecast_df = pd.DataFrame({
    'Order Date': forecast_dates,
    'Forecasted_Sales': forecast.round(2),
    'Type': 'Forecast'
})
print(f"\n✓ 6-month forecast:")
print(forecast_df[['Order Date', 'Forecasted_Sales']].to_string())

# Save forecast
forecast_path = r"C:\Users\yesas\Desktop\retail-sales-forecasting\data\forecast_sales.csv"
forecast_df.to_csv(forecast_path, index=False)
print(f"\n✓ Forecast saved!")

# ============================================================
# STEP 5: CHARTS
# ============================================================
SAVE = r"C:\Users\yesas\Desktop\retail-sales-forecasting\visuals"
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "font.family": "DejaVu Sans",
})
COLORS = ["#1D9E75", "#534AB7", "#D85A30", "#BA7517", "#185FA5"]

# CHART 1: Monthly Sales Trend + Forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ts.index, ts.values, color=COLORS[0],
        linewidth=2.5, label='Actual Sales', marker='o', markersize=4)
ax.plot(forecast_dates, forecast.values, color=COLORS[2],
        linewidth=2.5, linestyle='--', label='Forecasted Sales',
        marker='o', markersize=4)
ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1.5,
           label='Forecast Start')
ax.fill_between(forecast_dates, forecast.values * 0.85,
                forecast.values * 1.15, alpha=0.2, color=COLORS[2],
                label='Confidence Range')
ax.set_title("Monthly Sales Trend with 6-Month ARIMA Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Sales (USD)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{SAVE}\\chart1_sales_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 1 saved: sales forecast")

# CHART 2: Sales by Category
category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(category_sales.index, category_sales.values,
               color=COLORS[:3], edgecolor="none")
for bar, val in zip(bars, category_sales.values):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f'${val:,.0f}', va='center', fontsize=11, fontweight='bold')
ax.set_title("Total Sales by Product Category")
ax.set_xlabel("Total Sales (USD)")
plt.tight_layout()
plt.savefig(f"{SAVE}\\chart2_sales_by_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 2 saved: sales by category")

# CHART 3: Sales by Region
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(region_sales.index, region_sales.values,
              color=COLORS, edgecolor="none", width=0.5)
for bar, val in zip(bars, region_sales.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 100,
            f'${val:,.0f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title("Total Sales by Region")
ax.set_ylabel("Total Sales (USD)")
plt.tight_layout()
plt.savefig(f"{SAVE}\\chart3_sales_by_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 3 saved: sales by region")

# CHART 4: Monthly Sales by Year
fig, ax = plt.subplots(figsize=(12, 6))
for i, year in enumerate(sorted(df['Year'].unique())):
    year_data = df[df['Year'] == year].groupby('Month')['Sales'].sum()
    ax.plot(year_data.index, year_data.values,
            color=COLORS[i], linewidth=2,
            marker='o', markersize=5, label=str(year))
ax.set_title("Monthly Sales Comparison by Year")
ax.set_xlabel("Month")
ax.set_ylabel("Sales (USD)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.legend(title="Year", fontsize=10)
plt.tight_layout()
plt.savefig(f"{SAVE}\\chart4_yearly_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 4 saved: yearly comparison")

# CHART 5: Top 10 Sub-Categories by Sales
subcat_sales = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=True).tail(10)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(subcat_sales.index, subcat_sales.values,
               color=COLORS[0], edgecolor="none")
for bar, val in zip(bars, subcat_sales.values):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f'${val:,.0f}', va='center', fontsize=10)
ax.set_title("Top 10 Sub-Categories by Total Sales")
ax.set_xlabel("Total Sales (USD)")
plt.tight_layout()
plt.savefig(f"{SAVE}\\chart5_subcategory_sales.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 5 saved: subcategory sales")

# CHART 6: Sales by Customer Segment
segment_sales = df.groupby('Segment')['Sales'].sum()
fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(
    segment_sales.values,
    labels=segment_sales.index,
    autopct='%1.1f%%',
    colors=COLORS[:3],
    startangle=90,
    wedgeprops={"linewidth": 2, "edgecolor": "white"}
)
ax.set_title("Sales Distribution by Customer Segment")
plt.tight_layout()
plt.savefig(f"{SAVE}\\chart6_segment_sales.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 6 saved: segment sales")

print("\n✓ All charts saved!")
print("→ Now open Tableau Public for the dashboard!")