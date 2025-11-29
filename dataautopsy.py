import pandas as pd
import numpy as np

def rapid_data_assessment(df):
    print("🔍 === CRITICAL DATA ASSESSMENT ===")
    print(f"📊 Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    
    # 1. TIME RANGE ANALYSIS
    if 'Year' in df.columns:
        year_range = f"{df['Year'].min()} to {df['Year'].max()}"
        print(f"📅 Time range: {year_range}")
    elif 'Time' in df.columns:
        print(f"📅 Time column present - check format")
    
    # 2. COUNTRY COVERAGE  
    if 'Country' in df.columns:
        country_count = df['Country'].nunique()
        print(f"🌍 Countries: {country_count}")
        # Show top 5 countries by data points
        country_coverage = df['Country'].value_counts().head()
        print(f"📈 Best-covered countries: {country_coverage.to_dict()}")
    
    # 3. CRITICAL VARIABLE CHECK
    print("\n🚨 === MISSING DATA CRISIS CHECK ===")
    critical_vars = ['Government_Debt', 'Budget_Balance', 'Revenue', 'GDP_growth', 
                    'Inflation', 'Debt_service', 'Interest_payments']
    
    missing_report = {}
    for var in critical_vars:
        if var in df.columns:
            missing_pct = (df[var].isnull().sum() / len(df)) * 100
            missing_report[var] = missing_pct
            status = "❌ CRITICAL" if missing_pct > 30 else "⚠️ WARNING" if missing_pct > 15 else "✅ OK"
            print(f"{status} {var}: {missing_pct:.1f}% missing")
    
    # 4. DATA QUALITY RED FLAGS
    print("\n🎯 === DATA QUALITY INDICATORS ===")
    
    # Check for impossible values
    if 'Government_Debt' in df.columns:
        debt_range = f"{df['Government_Debt'].min():.1f}% to {df['Government_Debt'].max():.1f}%"
        print(f"🏦 Debt-to-GDP range: {debt_range}")
        
    if 'Budget_Balance' in df.columns:
        balance_range = f"{df['Budget_Balance'].min():.1f}% to {df['Budget_Balance'].max():.1f}%"
        print(f"💰 Budget Balance range: {balance_range}")
    
    # 5. QUICK SAMPLE INSPECTION
    print("\n👀 === DATA SAMPLE ===")
    print(df.head(3).T)  # Transpose to see columns clearly

# RUN THIS IMMEDIATELY ON YOUR DATASET
rapid_data_assessment(your_dataframe)