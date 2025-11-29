# app.py
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables
df = None
model = None
scaler = None

class AfricanFiscalAnalyzer:
    def __init__(self):
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load and preprocess the dataset"""
        try:
            self.df = pd.read_excel(file_path, sheet_name='Master_Dataset')
            print(f"✅ Dataset loaded: {self.df.shape}")
            
            # Data preprocessing
            self._preprocess_data()
            self._create_derived_metrics()
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _preprocess_data(self):
        """Clean and preprocess the data"""
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Create decade column for analysis
        self.df['Decade'] = (self.df['Year'] // 10) * 10
        
    def _create_derived_metrics(self):
        """Create innovative derived metrics"""
        # Fiscal Space Indicator
        self.df['Fiscal_Space'] = (
            self.df['Revenue_GDP'] - 
            self.df['Total_Expenditure_GDP'] - 
            (self.df['Interest_Payments_Revenue'] / 100)
        )
        
        # Debt Sustainability Score (0-100)
        self.df['Debt_Sustainability_Score'] = np.clip(
            100 - (
                self.df['Government_Debt_GDP'] * 0.4 +
                self.df['Debt_Service_Revenue_Ratio'] * 0.3 +
                (self.df['Budget_Balance_GDP'] * -2) * 0.3
            ), 0, 100
        )
        
        # Economic Resilience Index
        self.df['Economic_Resilience'] = (
            self.df['GDP_Growth'] * 0.25 +
            (100 - self.df['Inflation']) * 0.2 +
            self.df['Foreign_Reserves_Months'] * 5 +
            self.df['Human_Development_Index'] * 25
        )
        
        # Development Efficiency Ratio
        self.df['Development_Efficiency'] = (
            self.df['Human_Development_Index'] / 
            (self.df['Health_Spending_GDP'] + self.df['Education_Spending_GDP'])
        ) * 100

    def train_predictive_models(self):
        """Train machine learning models for predictions"""
        try:
            # Features for prediction
            features = [
                'Government_Debt_GDP', 'GDP_Growth', 'Inflation', 'Revenue_GDP',
                'Budget_Balance_GDP', 'Foreign_Reserves_Months', 'Export_GDP'
            ]
            
            # Target variables
            targets = ['GDP_Growth', 'Government_Debt_GDP', 'Inflation']
            
            for target in targets:
                # Prepare data
                model_features = [f for f in features if f != target]
                X = self.df[model_features]
                y = self.df[target]
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                self.models[target] = model
                
            print("✅ Predictive models trained successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return False
    
    def predict_future_trends(self, country, years_ahead=5):
        """Predict future economic trends for a country"""
        try:
            country_data = self.df[self.df['Country'] == country].sort_values('Year')
            if country_data.empty:
                return None
                
            latest_data = country_data.iloc[-1]
            predictions = {}
            
            for target, model in self.models.items():
                # Get feature importance
                features = [f for f in model.feature_names_in_ if f != target]
                current_values = latest_data[features].values.reshape(1, -1)
                
                # Simple trend-based prediction
                historical_trend = country_data[target].diff().mean()
                if pd.isna(historical_trend):
                    historical_trend = 0
                    
                current_value = latest_data[target]
                predicted_values = [current_value + (historical_trend * (i + 1)) for i in range(years_ahead)]
                predictions[target] = predicted_values
                
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

# Initialize analyzer
analyzer = AfricanFiscalAnalyzer()

# ===== INNOVATIVE ROUTES =====

@app.route('/')
def dashboard():
    """Main dashboard"""
    return jsonify({"message": "Welcome to the African Fiscal Analyzer API!"})

@app.route('/api/overview')
def get_overview():
    """Get comprehensive overview of all countries"""
    try:
        latest_year = analyzer.df['Year'].max()
        latest_data = analyzer.df[analyzer.df['Year'] == latest_year]
        
        overview = {
            'total_countries': analyzer.df['Country'].nunique(),
            'time_period': f"{analyzer.df['Year'].min()}-{latest_year}",
            'latest_year': int(latest_year),
            'key_metrics': {
                'avg_gdp_growth': round(latest_data['GDP_Growth'].mean(), 2),
                'avg_debt_gdp': round(latest_data['Government_Debt_GDP'].mean(), 2),
                'avg_inflation': round(latest_data['Inflation'].mean(), 2),
                'avg_hdi': round(latest_data['Human_Development_Index'].mean(), 3)
            },
            'regional_breakdown': analyzer.df['Region'].value_counts().to_dict(),
            'income_groups': analyzer.df['Income_Group'].value_counts().to_dict()
        }
        
        return jsonify(overview)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/country/<country_name>')
def get_country_analysis(country_name):
    """Get comprehensive analysis for a specific country"""
    try:
        country_data = analyzer.df[analyzer.df['Country'] == country_name]
        if country_data.empty:
            return jsonify({'error': 'Country not found'}), 404
            
        latest_year = country_data['Year'].max()
        latest_data = country_data[country_data['Year'] == latest_year].iloc[0]
        
        # Calculate trends
        trends = {}
        metrics = ['GDP_Growth', 'Government_Debt_GDP', 'Inflation', 'Budget_Balance_GDP']
        for metric in metrics:
            recent_data = country_data[country_data['Year'] >= latest_year - 5][metric]
            if len(recent_data) > 1:
                trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / len(recent_data)
                trends[metric] = round(trend, 2)
        
        # Risk assessment
        risk_factors = {
            'debt_risk': 'High' if latest_data['Government_Debt_GDP'] > 60 else 'Moderate' if latest_data['Government_Debt_GDP'] > 40 else 'Low',
            'inflation_risk': 'High' if latest_data['Inflation'] > 10 else 'Moderate' if latest_data['Inflation'] > 5 else 'Low',
            'fiscal_risk': 'High' if latest_data['Budget_Balance_GDP'] < -5 else 'Moderate' if latest_data['Budget_Balance_GDP'] < -3 else 'Low'
        }
        
        # Predictions
        predictions = analyzer.predict_future_trends(country_name, 3)
        
        analysis = {
            'basic_info': {
                'country': country_name,
                'region': latest_data['Region'],
                'income_group': latest_data['Income_Group'],
                'latest_year': int(latest_year)
            },
            'current_metrics': {
                'gdp_growth': round(latest_data['GDP_Growth'], 2),
                'government_debt': round(latest_data['Government_Debt_GDP'], 2),
                'inflation': round(latest_data['Inflation'], 2),
                'budget_balance': round(latest_data['Budget_Balance_GDP'], 2),
                'revenue_gdp': round(latest_data['Revenue_GDP'], 2),
                'hdi': round(latest_data['Human_Development_Index'], 3),
                'fiscal_space': round(latest_data.get('Fiscal_Space', 0), 2),
                'debt_sustainability': round(latest_data.get('Debt_Sustainability_Score', 0), 1),
                'economic_resilience': round(latest_data.get('Economic_Resilience', 0), 1)
            },
            'trends': trends,
            'risk_assessment': risk_factors,
            'predictions': predictions,
            'historical_data': country_data[['Year', 'GDP_Growth', 'Government_Debt_GDP', 'Inflation']].to_dict('records')
        }
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare')
def compare_countries():
    """Compare multiple countries"""
    try:
        countries = request.args.getlist('countries')
        if not countries:
            return jsonify({'error': 'No countries specified'}), 400
            
        comparison_data = {}
        latest_year = analyzer.df['Year'].max()
        
        for country in countries:
            country_data = analyzer.df[
                (analyzer.df['Country'] == country) & 
                (analyzer.df['Year'] == latest_year)
            ]
            if not country_data.empty:
                latest = country_data.iloc[0]
                comparison_data[country] = {
                    'gdp_growth': round(latest['GDP_Growth'], 2),
                    'government_debt': round(latest['Government_Debt_GDP'], 2),
                    'inflation': round(latest['Inflation'], 2),
                    'budget_balance': round(latest['Budget_Balance_GDP'], 2),
                    'revenue_gdp': round(latest['Revenue_GDP'], 2),
                    'hdi': round(latest['Human_Development_Index'], 3),
                    'debt_sustainability': round(latest.get('Debt_Sustainability_Score', 0), 1),
                    'region': latest['Region'],
                    'income_group': latest['Income_Group']
                }
        
        return jsonify({
            'comparison_year': int(latest_year),
            'countries': comparison_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster-analysis')
def cluster_analysis():
    """Perform cluster analysis to group similar countries"""
    try:
        latest_year = analyzer.df['Year'].max()
        latest_data = analyzer.df[analyzer.df['Year'] == latest_year].copy()
        
        # Features for clustering
        features = [
            'GDP_Growth', 'Government_Debt_GDP', 'Inflation', 'Revenue_GDP',
            'Budget_Balance_GDP', 'Human_Development_Index'
        ]
        
        # Prepare data
        X = latest_data[features].fillna(latest_data[features].median())
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        latest_data['Cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(4):
            cluster_data = latest_data[latest_data['Cluster'] == cluster_id]
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'countries': cluster_data['Country'].tolist(),
                'characteristics': {
                    'avg_gdp_growth': round(cluster_data['GDP_Growth'].mean(), 2),
                    'avg_debt': round(cluster_data['Government_Debt_GDP'].mean(), 2),
                    'avg_inflation': round(cluster_data['Inflation'].mean(), 2),
                    'avg_hdi': round(cluster_data['Human_Development_Index'].mean(), 3)
                },
                'label': _get_cluster_label(cluster_id, cluster_data)
            }
        
        return jsonify(cluster_analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _get_cluster_label(cluster_id, cluster_data):
    """Generate descriptive labels for clusters"""
    avg_debt = cluster_data['Government_Debt_GDP'].mean()
    avg_growth = cluster_data['GDP_Growth'].mean()
    avg_hdi = cluster_data['Human_Development_Index'].mean()
    
    if avg_growth > 5 and avg_debt < 50:
        return "High Growth, Sustainable Debt"
    elif avg_growth > 3 and avg_debt < 70:
        return "Moderate Growth, Manageable Debt"
    elif avg_debt > 70:
        return "High Debt Burden"
    else:
        return "Stable, Moderate Performance"

@app.route('/api/early-warning')
def early_warning_system():
    """Early warning system for economic risks"""
    try:
        latest_year = analyzer.df['Year'].max()
        latest_data = analyzer.df[analyzer.df['Year'] == latest_year].copy()
        
        warnings = []
        
        for _, country in latest_data.iterrows():
            country_warnings = []
            
            # Debt warning
            if country['Government_Debt_GDP'] > 70:
                country_warnings.append({
                    'type': 'DEBT_RISK',
                    'level': 'HIGH',
                    'message': f"Government debt ({country['Government_Debt_GDP']}% of GDP) exceeds sustainable levels"
                })
            elif country['Government_Debt_GDP'] > 60:
                country_warnings.append({
                    'type': 'DEBT_RISK', 
                    'level': 'MEDIUM',
                    'message': f"Government debt ({country['Government_Debt_GDP']}% of GDP) approaching risky levels"
                })
            
            # Inflation warning
            if country['Inflation'] > 15:
                country_warnings.append({
                    'type': 'INFLATION_RISK',
                    'level': 'HIGH',
                    'message': f"Inflation rate ({country['Inflation']}%) is very high"
                })
            elif country['Inflation'] > 8:
                country_warnings.append({
                    'type': 'INFLATION_RISK',
                    'level': 'MEDIUM', 
                    'message': f"Inflation rate ({country['Inflation']}%) is elevated"
                })
            
            # Fiscal warning
            if country['Budget_Balance_GDP'] < -6:
                country_warnings.append({
                    'type': 'FISCAL_RISK',
                    'level': 'HIGH',
                    'message': f"Large budget deficit ({country['Budget_Balance_GDP']}% of GDP)"
                })
            
            if country_warnings:
                warnings.append({
                    'country': country['Country'],
                    'warnings': country_warnings
                })
        
        return jsonify({'warnings': warnings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/policy-recommendations/<country_name>')
def get_policy_recommendations(country_name):
    """Generate AI-powered policy recommendations"""
    try:
        country_data = analyzer.df[analyzer.df['Country'] == country_name]
        if country_data.empty:
            return jsonify({'error': 'Country not found'}), 404
            
        latest_data = country_data[country_data['Year'] == country_data['Year'].max()].iloc[0]
        
        recommendations = []
        
        # Debt management recommendations
        if latest_data['Government_Debt_GDP'] > 60:
            recommendations.append({
                'category': 'DEBT_MANAGEMENT',
                'priority': 'HIGH',
                'recommendation': 'Implement fiscal consolidation to reduce debt burden',
                'actions': [
                    'Review and optimize public expenditure',
                    'Enhance revenue collection efficiency',
                    'Consider debt restructuring if necessary'
                ]
            })
        
        # Growth enhancement recommendations
        if latest_data['GDP_Growth'] < 3:
            recommendations.append({
                'category': 'ECONOMIC_GROWTH',
                'priority': 'HIGH', 
                'recommendation': 'Stimulate economic growth through targeted investments',
                'actions': [
                    'Increase infrastructure spending',
                    'Support private sector development',
                    'Improve business regulatory environment'
                ]
            })
        
        # Inflation control recommendations
        if latest_data['Inflation'] > 10:
            recommendations.append({
                'category': 'MONETARY_POLICY',
                'priority': 'HIGH',
                'recommendation': 'Implement measures to control inflation',
                'actions': [
                    'Tighten monetary policy if needed',
                    'Address supply-side constraints',
                    'Monitor food and energy prices'
                ]
            })
        
        # Development recommendations
        if latest_data['Human_Development_Index'] < 0.6:
            recommendations.append({
                'category': 'HUMAN_DEVELOPMENT',
                'priority': 'MEDIUM',
                'recommendation': 'Enhance human capital development',
                'actions': [
                    'Increase education and healthcare spending',
                    'Improve social protection systems',
                    'Promote gender equality and inclusion'
                ]
            })
        
        return jsonify({
            'country': country_name,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-data')
def export_data():
    """Export filtered data as CSV"""
    try:
        # Get filter parameters
        countries = request.args.getlist('countries')
        years = request.args.getlist('years')
        metrics = request.args.getlist('metrics')
        
        # Filter data
        filtered_df = analyzer.df.copy()
        
        if countries:
            filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
        if years:
            filtered_df = filtered_df[filtered_df['Year'].astype(str).isin(years)]
        if metrics:
            filtered_df = filtered_df[['Country', 'Year'] + metrics]
        
        # Create CSV
        output = io.StringIO()
        filtered_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'african_fiscal_data_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== INNOVATIVE FEATURES =====

@app.route('/api/fiscal-space-analysis')
def fiscal_space_analysis():
    """Analyze fiscal space across countries"""
    try:
        latest_year = analyzer.df['Year'].max()
        latest_data = analyzer.df[analyzer.df['Year'] == latest_year].copy()
        
        # Calculate fiscal space indicators
        latest_data['Fiscal_Space_Score'] = (
            latest_data['Revenue_GDP'] - 
            latest_data['Total_Expenditure_GDP'] -
            (latest_data['Interest_Payments_Revenue'] / 100) +
            (latest_data['GDP_Growth'] * 0.1) -
            (latest_data['Government_Debt_GDP'] * 0.01)
        )
        
        analysis = latest_data[['Country', 'Region', 'Fiscal_Space_Score', 
                              'Revenue_GDP', 'Total_Expenditure_GDP', 
                              'Government_Debt_GDP']].sort_values('Fiscal_Space_Score', ascending=False)
        
        return jsonify(analysis.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debt-sustainability-dashboard')
def debt_sustainability_dashboard():
    """Comprehensive debt sustainability analysis"""
    try:
        latest_year = analyzer.df['Year'].max()
        latest_data = analyzer.df[analyzer.df['Year'] == latest_year].copy()
        
        dashboard_data = []
        
        for _, country in latest_data.iterrows():
            # Calculate multiple sustainability metrics
            debt_ratio = country['Government_Debt_GDP']
            debt_service_ratio = country['Debt_Service_Revenue_Ratio']
            growth_rate = country['GDP_Growth']
            primary_balance = country['Primary_Balance_GDP']
            
            # Sustainability assessment
            if debt_ratio < 40 and debt_service_ratio < 15:
                sustainability = 'SUSTAINABLE'
            elif debt_ratio < 60 and debt_service_ratio < 25:
                sustainability = 'MODERATELY_SUSTAINABLE' 
            elif debt_ratio < 80:
                sustainability = 'AT_RISK'
            else:
                sustainability = 'UNSUSTAINABLE'
            
            dashboard_data.append({
                'country': country['Country'],
                'region': country['Region'],
                'debt_ratio': round(debt_ratio, 2),
                'debt_service_ratio': round(debt_service_ratio, 2),
                'growth_rate': round(growth_rate, 2),
                'primary_balance': round(primary_balance, 2),
                'sustainability': sustainability,
                'fiscal_space': round(country.get('Fiscal_Space', 0), 2)
            })
        
        return jsonify(sorted(dashboard_data, key=lambda x: x['debt_ratio'], reverse=True))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the application
    if analyzer.load_data('10Alytics_African_Fiscal_Dataset.xlsx'):
        analyzer.train_predictive_models()
        print("🚀 African Fiscal Analyzer started successfully!")
        print("📊 Available endpoints:")
        print("   - /api/overview - Comprehensive overview")
        print("   - /api/country/<name> - Country-specific analysis") 
        print("   - /api/compare - Compare countries")
        print("   - /api/cluster-analysis - Group similar countries")
        print("   - /api/early-warning - Risk assessment")
        print("   - /api/policy-recommendations/<name> - AI-powered recommendations")
        print("   - /api/fiscal-space-analysis - Fiscal capacity assessment")
        print("   - /api/debt-sustainability-dashboard - Debt analysis")
    else:
        print("⚠️  Could not load dataset. Please check file path.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)