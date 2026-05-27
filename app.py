import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global variables for data and model
df = pd.DataFrame()
model = None

def init_app():
    global df, model
    # Load dataset
    try:
        df = pd.read_csv("data/processed/poverty_macro_clean.csv")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
    # Load model from the extracted 'deploy_model' directory
    try:
        model_path = os.path.join(os.getcwd(), "deploy_model")
        if os.path.exists(model_path):
            model = mlflow.sklearn.load_model(model_path)
            print("Model loaded successfully from deploy_model/")
        else:
            print("Deploy model directory not found. Please run extract_model.py first.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Initialize when starting the server
init_app()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/simulator")
def simulator():
    provinsi_list = []
    if not df.empty:
        # Get only 2025 for baseline selection
        df_2025 = df[df['Tahun'] == 2025]
        provinsi_list = sorted(df_2025['Provinsi'].unique().tolist())
    return render_template("simulator.html", provinsi_list=provinsi_list)

@app.route("/api/baseline/<provinsi>", methods=["GET"])
def get_baseline(provinsi):
    if df.empty:
        return jsonify({"error": "Data tidak tersedia"}), 500
        
    df_2025 = df[df['Tahun'] == 2025]
    if provinsi not in df_2025['Provinsi'].values:
        return jsonify({"error": "Provinsi tidak ditemukan"}), 404
        
    prov_data = df_2025[df_2025['Provinsi'] == provinsi].iloc[0]
    
    return jsonify({
        "pdrb": float(prov_data['Laju_PDRB']),
        "tpt": float(prov_data['TPT']),
        "gk_urban": float(prov_data['GK_Urban']),
        "gk_rural": float(prov_data['GK_Rural'])
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or df.empty:
        return jsonify({"error": "Model atau data tidak tersedia di server"}), 500
        
    data = request.json
    provinsi = data.get("provinsi")
    
    try:
        target_pdrb = float(data.get("pdrb"))
        target_tpt = float(data.get("tpt"))
    except (ValueError, TypeError):
        return jsonify({"error": "Input harus berupa angka"}), 400
    
    # Get baseline 2025 data for the selected province
    df_2025 = df[df['Tahun'] == 2025]
    if provinsi not in df_2025['Provinsi'].values:
        return jsonify({"error": "Provinsi tidak ditemukan"}), 404
        
    prov_data = df_2025[df_2025['Provinsi'] == provinsi].iloc[0]
    
    base_pdrb = float(prov_data['Laju_PDRB'])
    base_tpt = float(prov_data['TPT'])
    base_gk_urban = float(prov_data['GK_Urban'])
    base_gk_rural = float(prov_data['GK_Rural'])
    
    # Perform prediction for year 2026
    input_df = pd.DataFrame({
        'Tahun': [2026], 
        'Laju_PDRB': [target_pdrb], 
        'TPT': [target_tpt]
    })
    
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    pred_gk_urban = float(prediction[0])
    pred_gk_rural = float(prediction[1])
    
    pct_urban = ((pred_gk_urban - base_gk_urban) / base_gk_urban) * 100
    pct_rural = ((pred_gk_rural - base_gk_rural) / base_gk_rural) * 100
    
    # Generate Rule-based Recommendation
    alert_type = "info"
    alert_title = "KONDISI MODERAT"
    alert_text = "Fluktuasi kemiskinan diprediksi berada dalam batas wajar. Lanjutkan program pembangunan berjalan dengan tetap memonitor indeks harga bahan pokok utama setiap bulan."
    
    if pct_urban > 10.0 or pct_rural > 10.0:
        alert_type = "danger"
        alert_title = "INTERVENSI MENDESAK"
        alert_text = "Sistem mendeteksi potensi lonjakan biaya hidup ekstrem (>10%). Segera aktifkan operasi pasar, optimalkan distribusi pangan melalui tol laut/udara, dan siapkan bantalan sosial ekstra untuk masyarakat rentan di wilayah ini."
    elif target_tpt > base_tpt + 2.0:
        alert_type = "warning"
        alert_title = "WASPADA GELOMBANG PHK"
        alert_text = "Kenaikan target pengangguran berisiko memukul daya beli. Percepat peluncuran proyek infrastruktur padat karya dan program pelatihan upskilling lokal."
    elif pct_urban < 0 and pct_rural < 0:
        alert_type = "success"
        alert_title = "KONDISI STABIL"
        alert_text = "Biaya hidup diprediksi mengalami deflasi/penurunan. Momentum yang sangat baik untuk mengalihkan subsidi konsumsi menjadi insentif permodalan bagi UMKM lokal."

    # Return as JSON
    return jsonify({
        "baseline": {
            "pdrb": base_pdrb,
            "tpt": base_tpt,
            "gk_urban": base_gk_urban,
            "gk_rural": base_gk_rural
        },
        "prediction": {
            "gk_urban": pred_gk_urban,
            "gk_rural": pred_gk_rural,
            "pct_urban": pct_urban,
            "pct_rural": pct_rural
        },
        "recommendation": {
            "type": alert_type,
            "title": alert_title,
            "text": alert_text
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
