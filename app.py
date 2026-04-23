from flask import Flask, render_template, request, jsonify
from utils.ml_pipeline import initialize_ml_models, get_metrics_data, predict_aqi

app = Flask(__name__)

# Run ML setup on startup
initialize_ml_models()

@app.route("/")
def home():
    """Render main UI"""
    return render_template("index.html")

@app.route("/api/metrics")
def get_metrics():
    """Return trained model metrics and scatter plot data"""
    metrics, scatter, error = get_metrics_data()
    if error:
        return jsonify({"error": error}), 500
    return jsonify({
        "metrics": metrics,
        "scatter": scatter
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """Real time prediction using Gradient Boosting Model"""
    data = request.get_json()
    result, error = predict_aqi(data)
    
    if error:
        return jsonify({"error": error}), 500
        
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
