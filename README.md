# airIQ

airIQ is a web application designed to predict the Air Quality Index (AQI) based on various air pollutant levels. The backend runs on Python and Flask, utilizing a machine learning model to serve real-time predictions.

## Features

- Predict AQI based on inputted air pollutant data.
- Built with Flask for the backend web interface.
- Includes pre-trained machine learning models for fast predictions.
- Data analysis and model training notebooks included in the repository.

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/GitH-Priyanshu/airIQ.git
   cd airIQ
   ```

2. Create a virtual environment and activate it (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://localhost:5000` to interact with the application.

## Project Structure

- `app.py`: The main Flask application entry point.
- `models/`: Contains the pre-trained machine learning models.
- `notebooks/`: Jupyter notebooks used for data analysis and model training.
- `scripts/`: Python scripts for data processing.
- `static/`: Static assets (CSS, JS) for the web interface.
- `templates/`: HTML templates for the web interface.
- `data/`: Datasets used for training the models.

## Future Work

- Add a heatmap visualization for geographical AQI monitoring.
- Integrate real-time API data for live predictions.
- Refine the UI/UX design.

## Author

Priyanshu
