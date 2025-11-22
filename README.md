# AI Weather Predictor

A sophisticated weather prediction application that combines Bayesian Networks and Markov Models to forecast weather conditions. Built with Python and CustomTkinter for an intuitive user experience.

## Features

- **Hybrid AI Prediction System**
  - Bayesian Network for current weather classification
  - Markov Model for multi-day forecasting
  - Maximum Likelihood Estimation for model training

- **Interactive GUI**
  - Dark/Light theme toggle
  - Real-time weather predictions
  - 7-day forecast display
  - Temperature unit conversion (°C, °F, K)
  - Transition probability visualization

- **Data-Driven Insights**
  - Trained on historical weather data
  - Smart categorization of weather parameters
  - Probabilistic weather state transitions

## Requirements

```
customtkinter
pandas
numpy
matplotlib
networkx
pgmpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/win-x-u-r/csai350-project.git
cd csai350-project
```

2. Install required packages:
```bash
pip install customtkinter pandas numpy matplotlib networkx pgmpy
```

3. Ensure `cleaned_weather.csv` is in the project directory.

## Usage

Run the application:
```bash
python weather_predictor_app.py
```

### Making Predictions

1. **Enter Weather Parameters:**
   - Temperature (°C)
   - Humidity (%)
   - Wind Speed (m/h)

2. **Select Forecast Duration:**
   - Use the slider to choose 1-7 days

3. **Click "Predict Weather":**
   - Current weather condition displayed with icon
   - Future forecasts shown in daily cards

4. **View Transition Graph:**
   - Click "Show Graph" to visualize weather transition probabilities

## How It Works

### Bayesian Network
The application uses a Discrete Bayesian Network with the following structure:
- **Nodes:** Temperature Category, Humidity Category, Wind Speed Category
- **Target:** Weather Label
- **Training:** Maximum Likelihood Estimation on historical data

Weather parameters are discretized into categories:
- **Temperature:** Low (<5°C), Medium (5-20°C), High (>20°C)
- **Humidity:** Low (<40%), Medium (40-70%), High (>70%)
- **Wind Speed:** Low (≤3 m/h), Medium (3-7 m/h), High (>7 m/h)

### Markov Model
For multi-day forecasting:
- Builds transition probability matrix from historical weather sequences
- Predicts next day's weather based on current state
- Estimates continuous values (temp, humidity, wind) for each predicted state

### Continuous Value Estimation
The system estimates specific numeric values using a fallback strategy:
1. Exact match on all categories + weather state
2. Partial matches (2/3 categories) with weighted averaging
3. Single category matches with lower weights
4. Weather state averages
5. Overall dataset averages

## Weather States

The model recognizes the following weather conditions:
- ☀ Sunny / Clear
- 🌧 Rainy
- ⚡ Thunderstorm / Storm
- ☁ Cloudy
- ⛅ Partly Cloudy
- 🌫 Fog / Mist
- ❄ Snow
- 💨 Strong Wind

## UI Features

- **Header:** Location display and theme toggle
- **Current Weather:** Large display with icon, temperature, humidity, and wind
- **Input Section:** Easy-to-use entry fields and forecast duration slider
- **Button Controls:** Predict and graph visualization buttons
- **Unit Selector:** Convert between Celsius, Fahrenheit, and Kelvin
- **Forecast Cards:** 7-day forecast with dates, icons, and detailed metrics

## Data Format

The application expects `cleaned_weather.csv` with the following columns:
- `temperature`: Temperature in Celsius
- `humidity`: Humidity percentage
- `wind`: Wind speed in m/h
- `weather_label`: Weather condition label

## Project Structure

```
csai350-project/
│
├── weather_predictor_app.py    # Main application file
├── cleaned_weather.csv          # Training data
└── README.md                    # Project documentation
```

## Technical Details

- **Framework:** CustomTkinter (modern tkinter-based UI)
- **Machine Learning:** pgmpy for Bayesian Network inference
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, networkx
- **Python Version:** 3.7+

## Input Validation

The application includes robust error handling:
- Humidity must be between 0-100%
- Wind speed cannot be negative
- All inputs must be valid numbers
- Graceful error messages for invalid inputs

## Future Enhancements

- Real-time weather data integration
- Location-based predictions
- Historical weather comparison
- Export forecast data
- Advanced visualization options

## License

This project is part of the CSAI 350 coursework.

## Author

Created for CSAI 350 - Introduction to Artificial Intelligence

## Acknowledgments

- pgmpy library for Bayesian Network implementation
- CustomTkinter for modern UI components
- Weather data contributors
