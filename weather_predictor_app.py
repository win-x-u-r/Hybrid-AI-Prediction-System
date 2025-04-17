import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from collections import defaultdict

# === Load and preprocess ===
data = pd.read_csv("csai350-project/cleaned_weather.csv")
data.dropna(inplace=True)
data['weather_label'] = data['weather_label'].str.strip().str.title()

# Discretization
def categorize_temp(t):
    if t < 5: return "Low"
    elif t <= 20: return "Medium"
    else: return "High"

def categorize_humidity(h):
    if h < 40: return "Low"
    elif h <= 70: return "Medium"
    else: return "High"

def categorize_wind(w):
    if w <= 3: return "Low"
    elif w <= 7: return "Medium"
    else: return "High"

data['temp_cat'] = data['temperature'].apply(categorize_temp)
data['humidity_cat'] = data['humidity'].apply(categorize_humidity)
data['wind_cat'] = data['wind'].apply(categorize_wind)

# Train Bayesian Network
def train_bayesian_network(df):
    model = DiscreteBayesianNetwork([
        ('temp_cat', 'weather_label'),
        ('humidity_cat', 'weather_label'),
        ('wind_cat', 'weather_label')
    ])
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(model)
    return model, infer

bn_model, bn_infer = train_bayesian_network(data)

# === Markov Model Setup ===
states = sorted(data['weather_label'].unique())
icons = {
    "Sunny": "☀️", "Rainy": "🌧️", "Thunderstorm": "⚡", "Cloudy": "☁️", "Fog": "🌫",
    "Snow": "❄️", "Mist": "🌫", "Clear": "☀️", "Strong Wind": "💨", "Partly Cloudy": "⛅", "Storm": "⚡"
}
state_to_index = {state: i for i, state in enumerate(states)}
index_to_state = {i: state for state, i in state_to_index.items()}

def build_transition_matrix(sequence):
    transition_counts = defaultdict(lambda: defaultdict(int))
    for current, next_ in zip(sequence[:-1], sequence[1:]):
        transition_counts[current][next_] += 1
    matrix = np.zeros((len(states), len(states)))
    for i, curr in enumerate(states):
        total = sum(transition_counts[curr].values())
        for j, next_ in enumerate(states):
            matrix[i, j] = transition_counts[curr].get(next_, 0) / total if total > 0 else 1 / len(states)
    return matrix

transition_matrix = build_transition_matrix(data['weather_label'].values)

def predict_weather_markov(current_state, days, matrix):
    index = state_to_index[current_state]
    predictions = []
    for _ in range(days):
        next_index = np.random.choice(len(states), p=matrix[index])
        predictions.append(index_to_state[next_index])
        index = next_index
    return predictions

def estimate_continuous_values(state):
    subset = data[data['weather_label'] == state]
    if len(subset) == 0:
        return {"temperature": np.nan, "humidity": np.nan, "wind": np.nan}
    return {
        "temperature": subset['temperature'].mean().round(1),
        "humidity": subset['humidity'].mean().round(1),
        "wind": subset['wind'].mean().round(1)
    }

def predict_weather_bayes(temp, humidity, wind):
    evidence = {
        'temp_cat': categorize_temp(temp),
        'humidity_cat': categorize_humidity(humidity),
        'wind_cat': categorize_wind(wind)
    }
    prediction = bn_infer.map_query(variables=['weather_label'], evidence=evidence)
    return prediction['weather_label']

# === Visualization ===
def show_transition_graph(current_state):
    G = nx.DiGraph()
    probs = transition_matrix[state_to_index[current_state]]

    for i, prob in enumerate(probs):
        if prob > 0:
            G.add_edge(current_state, index_to_state[i], weight=round(prob, 2))

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"Transition Probabilities from '{current_state}'")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar([index_to_state[i] for i in range(len(states))], probs, color='skyblue')
    plt.ylabel("Probability")
    plt.title(f"Next-State Distribution from '{current_state}'")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# === GUI ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Weather Predictor")
        self.geometry("950x600")
        self.current_unit = "C"
        self.forecast_celsius = [None] * 7
        self.build_ui()

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        self.location_label = ctk.CTkLabel(self, text="3mman", font=("Segoe UI", 24, "bold"))
        self.location_label.pack(pady=(20, 5))

        self.current_weather_frame = ctk.CTkFrame(self)
        self.current_weather_frame.pack(pady=10)

        self.current_weather_icon = ctk.CTkLabel(self.current_weather_frame, text="☀️", font=("Segoe UI", 40))
        self.current_weather_icon.grid(row=0, column=0, padx=10)

        self.current_weather_text = ctk.CTkLabel(self.current_weather_frame, text="Partly cloudy", font=("Segoe UI", 16))
        self.current_weather_text.grid(row=0, column=1, padx=10)

        self.current_temp_label = ctk.CTkLabel(self.current_weather_frame, text="15°C", font=("Segoe UI", 16))
        self.current_temp_label.grid(row=0, column=2, padx=10)

        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=20, padx=20, fill="x")

        ctk.CTkLabel(input_frame, text="Temperature (°C):").grid(row=0, column=0, padx=5, sticky="w")
        self.temp_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 15")
        self.temp_input.grid(row=1, column=0, padx=5)

        ctk.CTkLabel(input_frame, text="Humidity (%):").grid(row=0, column=1, padx=5, sticky="w")
        self.humidity_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 65")
        self.humidity_input.grid(row=1, column=1, padx=5)

        ctk.CTkLabel(input_frame, text="Wind Speed (km/h):").grid(row=0, column=2, padx=5, sticky="w")
        self.wind_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 12")
        self.wind_input.grid(row=1, column=2, padx=5)

        ctk.CTkLabel(input_frame, text="Days to Forecast:").grid(row=0, column=3, padx=5, sticky="w")
        self.days_input = ctk.CTkSlider(input_frame, from_=1, to=7, number_of_steps=6)
        self.days_input.set(3)
        self.days_input.grid(row=1, column=3, padx=10, sticky="ew")

        self.predict_btn = ctk.CTkButton(self, text="Predict Weather", command=self.run_prediction)
        self.predict_btn.pack(pady=10)

        self.unit_frame = ctk.CTkFrame(self)
        self.unit_frame.pack(pady=5)
        self.unit_var = ctk.StringVar(value="C")
        ctk.CTkLabel(self.unit_frame, text="Temperature Unit:").pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="°C", variable=self.unit_var, value="C", command=self.update_units).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="°F", variable=self.unit_var, value="F", command=self.update_units).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="K", variable=self.unit_var, value="K", command=self.update_units).pack(side="left", padx=5)

        self.forecast_frame = ctk.CTkFrame(self)
        self.forecast_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.day_labels, self.temp_labels, self.weather_icons = [], [], []
        for i in range(7):
            self.forecast_frame.grid_columnconfigure(i, weight=1)
            day = ctk.CTkLabel(self.forecast_frame, text="", font=("Segoe UI", 12))
            temp = ctk.CTkLabel(self.forecast_frame, text="--", font=("Segoe UI", 14, "bold"))
            icon = ctk.CTkLabel(self.forecast_frame, text="", font=("Segoe UI", 24))
            day.grid(row=0, column=i, padx=10, pady=5)
            temp.grid(row=1, column=i, padx=10)
            icon.grid(row=2, column=i, padx=10)
            self.day_labels.append(day)
            self.temp_labels.append(temp)
            self.weather_icons.append(icon)

    def convert_temp(self, temp_c, unit):
        if unit == "F":
            return (temp_c * 9 / 5) + 32, "°F"
        elif unit == "K":
            return temp_c + 273.15, "K"
        return temp_c, "°C"

    def update_units(self):
        self.current_unit = self.unit_var.get()
        self.update_current_weather_display()
        self.update_forecast_display()

    def update_current_weather_display(self):
        try:
            temp_c = float(self.temp_input.get())
            temp, unit = self.convert_temp(temp_c, self.current_unit)
            self.current_temp_label.configure(text=f"{temp:.1f}{unit}")
        except:
            self.current_temp_label.configure(text="--")

    def update_forecast_display(self):
        for i in range(7):
            if self.forecast_celsius[i] is not None:
                temp, unit = self.convert_temp(self.forecast_celsius[i], self.current_unit)
                self.temp_labels[i].configure(text=f"{temp:.1f}{unit}")

    def run_prediction(self):
        try:
            temp = float(self.temp_input.get())
            humidity = float(self.humidity_input.get())
            wind = float(self.wind_input.get())
            days = int(self.days_input.get())

            predicted_label = predict_weather_bayes(temp, humidity, wind)
            self.current_weather_text.configure(text=predicted_label)
            self.current_weather_icon.configure(text=icons.get(predicted_label, "☀️"))
            self.update_current_weather_display()

            show_transition_graph(predicted_label)

            predictions = predict_weather_markov(predicted_label, days, transition_matrix)
            self.forecast_celsius = [None] * 7

            for i in range(7):
                if i < days:
                    state = predictions[i]
                    if i == 0:
                        temp_c = temp  # match user input for today
                    else:
                        values = estimate_continuous_values(state)
                        temp_c = values["temperature"]
                    self.forecast_celsius[i] = temp_c
                    temp_disp, unit = self.convert_temp(temp_c, self.current_unit)
                    self.day_labels[i].configure(text=["Today", "Tomorrow", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"][i])
                    self.temp_labels[i].configure(text=f"{temp_disp:.1f}{unit}")
                    self.weather_icons[i].configure(text=icons.get(state, "☀️"))
                else:
                    self.day_labels[i].configure(text="")
                    self.temp_labels[i].configure(text="--")
                    self.weather_icons[i].configure(text="")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app = WeatherApp()
    app.mainloop()
