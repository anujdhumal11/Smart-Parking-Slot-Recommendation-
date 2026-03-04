# Smart-Parking-Slot-Recommendation-
🚕 Smart Parking Slot Recommendation (K-Means + ANN + Agent Simulation)This project addresses urban parking inefficiencies by combining Unsupervised Learning for pattern recognition, Supervised Deep Learning for availability forecasting, and Agent-Based Simulation for real-world validation.

🚀 Key FeaturesIntelligent Clustering: Groups parking zones based on historical occupancy using K-Means to understand spatial-temporal trends.Predictive Modeling: Utilizes an Artificial Neural Network (ANN) with a multi-layer architecture (Dense layers with ReLU and Linear activations) to predict slot availability.Dynamic Recommendation Engine: Recommends the optimal zone for incoming drivers based on real-time features like hour, day of the week, and weather.Autonomous Navigation: Includes a Grid-World Agent Simulation that demonstrates the driver's journey from their starting position to the recommended slot.

🏗️ System ArchitectureThe pipeline follows a modular flow:Data Preprocessing: Feature engineering from timestamps, weather data, and zone IDs.Unsupervised Stage: Zone clustering via K-Means to identify peak usage patterns.Supervised Stage: Training an ANN regressor to forecast availability.Deployment: A recommendation engine and an agent-based simulation for end-to-end testing.

🛠️ Tech StackLanguages: Python Deep Learning: TensorFlow, Keras Machine Learning: Scikit-Learn (K-Means, StandardScaler) Data Analysis: Pandas, NumPy Visualization: Matplotlib, Seaborn Simulation: Gymnasium 

📊 Performance MetricsThe model is evaluated using Root Mean Squared Error (RMSE) for availability predictions, ensuring the recommendations are grounded in accurate numerical forecasts.
