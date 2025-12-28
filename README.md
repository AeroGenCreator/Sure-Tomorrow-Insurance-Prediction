# Sure-Tomorrow-Insurance-Prediction
This repository features a robust classification pipeline trained on 5,000 records from Sure Tomorrow. Beyond standard prediction, it explores the intersection of Machine Learning and Data Privacy (Notebook), implementing an obfuscation proof using Linear Algebra and a real-time deployment for model interaction.

## Sure Tomorrow: Insurance Prediction & Data Privacy

![image alt](https://github.com/AeroGenCreator/Sure-Tomorrow-Insurance-Prediction/blob/main/snaps/cover.png)

This repository features a comprehensive Machine Learning solution developed for the Sure Tomorrow insurance company. The project focuses on predicting client eligibility for benefits while demonstrating advanced data privacy techniques through linear algebra.

🚀 Interactive Demo
Experience the model in action and explore data correlations in real-time through the web interface: 👉 [Link to Web App]()
![image alt](https://github.com/AeroGenCreator/Sure-Tomorrow-Insurance-Prediction/blob/main/snaps/snap_1.png)

🛠️ Key Project Pillars

1. Validated Binary Classification

Beyond standard modeling, this project implements a Neighborhood Audit phase:

    Confidence Clustering: Validated that benefit labeling is consistent among similar profiles, 
    achieving a 97.34% Integrity Score.

    K-Nearest Neighbors (KNN): Optimized with k=3 and MaxAbsScaler, 
    significantly outperforming all dummy baselines.

    Performance Metrics: F1-Score: 95.21% | Precision: 96.36%.

2. Linear Algebra & Data Obfuscation (Privacy)

A methodological section demonstrating the feasibility of training models on encrypted data:

- Linear Regression from Scratch: Implementation based on the analytical matrix solution: $w=(XTX)−1XTy$.

    Analytical Proof: Mathematical and practical demonstration showing that multiplying 
    the feature matrix by an invertible matrix P does not alter predictions $\hat{y}$​, ensuring 
    user privacy without loss of accuracy.

3. Interactive Dashboard (Streamlit & DuckDB)

The cloud deployment features:

    Real-time Prediction: A dynamic form to evaluate insurance eligibility instantly.

    SQL Integration: Utilizes DuckDB for efficient on-the-fly data transformation.

    Correlation Analysis: Interactive Plotly visualizations explaining why 
    Age is the dominant risk factor (65% correlation).

📁 Repository Structure

    app.py: Streamlit dashboard source code.

    KNeighborsClassifier.joblib: The production-ready trained model.

    scaler.joblib: Fitted scaler for input normalization.

    insurance_us.csv: Historical insurance dataset.

    metadata.json: Model hyperparameters.

    notebook.ipynb: Full analysis, EDA, and mathematical proofs.
