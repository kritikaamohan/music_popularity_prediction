# Music Popularity Prediction Using Machine Learning
## Project Overwiew
- This project focuses on building a regression model to predict the popularity score of music tracks based on their audio features and metadata. By leveraging Machine Learning (ML) techniques, this system can help music streaming platforms, artists, and marketers better understand user preferences, optimize playlists, and enhance recommendation systems.

- The core of the project involves Exploratory Data Analysis (EDA) to understand the relationship between various song attributes (e.g., Energy, Danceability) and their popularity, followed by training and optimizing a Random Forest Regressor model.

## Key Features 
- Acousticness, Loudness, Energy, Danceability, Valence, Other Features

## Methodology
The Prediction task was achieved through the following steps:
1. Data Loading & Cleaning:
   - Imported the dataset and handled initial cleaning, such as dropping unnecessary columns (e.g., Unnamed: 0)
2. Exploratory Data Analysis (EDA):
   - Visualized the relationship between key features (Energy, Danceability, Loudness, Acousticness, Valence) and the target variable (Popularity) using scatter plots.
   - Generated a correlation matrix to quantify the relationships between all numerical features.
3. Feature Selection:
   - Selected eight key audio features for model training.
4. Data Preprocessing:
   - Splitting dataset into training and testing sets (80/20).
   - Applied Standard Scaling to normalize the features.
5. Model Training & Optimization :
   - Trained a Random Forest Regressor model.
   - Utilized GridSearchCV for hyperparameter tuning to find the optimal settings for the Random Forest model, resulting in the best performance compared to other tested algorithms.
6. Evaluation:
   - Visualized the Actual vs. Predicted Popularity scores on the test set to assess the model's accuracy.
  
## Dataset
The project uses a Spotify dataset containing 227 music tracks. Each track includes:
- Metadata: Track Name, Artists, Album Name, Release Date.
- Target Variable: Popularity (integer score).
- Audio Features: Duration, Explicit status, Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, and Tempo.

## Technologies and Libraries
- Language : Python
- Data Manipulation : pandas
- Data Visualization : matplotlib, seaborn
- Machine Learning : scikit-learn (sklearn)
  - RandomForestRegressor (Core Model)
  - StandardScaler (Prepocessing)
  - GridSearchCV (Hyperparameter tuning)
  - train_test_split , mean_squared_error, r2_score (Utilities and Evaluation)

## Installation and Setup
1. Clone the repository:
   - git clone   cd music_popularity_prediction
2. Install Dependencies:
   - pip install
      - pandas, matplotlib, seaborn, scikit-learn
3. Obtain the data:
   - Download the Spotify_data.csv dataset.
4. Run the script:
   - python music_popluarity_prediction.py
  
## Summary
The final tuned Random Forest Regressor showed strong predictive capabilities. The visualization of actual vs. predicted values indicated that most of the model's predictions clustered closely around the line of perfect prediction, demonstrating reasonable accuracy in estimating music popularity scores, though some deviations were observed, particularly for tracks with very low popularity.

