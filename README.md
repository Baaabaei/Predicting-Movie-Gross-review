 

# Predicting Movie Gross Revenue

This repository contains a machine learning project that predicts the gross revenue of movies. The model is built using a Stochastic Gradient Descent (SGD) Regressor and is trained on the "CSM_dataset.xlsx" which includes various features like movie budget, ratings, and social media sentiment.

This project was completed as part of my Machine Learning course curriculum.

## Table of Contents

- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [2. Feature Engineering and Selection](#2-feature-engineering-and-selection)
  - [3. Model Training](#3-model-training)
  - [4. Model Evaluation](#4-model-evaluation)
- [Technologies Used](#technologies-used)
- [How to Run This Project](#how-to-run-this-project)
- [Key Findings](#key-findings)

## Project Goal

The primary objective of this project is to build a regression model that can accurately predict a movie's gross revenue. This involves data cleaning, feature scaling, training a machine learning model, and evaluating its performance on unseen data.

## Dataset

The dataset used for this project is `CSM_dataset.xlsx`. It contains the following columns:

-   `Movie`: The title of the movie.
-   `Year`: The release year of the movie.
-   `Ratings`: The movie's rating.
-   `Genre`: The genre of the movie.
-   `Gross`: The gross revenue of the movie (our target variable).
-   `Budget`: The production budget of the movie.
-   `Screens`: The number of screens the movie was released on.
-   `Sequel`: Indicates if the movie is a sequel.
-   `Sentiment`: A sentiment score based on reviews or social media.
-   `Views`: The number of views for the movie's trailer.
-   `Likes`: The number of likes for the movie's trailer.
-   `Dislikes`: The number of dislikes for the movie's trailer.
-   `Comments`: The number of comments on the movie's trailer.
-   `Aggregate Followers`: The total number of followers on social media for the movie.

## Methodology

The project follows a standard machine learning workflow, which is detailed below.

### 1. Data Loading and Preprocessing

The dataset was loaded using the `pandas` library. Initial preprocessing steps included:
-   Handling missing values (`NaN`) by filling them with the median of their respective columns. This ensures that the dataset is complete and ready for model training without introducing significant bias.
-   Dropping non-numeric or irrelevant columns like `Movie`.

### 2. Feature Engineering and Selection

-   **Feature Selection:** The `Gross` column was selected as the target variable (`y`), while other relevant columns were chosen as features (`X`). The `Movie` title and `Year` were excluded from the feature set as they are not suitable for direct use in a numerical regression model.
-   **Feature Scaling:** All selected features were scaled using the `MinMaxScaler` from `scikit-learn`. This normalizes the features to a range between 0 and 1, which is crucial for the performance of gradient-based algorithms like the SGD Regressor.

### 3. Model Training

-   **Train-Test Split:** The dataset was split into training (80%) and testing (20%) sets to ensure that the model could be evaluated on data it has not seen before.
-   **Algorithm:** A `SGDRegressor` (Stochastic Gradient Descent Regressor) from `scikit-learn` was used to build the regression model. This is an efficient and scalable algorithm, well-suited for linear regression tasks.

### 4. Model Evaluation

The performance of the trained model was evaluated using the following metrics:
-   **R-squared (Score):** The model achieved an R-squared score on the test set, indicating the proportion of the variance in the dependent variable that is predictable from the independent variables.
-   **Mean Squared Error (MSE):** The MSE was calculated to measure the average squared difference between the estimated values and the actual value.

## Technologies Used

This project was implemented in Python 3 and utilized the following libraries:

-   **Pandas:** For data manipulation and analysis.
-   **NumPy:** For numerical operations.
-   **Scikit-learn:** For machine learning tasks including data preprocessing, model training (`SGDRegressor`), and evaluation.
-   **Matplotlib & Seaborn:** For data visualization.
-   **Jupyter Notebook:** As the development environment.

## How to Run This Project

To run this project on your local machine, please follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install the required dependencies:**
    It is recommended to use a virtual environment.
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file listing the libraries mentioned above.)*

3.  **Run the Jupyter Notebook:**
    Launch Jupyter Notebook and open the `.ipynb` file to see the code and its output.
    ```sh
    jupyter notebook 1.1.ipynb
    ```

## Key Findings

-   The project demonstrates a complete workflow for a regression task, from data preparation to model evaluation.
-   Feature scaling proved to be an essential step for the `SGDRegressor`.
-   As noted in the notebook, categorical features like the movie title (`Movie`) are not suitable for direct use in this type of regression model and must be either dropped or properly encoded. In this case, it was dropped.
