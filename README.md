This project develops a machine learning model to recommend coffee preferences based on tokenized feature inputs. Several classifiers are trained, and the best model is selected based on the Root Mean Squared Error (RMSE) score on the test data. The code includes data preprocessing, training, evaluation, and model saving.

1. Project Files
- coffee_recommendation_dataset.xlsx: Input dataset with features (Token_0 to Token_9) and labels for coffee preference.
- label_encoder.pkl: Saved label encoder to handle categorical label encoding in future predictions.
- best_model.pkl: The best-performing model saved for making predictions.

2. Project Workflow
- Data Preprocessing:

  - Label Encoding: The Label column (target variable) is label-encoded using LabelEncoder, enabling it to be used in the models.
  - One-Hot Encoding: Categorical features (Token_0 to Token_9) are one-hot encoded to create binary variables, making the data suitable for classification models.
  - Data Splitting: The dataset is divided into training (80%) and testing (20%) sets.
  - Standardization: The features are standardized using StandardScaler for the Support Vector Machine (SVM) model, as SVMs perform better on standardized data.

- Model Training and Evaluation:

  - Models Used: Four classifiers are implemented:
    - Random Forest Classifier
    - Logistic Regression
    - Decision Tree Classifier
    - Support Vector Machine (SVM)
  - Training Process:
    - Each model is trained on the training data. For SVM, the standardized dataset (X_train_scaled and X_test_scaled) is used, while the other models use the non-scaled data.
  - Model Evaluation: The performance is measured using the Root Mean Squared Error (RMSE) on the test data. RMSE provides a sense of how closely the model predictions match the actual labels.

- Selecting the Best Model:
  - The model with the lowest RMSE score on the test dataset is selected as the "best model" for deployment.
  - This best model is saved as best_model.pkl for future predictions, along with the label_encoder.pkl for label decoding.

How to Use
  Run the Script: Ensure coffee_recommendation_dataset.xlsx is in the same directory as the script and run the script to train the models.
  Model Output:
  - The console will display the RMSE for each model on the test data.
  - The script will print the name of the best model along with its RMSE.

- Deploying the Model: The saved best_model.pkl can be used to make predictions on new data. Additionally, label_encoder.pkl allows decoding of the model’s predictions back to the original label format.
