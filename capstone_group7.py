import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import folium
from folium import Map, CircleMarker, Html, Element
from imblearn.over_sampling import SMOTE
from itertools import cycle

def analyze_crime_data():
    """
    Analyze and visualize crime data by hour of the day from a specific dataset, and return the cleaned DataFrame.

    This function reads a CSV file containing crime data from a predefined URL, 
    cleans it by removing specified columns, extracts the hour from the 'TIME OCC' 
    column, groups the data by hour, plots the number of crimes reported in 
    each hour of the day, and returns the cleaned data as a DataFrame before grouping. 
    The 'TIME OCC' column is expected to be in HHMM format for correct hour extraction.

    Parameters
    ----------
    None

    Returns
    -------
    DataFrame
        A cleaned DataFrame with specified columns removed and 'Hour' extracted from 'TIME OCC'.

    Notes
    -----
    - The dataset is expected to have a 'TIME OCC' column in HHMM format.
    - The function uses a hardcoded URL and set of columns to remove, which are 
      specific to the provided crime dataset.
    - The plot displayed using matplotlib shows the hours on the x-axis and crime 
      counts on the y-axis based on the grouped data by hour, but the returned DataFrame
      is before this grouping.

    Examples
    --------
    To analyze and visualize the crime data and get the cleaned DataFrame before grouping, call the function:

    >>> df_cleaned = analyze_crime_data()
    """

    # Locally defined file path and columns to remove
    file_path = 'https://media.githubusercontent.com/media/Anoop-Chandra-19/Python_ALY_6140/main/Crime_Data_from_2020_to_Present.csv'
    columns_to_remove = ['Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4']

    # Load the dataset
    data = pd.read_csv(file_path)

    # Drop specified columns
    df_cleaned = data.drop(columns=columns_to_remove)

    # Extracting relevant columns for time and crime analysis
    df_time_crime = df_cleaned[['TIME OCC', 'Date Rptd', 'Rpt Dist No']]

    # Extract hour from 'TIME OCC'
    df_time_crime.loc[:, 'Hour'] = pd.to_datetime(df_time_crime['TIME OCC'], format='%H', exact=False).dt.hour

    # Group by hour and count reported crimes
    crime_by_hour = df_time_crime.groupby('Hour').size().reset_index(name='Crime Count')

    # Plotting
    plt.figure(figsize=(10, 6))  # Set the figure size for better visibility
    plt.scatter(crime_by_hour['Hour'], crime_by_hour['Crime Count'], color='blue')  # Scatter plot

    # Adding titles and labels
    plt.title('Crime Count by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Crime Count')
    plt.xticks(range(0, 24))  # Ensure x-ticks for each hour

    # Show plot
    plt.grid(True)  # Optional: Adds a grid for easier reading
    plt.show()

    # Return the cleaned DataFrame
    return df_cleaned

def lg_data(df_cleaned):
    """
    Analyze data using logistic regression to predict a binary outcome based on provided features.
    
    This function preprocesses the input DataFrame by handling missing values, encoding the target variable,
    scaling features, and then applies logistic regression to predict the target variable. It evaluates the model
    performance using a ROC curve, classification report, and confusion matrix.

    Parameters
    ----------
    df_cleaned: pandas.DataFrame
        A cleaned DataFrame that must contain the features 'Vict Age', 'Premis Cd', 'Weapon Used Cd', 'LAT', 'LON',
        and a target variable 'Severity_Label'. The target variable should be binary.

    Returns
    -------
    None
        This function does not return any value. It plots the ROC curve and the confusion matrix,
        and prints the classification report for the logistic regression model's predictions.
    """
    # Define features and target variable
    features = ['Vict Age', 'Premis Cd', 'Weapon Used Cd', 'LAT', 'LON']
    X = df_cleaned[features]
    
    # Encoding the target variable
    y = LabelEncoder().fit_transform(df_cleaned['Severity_Label'])
    
    # Handling missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)
    
    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression Model
    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_train)
    
    # Making predictions and obtaining probabilities
    y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    y_pred = log_reg.predict(X_test_scaled)  # Predicting class labels for the test set
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Visualization of ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # Classification Report
    print("Classification Report of Binary Classification using Logistic Regression")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizing the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def dt_data(df_cleaned):
    """
    Analyze and classify a cleaned dataset using a Decision Tree classifier within a OneVsRest strategy 
    for multi-class classification. This function encodes categorical variables, handles class imbalance 
    with SMOTE, evaluates model performance with a classification report and confusion matrix, and plots 
    ROC curves for each class.

    Parameters
    ----------
    df_cleaned : pandas.DataFrame
        A cleaned DataFrame containing at least the columns 'Crm Cd Desc', 'Status', 'TIME OCC', 'LAT', 
        and 'LON'. The 'Crm Cd Desc' and 'Status' columns are expected to be categorical and will be 
        encoded numerically.

    Returns
    -------
    None
        The function does not return any value. It prints a classification report and displays a confusion 
        matrix and ROC curves for the multi-class classification.
    """
    # Encoding categorical variables
    label_encoder_crm = LabelEncoder()
    df_cleaned['Crm Cd Desc Encoded'] = label_encoder_crm.fit_transform(df_cleaned['Crm Cd Desc'])

    label_encoder_status = LabelEncoder()
    df_cleaned['Status Encoded'] = label_encoder_status.fit_transform(df_cleaned['Status'])

    # Selecting features and target variable for modeling
    features = df_cleaned[['Crm Cd Desc Encoded', 'TIME OCC', 'LAT', 'LON']]
    target = df_cleaned['Status Encoded']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Applying SMOTE
    min_samples = min(y_train.value_counts())
    if min_samples > 1:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    else:
        print("SMOTE cannot be applied due to extremely small class size.")

    # Binarizing the output for multi-class classification
    y = label_binarize(target, classes=np.unique(target))
    n_classes = y.shape[1]
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(features, y, test_size=0.3, random_state=42)
    X_train_scaled_bin = scaler.fit_transform(X_train_bin)
    X_test_scaled_bin = scaler.transform(X_test_bin)
    X_train_scaled_bin, y_train_bin = smote.fit_resample(X_train_scaled_bin, y_train_bin)

    # Training the Decision Tree Model
    decision_tree_classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
    decision_tree_classifier.fit(X_train_scaled_bin, y_train_bin)

    # Predictions
    predictions_dt = decision_tree_classifier.predict(X_test_scaled_bin)

    # Evaluation
    print("Decision Tree Classification Report")
    print(classification_report(y_test_bin, predictions_dt, target_names=label_encoder_status.classes_, zero_division=0))

    # Confusion Matrix Visualization
    conf_matrix_dt = confusion_matrix(y_test_bin.argmax(axis=1), predictions_dt.argmax(axis=1))
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_dt, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder_status.classes_, yticklabels=label_encoder_status.classes_,           annot_kws={"size": 12})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Decision Tree')
    plt.show()

def svm_data(df_cleaned):
    """
    Perform SVM binary classification on a cleaned DataFrame.

    This function preprocesses the data by encoding categorical variables, scaling features,
    handling class imbalance with SMOTE, training an SVM model for binary classification, and
    evaluating the model's performance through a classification report and a confusion matrix visualization.

    Parameters
    ----------
    df_cleaned : pandas.DataFrame
        A cleaned DataFrame containing at least the following columns: 'Crm Cd Desc', 'Status',
        'TIME OCC', 'LAT', 'LON'. 'Crm Cd Desc' and 'Status' are categorical variables that
        will be encoded. 'Status' will be converted into a binary target variable.

    Returns
    -------
    None
        This function does not return a value. It prints the classification report for the
        binary classification ('IC' vs. not) and displays a confusion matrix.
    """
    # Encoding the 'Crm Cd Desc' to numerical values since it's categorical
    label_encoder_crm = LabelEncoder()
    df_cleaned['Crm Cd Desc Encoded'] = label_encoder_crm.fit_transform(df_cleaned['Crm Cd Desc'])

    # Convert 'Status' to a binary target variable: 1 if 'IC', 0 otherwise
    df_cleaned['Status Binary'] = (df_cleaned['Status'] == 'IC').astype(int)

    # Selecting features and the binary target variable for modeling
    features = df_cleaned[['Crm Cd Desc Encoded', 'TIME OCC', 'LAT', 'LON']]
    target_binary = df_cleaned['Status Binary']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target_binary, test_size=0.3, random_state=42)

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handling class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    # SVM Model for Binary Classification
    svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)

    # Making predictions with the SVM model
    predictions_svm = svm_classifier.predict(X_test_scaled)

    # Evaluating the SVM Model
    print("SVM Classification Report for Binary Classification ('IC' vs. not):")
    print(classification_report(y_test, predictions_svm))

    # Visualizing the Confusion Matrix for SVM
    conf_matrix_svm = confusion_matrix(y_test, predictions_svm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_svm, annot=True, fmt='g', cmap='Blues', xticklabels=['Not IC', 'IC'], yticklabels=['Not IC', 'IC'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - SVM Binary Classification')
    plt.show()
