try:
    import joblib
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
except ImportError as e:
    raise("Please install these libraries by running `pip install torch joblib numpy pandas xgboost scikit-learn`", e)


Note = """The Input dataframe must have a column with 'target' as its name."""

class Random_Forest_Classifier:
  
  @staticmethod
  def train_random_forest(df: pd.DataFrame, n_estimators: int=100):
    """
        Train a Random Forest Classifier on a DataFrame.

        Parameters:
        - df: A Pandas DataFrame containing both features and the 'target' column.
        - n_estimators: Number of trees in the forest (default=100).

        Returns:
        - rf_classifier: The trained Random Forest Classifier.

        Usage:
        ```python
        # Load your dataset into a DataFrame (df)
        # Train the Random Forest Classifier
        trained_model = Random_Forest_Classifier.train_random_forest(df)
        ```
    """
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=48)
    rf_classifier.fit(X_train, y_train)
      
    # Evaluate the model on the testing data
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Classifier Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion)

    # Classification Report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    return rf_classifier

  @staticmethod
  def save_model(model, filename: str="random_forest_classifier.joblib"):
    """
        Save a trained model to a file using joblib.

        Parameters:
        - model: The trained machine learning model to save.
        - filename: The name of the file to save the model to (default='random_forest_classifier.joblib').

        Usage:
        ```python
        # Save the trained model to a file
        Random_Forest_Classifier.save_model(trained_model, 'random_forest_model.joblib')
        ```
    """
    try:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving the model: {str(e)}")
  
  @staticmethod
  def load_model(filename):
    """
        Load a trained model from a file using joblib.

        Parameters:
        - filename: The name of the file containing the saved model.

        Returns:
        - model: The loaded machine learning model.

        Usage:
        ```python
        # Load the trained model from a file
        loaded_model = Random_Forest_Classifier.load_model('random_forest_model.joblib')
        ```
    """
    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return None
    
class XGBoostClassifier:
    def __init__(self):
        self.model = None

    def train_xgboost(self, df: pd.DataFrame, n_estimators: int=100):
        """
        Train an XGBoost Classifier on a DataFrame.

        Parameters:
        - df: A Pandas DataFrame containing both features and the 'target' column.
        - n_estimators: Number of boosting rounds (default=100).

        Returns:
        - xgb_classifier: The trained XGBoost Classifier.

        Usage:
        ```python
        # Load your dataset into a DataFrame (df)
        # Train the XGBoost Classifier
        trained_model = XGBoostClassifier.train_xgboost(df)
        ```
        """
        df['target'] = df['target'].map({-1: 0, 1: 1})
        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

        xgb_classifier = xgb.XGBClassifier(n_estimators=n_estimators, random_state=48)
        xgb_classifier.fit(X_train, y_train)
      
        # Evaluate the model on the testing data
        y_pred = xgb_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"XGBoost Classifier Accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix
        confusion = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion)

        # Classification Report
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)

        self.model = xgb_classifier
        return xgb_classifier

    @staticmethod
    def save_model(model, filename: str="xgboost_classifier.joblib"):
        """
        Save a trained XGBoost model to a file using joblib.

        Parameters:
        - model: The trained machine learning model to save.
        - filename: The name of the file to save the model to (default='xgboost_classifier.joblib').

        Usage:
        ```python
        # Save the trained model to a file
        XGBoostClassifier.save_model(trained_model, 'xgboost_model.joblib')
        ```
        """
        try:
            joblib.dump(model, filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving the model: {str(e)}")
  
    @staticmethod
    def load_model(filename):
        """
        Load a trained XGBoost model from a file using joblib.

        Parameters:
        - filename: The name of the file containing the saved model.

        Returns:
        - model: The loaded machine learning model.

        Usage:
        ```python
        # Load the trained model from a file
        loaded_model = XGBoostClassifier.load_model('xgboost_model.joblib')
        ```
        """
        try:
            model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return model
        except Exception as e:
            print(f"Error loading the model: {str(e)}")
            return None
        

class SupportVectorMachineClassifier:
    def __init__(self):
        self.model = None

    def train_svm(self, df: pd.DataFrame, kernel='linear'):
        """
        Train a Support Vector Machine (SVM) classifier on a DataFrame.

        Parameters:
        - df: A Pandas DataFrame containing both features and the 'target' column.
        - kernel: The SVM kernel type (default='linear').

        Returns:
        - svm_model: The trained SVM Classifier.

        Usage:
        ```python
        # Load your dataset into a DataFrame (df)
        # Train the SVM Classifier
        trained_model = SupportVectorMachineClassifier.train_svm(df, kernel='linear')
        ```
        """

        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

        # Train SVM
        svm_classifier = SVC(kernel=kernel, random_state=48, probability=True)
        svm_classifier.fit(X_train, y_train)
        
        # Evaluate the model on the testing data
        y_pred = svm_classifier.predict(X_test)
        svm_accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM Classifier Accuracy: {svm_accuracy * 100:.2f}%")

        # Confusion Matrix
        confusion = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion)

        # Classification Report
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)

        self.model = svm_classifier

        return svm_classifier

    @staticmethod
    def save_model(model, filename: str = "svm_classifier.joblib"):
        """
        Save a trained SVM model to a file using joblib.

        Parameters:
        - model: The trained machine learning model to save.
        - filename: The name of the file to save the model to (default='svm_classifier.joblib').

        Usage:
        ```python
        # Save the trained model to a file
        SupportVectorMachineClassifier.save_model(trained_model, 'svm_model.joblib')
        ```
        """
        try:
            joblib.dump(model, filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving the model: {str(e)}")

    @staticmethod
    def load_model(filename):
        """
        Load a trained SVM model from a file using joblib.

        Parameters:
        - filename: The name of the file containing the saved model.

        Returns:
        - model: The loaded machine learning model.

        Usage:
        ```python
        # Load the trained model from a file
        loaded_model = SupportVectorMachineClassifier.load_model('svm_model.joblib')
        ```
        """
        try:
            model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return model
        except Exception as e:
            print(f"Error loading the model: {str(e)}")
            return None
    
class NaiveBayesClassifier:
    def __init__(self):
        self.model = None

    def train_naive_bayes(self, df: pd.DataFrame):
        """
        Train a Naive Bayes classifier on a DataFrame.

        Parameters:
        - df: A Pandas DataFrame containing both features and the 'target' column.

        Returns:
        - nb_model: The trained Naive Bayes Classifier.

        Usage:
        ```python
        # Load your dataset into a DataFrame (df)
        # Train the Naive Bayes Classifier
        trained_model = NaiveBayesClassifier.train_naive_bayes(df)
        ```
        """
        
        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

        # Train Naive Bayes
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)
        
        # Evaluate the model on the testing data
        y_pred = nb_classifier.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_pred)
        print(f"Naive Bayes Classifier Accuracy: {nb_accuracy * 100:.2f}%")

        # Confusion Matrix
        confusion = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion)

        # Classification Report
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)

        self.model = nb_classifier

        return nb_classifier

    @staticmethod
    def save_model(model, filename: str = "naive_bayes_classifier.joblib"):
        """
        Save a trained Naive Bayes model to a file using joblib.

        Parameters:
        - model: The trained machine learning model to save.
        - filename: The name of the file to save the model to (default='naive_bayes_classifier.joblib').

        Usage:
        ```python
        # Save the trained model to a file
        NaiveBayesClassifier.save_model(trained_model, 'naive_bayes_model.joblib')
        ```
        """
        try:
            joblib.dump(model, filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving the model: {str(e)}")

    @staticmethod
    def load_model(filename):
        """
        Load a trained Naive Bayes model from a file using joblib.

        Parameters:
        - filename: The name of the file containing the saved model.

        Returns:
        - model: The loaded machine learning model.

        Usage:
        ```python
        # Load the trained model from a file
        loaded_model = NaiveBayesClassifier.load_model('naive_bayes_model.joblib')
        ```
        """
        try:
            model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return model
        except Exception as e:
            print(f"Error loading the model: {str(e)}")
            return None

class EnsembleClassifier:
    """
    A class for training and managing an ensemble classifier consisting of
    Random Forest, Support Vector Machine, Naive Bayes, and XGBoost models.

    Parameters:
    None

    Methods:
    - train(df: pd.DataFrame, n_estimators: int = 100, kernel: str = "linear"):
        Trains the ensemble classifier on the given DataFrame.

    - save_model(filename: str = "classifier_model.joblib"):
        Saves the trained model to a specified file using joblib.

    - load_model(filename: str = "classifier_model.joblib"):
        Loads a trained model from a specified file using joblib.

    Attributes:
    - model: The trained machine learning model.
    - model_name: The name of the currently selected model (Random Forest, Support Vector, Naive Bayes, or XGBoost).
    """

    def __init__(self):
        self.model = None
        self.model_name = None

    def train(self, df: pd.DataFrame, n_estimators: int=100, kernel: str="linear"):
        """
        Trains the ensemble classifier on the given DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the training data.
        - n_estimators (int): The number of estimators for Random Forest and XGBoost models.
        - kernel (str): The kernel type for the Support Vector Machine model.

        Returns:
        None
        """

        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=48)
        rf_classifier.fit(X_train, y_train)
        rf_accuracy = rf_classifier.score(X_test, y_test)
        highest_score = rf_accuracy
        self.model = rf_classifier
        self.model_name = "Random Forest"

        svm_classifier = SVC(kernel=kernel, random_state=48, probability=True)
        svm_classifier.fit(X_train, y_train)
        svm_accuracy = svm_classifier.score(X_test, y_test)
        if svm_accuracy > highest_score:
            highest_score = svm_accuracy
            self.model = svm_classifier
            self.model_name = "Support Vector"

        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)
        nb_accuracy = nb_classifier.score(X_test, y_test)
        if nb_accuracy > highest_score:
            highest_score = nb_accuracy
            self.model = nb_classifier
            self.model_name = "Naive Bias"

        df['target'] = df['target'].map({-1: 0, 1: 1})
        X = df.drop('target', axis=1)
        y = df['target']

        xgb_classifier = xgb.XGBClassifier(n_estimators=n_estimators, random_state=48)
        xgb_classifier.fit(X_train, y_train)
        xgb_accuracy = xgb_classifier.score(X_test, y_test)
        if xgb_accuracy > highest_score:
            highest_score = xgb_accuracy
            self.model = xgb_classifier
            self.model_name = "XG Boost"

    
    def save_model(self, filename: str="classifier_model.joblib"):
        """
        Saves the trained model to a specified file using joblib.

        Parameters:
        - filename (str): The name of the file to save the model.

        Returns:
        None
        """
        try:
            joblib.dump(self.model, filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving the model: {str(e)}")

    def load_model(self, filename: str="classifier_model.joblib"):
        """
        Loads a trained model from a specified file using joblib.

        Parameters:
        - filename (str): The name of the file containing the saved model.

        Returns:
        model: The loaded machine learning model.
        None: If loading the model fails.
        """
        try:
            model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return model
        except Exception as e:
            print(f"Error loading the model: {str(e)}")
            return None
