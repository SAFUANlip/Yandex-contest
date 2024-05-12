import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class MySolutions:
    """ a kNN classifier with L2 distance """

    def __init__(self, X_train, y_train, X_test):
        self.X_train_original = X_train
        self.y_train_original = y_train
        self.X_test_original = X_test
        pass

    def get_simple_logistic_reg(self):
        """
        Train the simple Logistic Regression classifier (from sklearn) with default hyperparameters.
        Return predicted class probabilities on the training and testing data.

        Returns:
            Predicted class probabilities and labels for train and test data in format:
            train_predicted_labels, train_predicted_probas, test_predicted_labels, test_predicted_probas
        """
        # some imports if needed
        clf = LogisticRegression().fit(X=self.X_train_original, y=self.y_train_original)
        labels_train = clf.predict(self.X_train_original)
        probas_train = clf.predict_proba(X=self.X_train_original)

        labels_test = clf.predict(self.X_test_original)
        probas_test = clf.predict_proba(X=self.X_test_original)

        return labels_train, probas_train, labels_test, probas_test

    def get_simple_naive_bayes(self):
        """
        Train the Naive Bayes classifier with Normal distribution as a prior.
        Use sklearn version (correct one!) and default hyperparameters.

        Returns:
            Predicted class probabilities for train and test data.
        """
        # some imports if needed
        gnb = GaussianNB()
        gnb.fit(X=self.X_train_original, y=self.y_train_original)

        labels_train = gnb.predict(self.X_train_original)
        probas_train = gnb.predict_proba(X=self.X_train_original)

        labels_test = gnb.predict(self.X_test_original)
        probas_test = gnb.predict_proba(X=self.X_test_original)

        return labels_train, probas_train, labels_test, probas_test

    def get_best_solution(self):
        """
        Train your best model. You can run some preprocessing (analysing the dataset might be useful),
        normalize the data, use nonlinear model etc. Get highscore!
        Please, do not use any external libraries but sklearn and numpy.

        Returns:
            Predicted class probabilities for train and test data.
        """
        # some imports if needed
        scaler = StandardScaler().fit(self.X_train_original)
        X_train_scaled = scaler.transform(self.X_train_original)
        X_test_scaled = scaler.transform(self.X_test_original)

        clf = LogisticRegression().fit(X=X_train_scaled, y=self.y_train_original)
        labels_train = clf.predict(X_train_scaled)
        probas_train = clf.predict_proba(X_train_scaled)

        labels_test = clf.predict(X_test_scaled)
        probas_test = clf.predict_proba(X_test_scaled)

        return labels_train, probas_train, labels_test, probas_test