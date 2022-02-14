"""
Project: Heart Stroke Prediction
Author: Shuai Xu | sxu75374@usc.edu
Date: 02/13/2022
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, \
    classification_report, roc_auc_score, fbeta_score, make_scorer, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_validate, \
    RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
import random


class DataLoader:
    def __init__(self):
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()

    def load(self, path_for_train, path_for_test):
        self.df_train = pd.read_csv(path_for_train)
        self.df_test = pd.read_csv(path_for_test)
        return self.df_train, self.df_test

    def dataInfo(self):
        print('\nInformation of the Training set:')
        self.df_train.info()
        print('Numebr of stroke = 0 in Training set', (self.df_train['stroke'] == 0).sum())
        print('Numebr of stroke = 1 in Training set', (self.df_train['stroke'] == 1).sum())
        print('\nInformation of the Testing set:')
        self.df_test.info()

    def checkMissingValues(self):
        print('\nMissing values in Training set: \n', self.df_train.isnull().sum())
        print('\nMissing values in Testing set: \n', self.df_test.isnull().sum())
        print(self.df_train.isna().sum() / len(self.df_train))
        print(self.df_test.isna().sum() / len(self.df_test))
        plt.figure()
        plt.bar(df_train.columns, self.df_train.isna().sum())
        plt.title('Missing Values in the dataset')
        plt.xlabel('Features')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.2, right=0.8)
        plt.ylabel('Number of Missing Values')


class MultiLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns

    def fit_transform(self, x_train, x_test, y_train=None):
        for column in self.columns:
            le = LabelEncoder()
            x_train[column] = le.fit_transform(x_train[column])
            x_test[column] = le.transform(x_test[column])
        return x_train, x_test


class DataPreProcessing:
    def __init__(self, resample_type):
        self.resample_type = resample_type

    def dealImbalance(self, resample_model, x_train, y_train):
        oversample = ['ovr', 'over', 'oversample', 'over_sample']
        undersample = ['udr', 'under', 'undersample', 'under_sample']
        if self.resample_type in oversample:
            print('===' * 20)
            print('Before OverSampling, counts of label 1: {}'.format(sum(y_train == 1)))
            print('Before OverSampling, counts of label 0: {} \n'.format(sum(y_train == 0)))
            print('Use Oversample method: {}'.format(resample_model))
            resample_x, resample_y = resample_model.fit_resample(x_train, y_train)
            print('After OverSampling, the shape of train_x: {}'.format(resample_x.shape))
            print('After OverSampling, the shape of train_y: {}'.format(resample_y.shape))
            print('After OverSampling, counts of label 1: {}'.format(sum(resample_y == 1)))
            print('After OverSampling, counts of label 0: {}'.format(sum(resample_y == 0)))
            return resample_x, resample_y

        if self.resample_type in undersample:
            print('===' * 20)
            print('Before UnderSampling, counts of label 1: {}'.format(sum(y_train == 1)))
            print('Before UnderSampling, counts of label 0: {} \n'.format(sum(y_train == 0)))
            print('Use Undersample method: {}'.format(resample_model))
            resample_x, resample_y = resample_model.fit_resample(x_train, y_train)
            print('After UnderSampling, the shape of train_x: {}'.format(resample_x.shape))
            print('After UnderSampling, the shape of train_y: {}'.format(resample_y.shape))
            print('After UnderSampling, counts of label 1: {}'.format(sum(resample_y == 1)))
            print('After UnderSampling, counts of label 0: {}'.format(sum(resample_y == 0)))
            return resample_x, resample_y


class DataVisualization:
    def __init__(self):
        pass


class Baseline:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def trivial(self):
        # try:
        from scipy import stats as s
        mode = int(s.mode(self.y_train)[0])
        print('mmmmmooooooddddd', mode)
        y_pred = [mode] * len(self.y_test)
        y_true = self.y_test
        print('trivial system:', accuracy_score(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        print(confusion_matrix(y_true, y_pred))
        plt.figure()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='Blues', fmt='.20g')
        plt.title('Confusion Matrix of Trivial Baseline')
        print(classification_report(y_true, y_pred))
        print('precision: ', precision_score(y_true, y_pred))
        print('recall: ', recall_score(y_true, y_pred))
        print('Sensitivity: ', recall_score(y_true, y_pred))
        print('Specificity: ', specificity)
        print('f1 score: ', f1_score(y_true, y_pred))
        print('f2 score: ', fbeta_score(y_true, y_pred, beta=2))

        k = random.randint(0, 1)  # decide on k once
        lists = []
        for _ in range(len(self.y_test)):
            lists.append(k)
        lists = np.array(lists).reshape(-1,1)
        fpr_lr, tpr_lr, _ = roc_curve(y_true, lists)

        plt.figure()
        plt.plot(fpr_lr, tpr_lr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve for {}".format('trivial'))
        plt.subplots_adjust(bottom=0.2, right=0.8)
        plt.plot([0, 1], [0, 1], lw=3, linestyle="--")
        sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
        print("Auc : ", auc(fpr_lr, tpr_lr))

    def non_trivial(self):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.x_train, self.y_train)
        y_pred = knn.predict(self.x_test)
        y_true = self.y_test
        print('trivial system:', accuracy_score(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        print(confusion_matrix(y_true, y_pred))
        plt.figure()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='Blues', fmt='.20g')
        plt.title('Confusion Matrix of Non-Trivial Baseline')
        print(classification_report(y_true, y_pred))
        print('precision: ', precision_score(y_true, y_pred))
        print('recall: ', recall_score(y_true, y_pred))
        print('Sensitivity: ', recall_score(y_true, y_pred))
        print('Specificity: ', specificity)
        print('f1 score: ', f1_score(y_true, y_pred))
        print('f2 score: ', fbeta_score(y_true, y_pred, beta=2))

        fpr_lr, tpr_lr, _ = roc_curve(y_true, knn.predict_proba(self.x_test)[:, 1])

        plt.figure()
        plt.plot(fpr_lr, tpr_lr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve for {}".format('non_trivial'))
        plt.subplots_adjust(bottom=0.2, right=0.8)
        plt.plot([0, 1], [0, 1], lw=3, linestyle="--")
        sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
        print("Auc : ", auc(fpr_lr, tpr_lr))


class TrainModel:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def predict_by_label(self, train=True):
        if train:
            self.model.fit(self.x_train, self.y_train)
        y_pred = self.model.predict(self.x_test)
        y_true = self.y_test
        print('===' * 20)
        if self.model:
            print('{}'.format(self.model))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        print('confusion_matrix')
        print(confusion_matrix(y_true, y_pred))
        plt.figure()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='Blues', fmt='.20g')
        plt.title('Confusion Matrix of {}'.format(self.model))
        print(classification_report(y_true, y_pred))
        print('precision: ', precision_score(y_true, y_pred))
        print('recall: ', recall_score(y_true, y_pred))
        print('Sensitivity: ', recall_score(y_true, y_pred))
        print('Specificity: ', specificity)
        print('f1 score: ', f1_score(y_true, y_pred))
        print('f2 score: ', fbeta_score(y_true, y_pred, beta=2))
        return y_pred

    def predict_by_probability(self, threshold=0.5, train=True):
        if train:
            self.model.fit(self.x_train, self.y_train)
        results = self.model.predict_proba(self.x_test)
        y_proba = results[:, 1]

        fpr_lr, tpr_lr, _ = roc_curve(y_val, y_proba)

        plt.figure()
        plt.plot(fpr_lr, tpr_lr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve for {}".format(self.model))
        plt.subplots_adjust(bottom=0.2, right=0.8)
        plt.plot([0, 1], [0, 1], lw=3, linestyle="--")

        sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
        print("Auc : ", auc(fpr_lr, tpr_lr))

        y_pred = (results[:, 1] > threshold).astype(int)
        y_true = self.y_test
        print('===' * 20)
        if self.model:
            print('{}'.format(self.model))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        print('confusion_matrix')
        print(confusion_matrix(y_true, y_pred))
        plt.figure()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='Blues', fmt='.20g')
        plt.title('Confusion Matrix of {}'.format(self.model))
        print(classification_report(y_true, y_pred))
        print('roc_aug score: ', roc_auc_score(y_true, y_proba))
        print('precision: ', precision_score(y_true, y_pred))
        print('recall: ', recall_score(y_true, y_pred))
        print('Sensitivity: ', recall_score(y_true, y_pred))
        print('Specificity: ', specificity)
        print('f1 score: ', f1_score(y_true, y_pred))
        print('f2 score: ', fbeta_score(y_true, y_pred, beta=2))
        return y_proba


class CV:
    def gridSearch(self, pipeline, params, scoring, x_train, y_train, show_cv_result=False, show_cv_plot=False):
        grid = GridSearchCV(pipeline, params, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1),
                            scoring=scoring, n_jobs=-1, return_train_score=True)
        grid.fit(x_train, y_train)
        print('===' * 10 + 'Cross Validation Result' + '===' * 10)
        print('The best score for scoring method \'{}\' is: {} '.format(scoring, grid.best_score_))
        print('The best settings for the best score is:', grid.best_params_)
        results = grid.cv_results_
        if show_cv_result:
            print('CV result:\n', results)
        if show_cv_plot:
            plt.figure(figsize=(13, 13))
            plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

            plt.xlabel("min_samples_split")
            plt.ylabel("Score")

            ax = plt.gca()
            ax.set_xlim(0, 402)
            ax.set_ylim(0.73, 1)

            # Get the regular numpy array from the MaskedArray
            X_axis = np.array(results["param_min_samples_split"].data, dtype=float)
            for scorer, color in zip(sorted(scoring), ["g", "k"]):
                for sample, style in (("train", "--"), ("test", "-")):
                    sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
                    sample_score_std = results["std_%s_%s" % (sample, scorer)]
                    ax.fill_between(
                        X_axis,
                        sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == "test" else 0,
                        color=color,
                    )
                    ax.plot(
                        X_axis,
                        sample_score_mean,
                        style,
                        color=color,
                        alpha=1 if sample == "test" else 0.7,
                        label="%s (%s)" % (scorer, sample),
                    )

                best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
                best_score = results["mean_test_%s" % scorer][best_index]

                # Plot a dotted vertical line at the best score for that scorer marked by x
                ax.plot(
                    [
                        X_axis[best_index],
                    ]
                    * 2,
                    [0, best_score],
                    linestyle="-.",
                    color=color,
                    marker="x",
                    markeredgewidth=3,
                    ms=8,
                )

                # Annotate the best score for that scorer
                ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

            plt.legend(loc="best")
            plt.grid(False)
            plt.show()
        return grid.best_estimator_

class Test:
    def getResult(self, bestmodel, test, threshold):
        proba = bestmodel.predict_proba(test)[:, 1]
        result = (proba > threshold).astype(int)
        return result


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_colwidth', 100)
    plt.ion()
    random_state = 42
    " 1. Data Loading "
    # data loader
    dataloader = DataLoader()
    df_train, df_test = \
        dataloader.load('/Users/xs/PycharmProjects/EE638/Heart Stroke Prediction/dataset/Training_data.csv',
                        '/Users/xs/PycharmProjects/EE638/Heart Stroke Prediction/dataset/Test_data.csv')
    dataloader.dataInfo()
    dataloader.checkMissingValues()

    " 2. Data Scrubbing "
    # deal with missing values
    bmi_median_value = df_train['bmi'].median()
    bmi_mean_value = df_train['bmi'].mean()
    df_train['bmi'].fillna(value=bmi_median_value, inplace=True)
    df_test['bmi'].fillna(value=bmi_median_value, inplace=True)
    dataloader.checkMissingValues()

    " 3. Data Visualization for Exploratory Data Analysis (EDA) - what the data can tell us "
    # data correlation
    corr = df_train.iloc[:, 1:].corr()
    plt.figure()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.subplots_adjust(bottom=0.2, right=0.8)
    plt.title('Correlation matrix for features')

    " 4. Data Preprocessing "
    # encoding categorical features
    print(set(df_train['gender']))
    print(set(df_train['ever_married']))
    print(set(df_train['work_type']))
    print(set(df_train['Residence_type']))
    print(set(df_train['smoking_status']))
    df_train_le, df_test_le = MultiLabelEncoder(['gender', 'ever_married', 'work_type', 'Residence_type',
                                                 'smoking_status']).fit_transform(df_train, df_test)
    print(df_train_le.describe())
    print(df_test_le.describe())

    corr2 = df_train_le.iloc[:, 1:].corr()
    plt.figure()
    sns.heatmap(corr2, annot=True)
    plt.subplots_adjust(bottom=0.2, right=0.8)
    plt.title('Corr after label encoder')

    # get training and test set: X, y is the whole training set, x_test is the test set
    X, y = df_train_le.to_numpy()[:, 1:-1], df_train_le.to_numpy()[:, -1]
    x_test = df_test_le.to_numpy()[:, 1:]
    print(df_train_le.head(10))
    print(df_test_le.head(10))

    # train, test split for validation: x_train is the training set for model selection, x_val is the validation set
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    print(x_train, len(y_train))
    print(x_val, len(y_val))

    # standardize
    ss = StandardScaler()
    x_train_st = ss.fit_transform(x_train)
    x_val_st = ss.transform(x_val)
    x_test_st = ss.transform(x_test)
    y_train_st = y_train
    y_val_st = y_val

    # RF feature importance
    rfc = RandomForestClassifier()
    rfc.fit(x_train_st, y_train_st)
    print(rfc.feature_importances_)
    FI_df = pd.DataFrame(
        {'Features': list(df_train_le.columns.drop(['id', 'stroke'])),
         'Feature importance': rfc.feature_importances_})
    FI = FI_df.sort_values('Feature importance', ascending=False)
    print('Feature Importance based on engineered feature space: \n', FI)
    plt.figure()
    plt.bar(FI['Features'], FI['Feature importance'])
    plt.title('Feature Importance based on original feature space')
    plt.xlabel('Features')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2, right=0.8)
    plt.ylabel('Feature Importance')

    # oversample
    sm = SMOTE(random_state=random_state)
    x_train_ovr, y_train_ovr = sm.fit_resample(x_train_st, y_train_st)

    " 5. Model Selection "
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=500, class_weight='balanced'),
        "Naive Bayes": GaussianNB(),
        "Bernolli Bayes": BernoulliNB(),
        'Decison Tree': DecisionTreeClassifier(random_state=random_state, class_weight='balanced'),
        'Random Forest Classifier': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
        'AdaBoost Classifier': AdaBoostClassifier(random_state=random_state),
        'Support Vector Machine': SVC(random_state=random_state, class_weight='balanced'),
        'K Nearest Classifier': KNeighborsClassifier()
        }

    models_name = []
    validation_accuracy = []
    validation_precision = []
    validation_recall = []
    validation_f1 = []
    validation_f2 = []
    validation_roc_auc = []

    # oversample & undersample
    over_sampling = SMOTE(random_state=random_state)
    under_sampling = RandomUnderSampler(random_state=random_state)
    f2_scorer = make_scorer(fbeta_score, beta=2)
    scoring = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall",
        "f1": "f1", "roc_auc": "roc_auc", "f2": f2_scorer
    }
    for model, classifier in models.items():
        print("Validating ", model)
        pipeline = make_pipeline(over_sampling, classifier)

        validation_score = cross_validate(pipeline, x_train_st, y_train_st,
                                          cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                                                     random_state=random_state),
                                          scoring=scoring, n_jobs=-1)
        print(validation_score)
        models_name.append(model)
        validation_accuracy.append(validation_score["test_accuracy"].mean())
        validation_precision.append(validation_score["test_precision"].mean())
        validation_recall.append(validation_score["test_recall"].mean())
        validation_f1.append(validation_score["test_f1"].mean())
        validation_f2.append(validation_score["test_f2"].mean())
        validation_roc_auc.append(validation_score["test_roc_auc"].mean())

        Validation_results = pd.DataFrame({
            "Model": models_name,
            "Validation_accuracy": validation_accuracy,
            "Validation_precision": validation_precision,
            "Validation_recall": validation_recall,
            "Validation_f1_score": validation_f1,
            "Validation_f2_score": validation_f2,
            "Validation_roc_auc": validation_roc_auc
        })

    print("Validation Complete")
    print(Validation_results)

    print('++++++++++++++ use Validation set to select models ++++++++++++++++')
    # we find 1. logistic regression, 2. Naive Bayes, 3. SVM, and 4. AdaBoost perform better than other
    # use val set to see the performance on * default settings *
    models_selected = {"Logistic Regression": LogisticRegression(random_state=random_state, max_iter=500,
                                                                 class_weight='balanced'),
                       "Naive Bayes": GaussianNB(),
                       'K Nearest Neighbor': KNeighborsClassifier(),
                       'Support Vector Machine': SVC(random_state=random_state, class_weight='balanced',
                                                     probability=True),
                       # 'SGD Classifier': SGDClassifier(random_state=random_state, class_weight='balanced', max_iter=100000)
                       'AdaBoost Classifier': AdaBoostClassifier(random_state=random_state)
                       }
    for model, classifier in models_selected.items():
        print(40 * "++")
        print("Validating ", model)
        pipeline_val = make_pipeline(over_sampling, classifier)
        if model == 'SGD Classifier':
            _ = TrainModel(model=pipeline_val, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                           y_test=y_val_st).predict_by_label()
        else:
            _ = TrainModel(model=pipeline_val, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                           y_test=y_val_st).predict_by_probability()

        fpr_lr, tpr_lr, _ = roc_curve(y_val, pipeline_val.predict_proba(x_val_st)[:, 1])

        plt.figure()
        plt.plot(fpr_lr, tpr_lr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve val test for {}".format(model))
        plt.subplots_adjust(bottom=0.2, right=0.8)
        plt.plot([0, 1], [0, 1], lw=3, linestyle="--")

        sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
        print("Auc : ", auc(fpr_lr, tpr_lr))
    print('++++++++++++++ Finish using Validation set to select models ++++++++++++++++')

    print('++++++++++++++ Begin using CV to tune hyperparameters ++++++++++++++++')
    # Tune the parameters by GridSearchCV
    cv_scoring = "roc_auc"
    # 1. SVC
    pipeline_svc = make_pipeline(over_sampling, SVC(random_state=random_state, class_weight='balanced', probability=True))
    params_svc = [
        {"svc__kernel": ['poly', 'rbf', 'sigmoid'],
         "svc__degree": [2, 3]
         }]

    svc_best = CV().gridSearch(pipeline_svc, params_svc, cv_scoring, x_train_st, y_train_st,
                               show_cv_result=False, show_cv_plot=False)

    # use val to test
    pipeline_svc_best = make_pipeline(over_sampling, svc_best)
    prob_svc = TrainModel(model=svc_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                          y_test=y_val_st).predict_by_probability(train=False)

    # 2. Logistic Regression
    pipeline_log = make_pipeline(over_sampling,
                                 LogisticRegression(random_state=random_state, max_iter=500, class_weight='balanced'))
    params_log = [{'logisticregression__C': [200, 100, 10, 1.0, 0.1, 0.01],
                   'logisticregression__penalty': ['l2', 'l1'],
                   'logisticregression__solver': ['newton-cg', 'lbfgs', 'liblinear']
                   }]

    lr_best = CV().gridSearch(pipeline_log, params_log, cv_scoring, x_train_st, y_train_st,
                              show_cv_result=False, show_cv_plot=False)

    # use val to test
    # pipeline_lr_best = make_pipeline(over_sampling, lr_best)
    prob_lr = TrainModel(model=lr_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                         y_test=y_val_st).predict_by_probability(train=False)

    # 3. Naive Bayes
    # use val to test
    pipeline_nb_best = make_pipeline(over_sampling, GaussianNB())
    prob_nb = TrainModel(model=pipeline_nb_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                         y_test=y_val_st).predict_by_probability()

    # 4. KNN
    pipeline_knn = make_pipeline(over_sampling, KNeighborsClassifier())
    params_knn = [
        {'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
         'kneighborsclassifier__n_neighbors': [1, 3, 5, 7]
         }]

    knn_best = CV().gridSearch(pipeline_knn, params_knn, cv_scoring, x_train_st, y_train_st,
                               show_cv_result=False, show_cv_plot=False)

    # use val to test
    prob_knn = TrainModel(model=knn_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                          y_test=y_val_st).predict_by_probability(train=False)

    # # 5. SGD
    # pipeline_sgd = make_pipeline(over_sampling,
    #                              SGDClassifier(random_state=random_state, class_weight='balanced', max_iter=100000))
    #
    # params_sgd = {
    #     "sgdclassifier__loss": ["hinge", "log", "squared_hinge", "modified_huber"],
    #     "sgdclassifier__alpha": [.0001, .001, .01, .1],
    #     "sgdclassifier__penalty": ["l2", "l1", "none"]
    # }
    #
    # sgd_best = CV().gridSearch(pipeline_sgd, params_sgd, cv_scoring, x_train_st, y_train_st, show_cv_result=False)
    #
    # # use val to test
    # prob_sgd = TrainModel(model=sgd_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
    #                       y_test=y_val_st).predict_by_label(train=False)

    # 6. AdaBoost
    pipeline_ada = make_pipeline(over_sampling, AdaBoostClassifier(random_state=random_state))
    params_ada = [
        {'adaboostclassifier__n_estimators': [1, 5, 10, 20, 30, 40, 50],
         }]

    ada_best = CV().gridSearch(pipeline_ada, params_ada, cv_scoring, x_train_st, y_train_st,
                               show_cv_result=False, show_cv_plot=False)

    # use val to test
    prob_ada = TrainModel(model=ada_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                          y_test=y_val_st).predict_by_probability(train=False)

    print('++++++++++++++ Finish using CV to tune hyperparameters ++++++++++++++++')

    " 6. Thresholding "
    def test_threshold(probas, y_test):
        results = []
        for i in range(20, 60):
            result = (probas > i / 100).astype(int)
            results.append((f1_score(y_test, result), i / 100))
        print(results)
        return sorted(results, key=(lambda x: x[0]), reverse=True)


    svc_best_f1 = test_threshold(prob_svc, y_val_st)
    lr_best_f1 = test_threshold(prob_lr, y_val_st)
    nb_best_f1 = test_threshold(prob_nb, y_val_st)
    knn_best_f1 = test_threshold(prob_knn, y_val_st)
    ada_best_f1 = test_threshold(prob_ada, y_val_st)
    print(svc_best_f1)
    print(lr_best_f1)
    print(nb_best_f1)
    print(knn_best_f1)
    print(ada_best_f1)

    t_svc = svc_best_f1[0][1]
    t_lr = lr_best_f1[0][1]
    t_nb = nb_best_f1[0][1]
    t_knn = knn_best_f1[0][1]
    t_ada = ada_best_f1[0][1]
    print(t_svc, t_lr, t_nb, t_knn, t_ada)

    print('============ Final Result After Thresholding ===========')
    " 0. Baseline - Trivial System "
    bl = Baseline(x_train_ovr, y_train_ovr, x_val_st, y_val_st)
    bl.trivial()

    " 0. Baseline - Non-trivial System "
    bl.non_trivial()

    " 1. Models "
    _ = TrainModel(model=svc_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                   y_test=y_val_st).predict_by_probability(train=False, threshold=t_svc)
    _ = TrainModel(model=lr_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                   y_test=y_val_st).predict_by_probability(train=False, threshold=t_lr)
    _ = TrainModel(model=pipeline_nb_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                   y_test=y_val_st).predict_by_probability(threshold=t_nb)
    _ = TrainModel(model=knn_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                   y_test=y_val_st).predict_by_probability(train=False, threshold=t_knn)
    _ = TrainModel(model=ada_best, x_train=x_train_st, y_train=y_train_st, x_test=x_val_st,
                   y_test=y_val_st).predict_by_probability(train=False, threshold=t_ada)

    "2. Final Prediction on Test set "
    pred_final = Test().getResult(bestmodel=lr_best, test=x_test_st, threshold=t_lr)
    print(pred_final)
    dataframe = pd.DataFrame({'y_test_pred': pred_final})
    dataframe.to_csv("output", index=False, sep=',')

    plt.ioff()
    plt.show()