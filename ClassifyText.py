import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

paramGridSVC = [
    {
        'loss' : ['hinge', 'squared_hinge'], 
        'C': [0.01, 0.1, 1, 10, 100]
    },
    {
        'dual': [False],
        'penalty': ['l1'],
        'C': [0.1, 1, 10, 100]
    }
]

paramGridSGDClassifier = [
    {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss'],
        'alpha': [0.000001, 0.00001, 0.0001, 0.001]
    },
    {
        'loss' : ['epsilon_insensitive'],
        'epsilon' : [0.001, 0.0001, 0.00001],
        'alpha': [0.000001, 0.00001, 0.0001, 0.001]
    }
]

def loadData():
    print("Loading training data...")
    trainingData = fetch_20newsgroups(subset='train')
    print("Shape of trainingData:", trainingData.filenames.shape)
    x_train_vect = vectorizer.fit_transform(open(f).read() for f in trainingData.filenames)
    y_train_enc = trainingData.target
    encoder.fit(trainingData.target_names)
    print("Training data loaded.")

    print("Loading test data...")
    testData = fetch_20newsgroups(subset='test')
    print("Shape of testData:", testData.filenames.shape)
    x_test_vect = vectorizer.transform(open(f).read() for f in testData.filenames)
    y_test_enc = testData.target
    print("Test data loaded.")

    return x_train_vect, x_test_vect, y_train_enc, y_test_enc

def plotTrainingData():
    features = encoder.inverse_transform(y_train_enc)
    unique, counts = np.unique(features, return_counts=True)
    plt.title('Classes Bar Graph: y_train')
    plt.bar(unique, height=counts, color='skyblue', ec='black')
    plt.xticks(rotation=90)
    plt.show()

def testClassifier(_name, _classifier):
    print('Fitting %s...' % _name)
    _classifier.fit(x_train_vect, y_train_enc)
    print(_name, 'fitted.')
    predictions_enc = _classifier.predict(x_test_vect)
    
    report = classification_report(y_test_enc, predictions_enc, target_names=encoder.classes_)
    print(report)

    y_test = encoder.inverse_transform(y_test_enc)
    predictions = encoder.inverse_transform(predictions_enc)

    confusionMatrix = confusion_matrix(y_test, predictions)
    filename = 'images/' + _name + ' CM.png'
    plt.figure(figsize=(800/200, 800/200), dpi=200)
    plt.matshow(confusionMatrix)
    plt.title('Confusion Matrix: %s' % _name, y=1.08)
    plt.colorbar(fraction=0.046, pad=0.04)
    #plt.savefig(filename, dpi=200)
    #plt.show()

def getParameters(_classifier, _param):
    gridSearch = GridSearchCV(_classifier, _param, cv=3)
    gridSearch.fit(x_train_vect, y_train_enc)

    means = gridSearch.cv_results_['mean_test_score']
    stds = gridSearch.cv_results_['std_test_score']
    for mean, params in zip(means, gridSearch.cv_results_['params']):
            print("%0.5f for %r" % (mean, params))

    return gridSearch.best_params_

def predictText(_text, _classifier):
    _classifier.fit(x_train_vect, y_train_enc)
    text_vect = vectorizer.transform([_text])
    prediction_enc = _classifier.predict(text_vect)
    return encoder.inverse_transform(prediction_enc)

if __name__ == "__main__":
    vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()

    x_train_vect, x_test_vect, y_train_enc, y_test_enc = loadData()
    plotTrainingData()

    # Initialize all classifiers
    BernoulliNB = BernoulliNB()
    MultinomialNB = MultinomialNB()

    # Used to find best param, which are {'C': 1, 'loss': 'squared_hinge'}
    #print(getParameters(LinearSVC(), paramGridSVC))
    # Used to find best param for SGDClassifier, which are {'alpha': 0.0001, 'epsilon': 1e-05, 'loss': 'epsilon_insensitive'}
    #print(getParameters(SGDClassifier(), paramGridSGDClassifier))

    LinearSVCDefault = LinearSVC()
    SGDClassifierDefault = SGDClassifier()
    LinearSVCTuned = LinearSVC(C=1, loss='squared_hinge')
    SGDClassifierTuned = SGDClassifier(alpha=0.0001, epsilon=0.00001, loss='epsilon_insensitive')

    classifierDf = pd.DataFrame({
        'name': ['Bernoulli Naive Bayes', 'Multinomial Naive Bayes', 'LinearSVC Default', 'LinearSVC', 'Stochastic Gradient Descent Default', 'Stochastic Gradient Descent'],
        'classifier': [BernoulliNB, MultinomialNB, LinearSVCDefault, LinearSVCTuned, SGDClassifierDefault, SGDClassifierTuned]
    })

    # Test each classifier through looping
    for index, row in classifierDf.iterrows():
        testClassifier(row['name'], row['classifier'])

    # Use predictText to interface
    #text = 'This is a sample sentence about computer graphics. Articles should be much larger than this.'
    #print(predictText(text, LinearSVCTuned))

