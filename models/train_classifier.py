import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    '''
    INPUT
    database_filepath --> loads the DisasterResponse.db 

    OUTPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    category_names - list of the y vble names
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('data/DisasterResponse.db', con=engine)
    df = df[df['related'] != '2']
    X = df['message']
    Y = df[df.columns[4:]].apply(pd.to_numeric, errors ='ignore')
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT
    Any text --> loads the DisasterResponse.db 

    OUTPUT
    A normalised, tokenized, without stopwords and lemmatized text
    '''
    # Normalize text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    
    # Remove Stop Words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer() 
    lemmed = [lemmatizer.lemmatize(w) for w in tokens]

    return lemmed


def build_model():
    '''
    OUTPUT
    A pipeline using cv grid search
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        #'vect__max_features': (None, 5000),
        'clf__estimator__n_estimators': [10,20]
    }
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT --> imports the model, X_test, Y_test and category_names and predicts the X_test
    OUTPUT --> Y_pred
    '''
    Y_pred = model.predict(X_test)
    

def save_model(model, model_filepath):
    '''
    INPUT --> model and model_filepath
    OUTPUT --> saves the model on a 'Classifier.pkl'
    '''
    filename = 'Classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))


def main():
    '''
    INPUT --> takes 3 arguments: The train classifier, the database and the name of the file the model will be saved on
    OUTPUT --> Load the database
               Train_test split the data from the database
               Create a model
               Train the model
               Evaluate the model
               Save the model
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()