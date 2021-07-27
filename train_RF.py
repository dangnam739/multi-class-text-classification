import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

from utils.data_utils import preprocess_RF, CLASS_NAME_TO_ID


CUR_DIR = os.getcwd()
DATA_DIR_PATH = f'{CUR_DIR}/dataset'
TRAIN_DIR_PATH = f'{DATA_DIR_PATH}/train'
TEST_DIR_PATH = f'{DATA_DIR_PATH}/test'
STOP_WORD_FILE = f'{DATA_DIR_PATH}/stop_words.txt'

words_list = np.load(os.path.join(DATA_DIR_PATH, 'words_list.npy'))
words_list = words_list.tolist()

# parameter
max_sen_len = 2000
min_df = 5

if __name__ == '__main__':
    # load data
    train_df = pd.read_csv(os.path.join(TRAIN_DIR_PATH, 'train.csv'), sep='\t')
    test_df = pd.read_csv(os.path.join(TEST_DIR_PATH, 'test.csv'), sep='\t')

    # preprocess data
    train_df["preprocessed"] = train_df["text"].apply(lambda x: preprocess_RF(x, STOP_WORD_FILE))
    test_df["preprocessed"] = test_df["text"].apply(lambda x: preprocess_RF(x, STOP_WORD_FILE))

    # create train, test data
    X_train, Y_train = train_df["preprocessed"], train_df["class_id"]
    X_test, Y_test = test_df["preprocessed"], test_df["class_id"]

    # create vectorizer using tfidfvectorizer
    vectorizer = TfidfVectorizer(min_df=min_df,
                                norm='l2',
                                sublinear_tf=True,
                                smooth_idf=True,
                                use_idf=True)
    
    # create pipline 
    pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', RandomForestClassifier())])

    # training
    model = pipeline.fit(X_train, Y_train)

    # testing
    pred = model.predict(X_test)

    y_test = np.array(Y_test)

    acc = accuracy_score(pred, Y_test)*100
    print(classification_report(y_test, model.predict(X_test)))
    print("Accuracy: {}".format(acc))

    # write to log file
    log_file = 'log/RF_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.txt'
    f_log = open(log_file, 'w')
    f_log.write(classification_report(y_test, model.predict(X_test)))
    f_log.write("Accuracy: {}".format(acc))
    f_log.close()

    confusion_matrix = np.array(confusion_matrix(y_test, pred))

    # plot confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, index = [category for category in CLASS_NAME_TO_ID.keys()],
              columns = [category for category in CLASS_NAME_TO_ID.keys()])
    plt.figure(figsize = (20, 18))
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt="d", cmap="Blues")
    plt.title("Confusion matrix in Random Forest method")
    results_path = 'figure/RF_confusion_matrix.png'
    plt.savefig(results_path)