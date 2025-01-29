from os import lseek
#Libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#-------------------------------------------SETTING UP AND CLEANING THE DATA------------------------------------


csvFile = pd.read_csv('spam_ham_dataset.csv')


#print(csvFile.shape)
# Checking for missing values using isnull()


df = csvFile.dropna(axis=0, how='any')
#print(df.shape) #dropping no values no issues

missing_values = df.isnull()

for column in range(3):
    for row in range(5170):
        if missing_values.iloc[row][column] == True:
            print(df.iloc[column])
#No missing values double checked it
#print(df.head)

X = df['text']
y = df['label_num']

#print(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train.shape, X_test.shape)

def TextPreprocessing(df, n):
    filtered_sentence = []
    nltk.download('stopwords')
    nltk.download('punkt_tab')


    #If you have issues with some useless work change this
    #Anyone should feel free to add more
    manual_stop_words = {'subject', ':','','.', ',','/',';','"','-',"'", '(',')','_'}
    #should the following be added: '^', ...



    stop_words = set(stopwords.words('english'))
    for i in range(n):
        temp = []
        for w in word_tokenize(df.iloc[i].lower()):

            if w not in stop_words and w not in manual_stop_words:
                temp.append(w)
        #print(temp)
        filtered_sentence.append(temp)
    #print(filtered_sentence[1:3])
    return filtered_sentence


X_train_token = TextPreprocessing(X_train, X_train.size)
X_test_token = TextPreprocessing(X_test, X_test.size)


#print example if needed to understand the data
#print('Token:   ', X_train_token[:2])
#print('Training data X(What we are using to predcit):    ', X_train[:2])
#print('training data y(what we try to predict)', y_train[:2])




#vectorization of tokenize data
vect = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, lowercase=False)
xTrain = vect.fit_transform(X_train_token)
xTest = vect.transform(X_test_token)

#converting to array for GNB use
xTrainArray = xTrain.toarray()
xTestArray = xTest.toarray()

GNB = GaussianNB()
GNB.fit(xTrainArray, y_train)
test = GNB.predict(xTestArray)

#accuracy
accs = accuracy_score(y_test, test)
print()
print(f"Accuracy of GNB = {accs:.2f}")
print()

#confusion matrix
print("confusion matrix - ")
print(confusion_matrix(y_test, test))
print()

#precision, recall, f1=score
print(classification_report(y_test, test))

