import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Streamlit GUI
st.title('Sentiment Analysis')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        # Load and preprocess data
        data = pd.read_csv(uploaded_file)
        columns_to_drop = ['Timestamp', 'ID', 'User', 'Source', 'Topic', 'Country', 'Year', 'Month', 'Day', 'Hour', 'Retweets', 'Likes']
        data = data.drop(columns=columns_to_drop)

        # Initialize NLTK resources
        nltk.download('stopwords')
        nltk.download('wordnet')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Define function to expand contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot", "i'm": "i am", 
            "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is",
            "we're": "we are", "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'd": "i would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would",
            "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "hadn't": "had not", "doesn't": "does not", "don't": "do not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "shan't": "shall not", "shouldn't": "should not",
            "can't": "cannot", "couldn't": "could not", "mustn't": "must not", "mightn't": "might not",
            "needn't": "need not"
        }
        
        def expand_contractions(text, contractions_dict):
            pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), flags=re.IGNORECASE|re.DOTALL)
            def replace(match):
                return contractions_dict[match.group(0).lower()]
            return pattern.sub(replace, text)
        
        # Function for text preprocessing
        def preprocess_text(text):
            # Remove HTML tags
            text = re.sub(r'<.*?>', ' ', text)
            # Expand contractions
            text = expand_contractions(text, contractions)
            # Remove non-alphabetical characters and convert to lowercase
            text = re.sub('[^a-zA-Z]', ' ', text).lower()
            # Tokenize text
            words = text.split()
            # Remove stopwords and lemmatize
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            # Join words back into a single string
            return ' '.join(words)
        
        # Apply text preprocessing
        data['Text'] = data['Text'].apply(preprocess_text)
        data.drop_duplicates(inplace=True)

        # Preprocess labels
        data['Sentiment (Label)'] = data['Sentiment (Label)'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x).lower())

        # Define keyword sets for sentiment classification
        positive_keywords = {'positive', 'happiness', 'joy', 'love', 'amusement', 'enjoyment', 'admiration', 'excitement', 'kind', 'pride', 'gratitude', 'hope', 'empowerment', 'arousal', 'enthusiasm', 'hopeful', 'proud', 'grateful', 'free', 'inspired', 'overjoyed', 'inspiration', 'motivation', 'joyfulreunion', 'satisfaction', 'blessed', 'optimism', 'enchantment', 'playfuljoy', 'dreamchaser', 'thrill', 'creativity', 'adventure', 'euphoria', 'festivejoy', 'freedom', 'artisticburst', 'marvel', 'positivity', 'kindness', 'friendship', 'success', 'amazement', 'celebration', 'charm', 'ecstasy', 'iconic', 'engagement', 'touched', 'heartwarming', 'renewed effort', 'thrilling journey', 'celestial wonder', 'creative inspiration', 'runway creativity', 'relief', 'happy', 'elation', 'contentment', 'reverence', 'dazzle'}
        negative_keywords = {'negative', 'anger', 'fear', 'sadness', 'disgust', 'awe', 'disappointment', 'bitterness', 'shame', 'despair', 'grief', 'loneliness', 'jealousy', 'resentment', 'frustration', 'boredom', 'anxiety', 'intimidation', 'helplessness', 'envy', 'regret', 'melancholy', 'exhaustion', 'sorrow', 'darkness', 'desperation', 'desolation', 'heartbreak', 'overwhelmed', 'devastated', 'betrayal', 'suffering', 'isolation', 'suspense'}

        # Classify sentiment based on keywords
        def classify_sentiment(label):
            label_words = set(label.split())
            if label_words.intersection(positive_keywords):
                return 'positive'
            elif label_words.intersection(negative_keywords):
                return 'negative'
            return 'neutral'
        
        data['Sentiment_Class'] = data['Sentiment (Label)'].apply(classify_sentiment)

        # Prepare data for model training
        X = data['Text']
        y = data['Sentiment_Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=25)

        # Convert text data to TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Train Naive Bayes model
        nb_model = MultinomialNB()
        nb_model.fit(X_train_tfidf, y_train)
        y_pred_nb = nb_model.predict(X_test_tfidf)
        accuracy_nb = accuracy_score(y_test, y_pred_nb)

        # Hyperparameter tuning for Logistic Regression
        logreg_params = {'C': [0.01, 0.1, 1, 10, 100]}
        logreg_model = GridSearchCV(LogisticRegression(max_iter=1000), logreg_params, cv=5)
        logreg_model.fit(X_train_tfidf, y_train)
        y_pred_logreg = logreg_model.predict(X_test_tfidf)
        accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

        # Hyperparameter tuning for SVM
        svm_params = {'C': [0.01, 0.1, 1, 10, 100]}
        svm_model = GridSearchCV(SVC(kernel='linear'), svm_params, cv=5)
        svm_model.fit(X_train_tfidf, y_train)
        y_pred_svm = svm_model.predict(X_test_tfidf)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)

        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=25)
        rf_model.fit(X_train_tfidf, y_train)
        y_pred_rf = rf_model.predict(X_test_tfidf)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)

        # Streamlit Dashboard
        st.header('Model Evaluation')
        selected_model = st.selectbox("Select Model", ["Naive Bayes", "Logistic Regression", "SVM", "Random Forest"])

        if selected_model == "SVM":
            st.write(f'Accuracy: {accuracy_svm}')
            st.subheader('Classification Report')
            st.text(classification_report(y_test, y_pred_svm))
            st.subheader('Test Set Predictions')
            results_df = pd.DataFrame({'Text': X_test, 'Actual': y_test, 'Predicted': y_pred_svm})
            st.write(results_df)
            model = svm_model

        elif selected_model == "Naive Bayes":
            st.write(f'Accuracy: {accuracy_nb}')
            st.subheader('Classification Report')
            st.text(classification_report(y_test, y_pred_nb))
            st.subheader('Test Set Predictions')
            results_df = pd.DataFrame({'Text': X_test, 'Actual': y_test, 'Predicted': y_pred_nb})
            st.write(results_df)
            model = nb_model

        elif selected_model == "Logistic Regression":
            st.write(f'Accuracy: {accuracy_logreg}')
            st.subheader('Classification Report')
            st.text(classification_report(y_test, y_pred_logreg))
            st.subheader('Test Set Predictions')
            results_df = pd.DataFrame({'Text': X_test, 'Actual': y_test, 'Predicted': y_pred_logreg})
            st.write(results_df)
            model = logreg_model

        elif selected_model == "Random Forest":
            st.write(f'Accuracy: {accuracy_rf}')
            st.subheader('Classification Report')
            st.text(classification_report(y_test, y_pred_rf))
            st.subheader('Test Set Predictions')
            results_df = pd.DataFrame({'Text': X_test, 'Actual': y_test, 'Predicted': y_pred_rf})
            st.write(results_df)
            model = rf_model

        else:
            st.write("Please select a valid model")

        # Textbox for user input
        st.header('Predict Sentiment for a Custom Sentence')
        user_input = st.text_input("Enter a sentence for sentiment prediction")
        if st.button("Predict"):
            if user_input:
                # Preprocess user input
                user_input_clean = preprocess_text(user_input)
                user_input_tfidf = tfidf_vectorizer.transform([user_input_clean])

                # Predict sentiment
                prediction = model.predict(user_input_tfidf)
                st.write(f'Sentiment: {prediction[0]}')
            else:
                st.write("Please enter a sentence for prediction")

    except Exception as e:
        st.write("Error:", e)
