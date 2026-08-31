import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression




# IMPORTING RAW DATA INTO A TABLE 
'''starting with a plain python dictionary containing text sentences and their labels.
Tech recuiters call this "data ingestion'''

print("[System] Loading customer support tickets...")
data = {
    "ticket_text": [
        "My account is locked and I cannot log in",
        "Can I get a refund for my last invoice?",
        "The app crashes every time I open the dashboard",
        "Where can i change my billing payment method?",
        "Reset password link is not sending to my email"
    ],
    "category": ["Security" , "Billing", "Technical", "Billing", "Security"]
}

#turning the dictionry into rows-and-columns data table by using pd.DataFrame .
df = pd.DataFrame(data)




#PREPROCESSING AND DATA CLEANING
print("[System] Running basic data cleaning...")
df['ticket_text'] = df['ticket_text'].str.strip().str.lower()




#Feature Engineering & Machine Learning      
'''vectorizing the strings into numerical feature matrices '''
print("[System] Training the classification model...")

#converting text tokens into frequency scores based on how unique a word is using Tfidvectorizer
vectorizer = TfidfVectorizer()

#.fit_transform calculates the global word patterns and outputs the numeric training matrix (X)
X = vectorizer.fit_transform(df['ticket_text'])

#The variable Y holds the target categories (labels) that we want the model to predict
Y = df['category']

#LogisticRegression is a linear math classifier optimal for fast text classification sorting
model = LogisticRegression()

#.fit() exposes the model to the numbers (X) and answers(Y) so it can find historical links
model.fit(X,Y)
print("[System] Model training completed successfully.")




#REAL_TIME PRODUCTION ROUTING INFRENCE
new_email = "Hey, I need a refund on my credit card bill please"
clean_email = new_email.strip().lower()
vectorized_email = vectorizer.transform([clean_email])
prediction = model.predict(vectorized_email)

#Final string result showing which automated pipeline path the adta gets moved to
print("[System] Process complete. Ticket was successfully routed to department:" , prediction)
