import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv')

# Ordinal feature encoding
# It follows the approach used in https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# The purpose of this example is to predict the species of the penguin
# If you would like to predict sex, the target would have to be changed to sex
# The original approach from Pratik didn't use species as target

df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Here we will apply a function to encode
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and Y
X = df.drop('species', axis=1)
Y = df['species']

# Build a random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
# THe model is used as input to the function

pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
