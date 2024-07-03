"""## 1.2. Preprocess data"""

# Load Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
!pip install -U -q emoji
import emoji
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

"""*You are required to describe which data preprocessing techniques were conducted with justification of your decision. *

### Method Justification
The preprocessing method is in the last cell for this section. It is a function that applies to both the train and test data. As this is just text cleaning and not building any models, the steps are similar. 

The steps and justifications are as follows:
1. **Remove Random Characters**

There was a character that was appearing due to encoding. The easiest way to handle it was to remove the character. Other encoding options such as 'utf-8' or 'utf-32' did not fix the issue. 

2. **Remove Mentions**

The Twitter mentions provided no benefit to our analysis of *sentiment*. As such, there was no need to keep them. 

3. **Remove URLs**

Similarly to the Twitter mentions, URLs provide no influence on the sentiment of a Twitter post. As such, they do not need to be in our analysis. 

4. **Encode [Emoji's and Smileys](https://colab.research.google.com/drive/1JY12_FCF2WkTHqm8VHcWtISN0u3kiUfB#scrollTo=OVBrjMWhBhkQ&line=5&uniqifier=1)**

Once the unnecessary strings have been removed, we can then encode the smiley's into valuable information. The smiley in the text can represent positive and negative emotions. For example, the smiley face, :), can be a positive emotion, and a sad face, :(, be a negative emotion. 
The code and dictionary of smiley's were created before watching the lecture on handling Emoji and  Smiley through NLTK. However, double-checking later, the custom dictionary for cleaning the smiley's was preferred due to the 'double punctuations' and the smiley 'nose'. 

The double punctuation was when punctuation was repeated more than once. An example would be a smiley face with many mouths such as, :)))))))). This smiley would still be the same smiley face, however, extenuated to be very happy. The smiley nose was a '-', which was not encoding correctly. Therefore it was easier to remove the nose and then convert the smiley. 

The Emoji's were cleaned and encoded using the Emoji package. 

4. **Remove Numbers**

Numbers were removed after cleaning the smileys and Emojis. Some numbers were used to represent some smileys such as love, <3. Therefore, we would lose valuable information by removing the numbers. The decision to remove numbers was because they do not add to our sentiment. 

5. **Remove Punctuations**

Similar to numbers, punctuations added no benefit to sentiment. They also had to be removed after correcting for smileys.

6. **Remove Hashtags**

Similar to numbers and punctuations, hashtags added no benefit to sentiment. 

7. **Make to Lowercase**

All the remaining text was transformed into lowercase. The lowercase transformation would then mean that when we tokenise, similar words are not considered different. For example, 'I AM SAD' and 'I am sad' imply the same negative emotion. In contrast, the task at hand does not focus on the magnitude of emotion but rather the direction of emotion. 

8. **Remove Stopwords and Lemmantise Words**

Finally, the stop words, which again add nothing to sentiment, were filtered out, and the remaining words were subject to lemmatisation. Lemmatisation was prefered to stemming, as the stemming would create words to their root that would not be in use for English. Lemmatisation would also handle the slang better than stemming, which is relevant for a dataset such as Twitter that contains primarily spoken text.
"""

train_dataframe = pd.DataFrame(training_data) # Make it like a dataframe that can be easily accessed
test_dataframe = pd.DataFrame(testing_data)

train_labels = train_dataframe[0] # From the Code in 1.1 it appears the first column has the label. This can be double checked by printing the head later
train_text = train_dataframe[1] # And the second column is the twitter text
test_labels = test_dataframe[0]
test_text = test_dataframe[1]

print("------------------------------------")
print(f"Size of training labels dataset: {len(train_labels)}")
print(f"Size of training text dataset: {len(train_text)}")
print(f"Size of test labels dataset: {len(test_labels)}")
print(f"Size of test text dataset: {len(test_text)}")
print("------------------------------------")

print("------------------------------------")
print("Check if this is the same as the Sample Data in 1.1")
print("LABEL: {0} / SENTENCE: {1}".format(train_labels[0], train_text[0]))
print("------------------------------------")

"""#### Smileys and Emojis"""

# This website had a great source for emoticons and empjis: https://towardsdatascience.com/twitter-sentiment-analysis-using-fasttext-9ccd04465597, can also find a list on wikipedia https://en.wikipedia.org/wiki/List_of_emoticons


# emoticons
load_dict_smileys = {
        ":‑)":"smiley", ":-]":"smiley", ":-3":"smiley", ":->":"smiley", "8-)":"smiley", ":-}":"smiley", ":)":"smiley", ":]":"smiley", ":3":"smiley", ":>":"smiley", "8)":"smiley", ":}":"smiley", ":o)":"smiley", ":c)":"smiley", ":^)":"smiley", "=]":"smiley", "=)":"smiley", ":-))":"smiley", ":‑D":"smiley", "8‑D":"smiley", "x‑D":"smiley", "X‑D":"smiley", ":D":"smiley", "8D":"smiley", "xD":"smiley", "XD":"smiley", 
        ":‑(":"sad", ":(":"sad", ":‑c":"sad", ":‑<":"sad", ":‑[":"sad", ":(":"sad", ":c":"sad", ":<":"sad", ":[":"sad", ":-||":"sad", ">:[":"sad", ":{":"sad", ":@":"sad", ">:(":"sad", ":'‑(":"sad", ":'(":"sad",
        ":‑P":"playful", "X‑P":"playful", "x‑p":"playful", ":‑p":"playful", ":‑Þ":"playful", ":‑þ":"playful", ":‑b":"playful", ":P":"playful", "XP":"playful", "xp":"playful", ":p":"playful", ":Þ":"playful", ":þ":"playful", ":b":"playful", 
        "<3":"love"
        }


def clean_emoji_text(x):
    # Deal with emoticons source: https://en.wikipedia.org/wiki/List_of_emoticons
    # Get rid of duplication of string
    regex = r"([\(\)])\1+" # Adpated from https://stackoverflow.com/a/56961453/8692659
    subst = "\\1"
    x = re.sub(regex, subst, x, 0)
    x = re.sub(r"-","",x) # Somehow was not picking up the nose properly. 

    SMILEY = load_dict_smileys  
    words = x.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    x = " ".join(reformed)
    
    # Deal with emojis
    x = emoji.demojize(x, delimiters=("",""))

    return x

"""#### Punctuation"""

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def remove_punctuation(x): # Adapted from [Lab 5]
    x = str(x)
    for punct in puncts:
        if punct in x:

            x = x.replace(punct, '')
    return x

"""#### Lemmatisation"""

lemmatizer = WordNetLemmatizer()
def remove_stopwords_and_lemantize(x) :  # Adapted from [Lab 5]
  tokens = word_tokenize(x)

  stop_words = sw.words()
  filtered_sentence = [w for w in tokens if not w in stop_words]
  lemantized_sentence = [lemmatizer.lemmatize(word) for word in filtered_sentence]
  x = " ".join(lemantized_sentence)

  return x

"""#### Clean Train and Test Data"""

def clean_text(x): # Adapted from [Lab 5]
    x = str(x)
    # print("Remove Random Characters")
    x = re.sub("\u3000", "", x) # Remove this random character that pops up
    # print("Remove mentions")
    x = re.sub("@[A-Za-z0-9]+","", x) # How to remove mentions, adapted from https://stackoverflow.com/a/54734032/8692659
    # print("Remove URLS")
    x = re.sub(r"(?:\@|https?\://)\S+", "", x) # Removing URL links, adapted from https://stackoverflow.com/a/13896637/8692659 
    # print("Encode Emoji and Smileys")
    x = clean_emoji_text(x) # The emoji has useful information for sentiment
    # print("Remove Numbers")
    x = re.sub("[0-9]+","", x) # Remove numbers, they don't give us any extra information
    # print("Remove Punctuations")
    x = remove_punctuation(x)
    # print("Remove Hashtags")
    x = re.sub("#[A-Za-z0-9]+","", x) # Remove Hastags
    # print("Make to lowercase")
    x = x.lower()
    # print("Remove stopwords and Lemantise words")
    x = remove_stopwords_and_lemantize(x)
    # print("Finished Cleaning")

    return x

# This process can be time consuming
print("------------------------------------")
print("Cleaning Training Data")
train_text_clean = [clean_text(i) for i in train_text]
print("------------------------------------")
print("Cleaning Testing Data")
test_text_clean = [clean_text(i) for i in test_text]
print("------------------------------------\nFinished Cleaning")
