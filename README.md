# Tweet Sentiment Extraction (Kaggle)
"My ridiculous dog is amazing." [sentiment: positive]

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

## File structure
.
├── src                 # Source file 
├── input               # Training data  
├── notebooks           # EDA
├── model               # Ouptut model, cross validation file, and score

## Evaluation Metric
### Jaccard score 
A Python implementation based on the links above, and matched with the output of the C# implementation on the back end, is provided below.

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

## Final Score
Public  score : 0.71691 (144/2227)
Private score : 0.71558 (625/2227)