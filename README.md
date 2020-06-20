# Tweet Sentiment Extraction (Kaggle)
"My ridiculous dog is amazing." [sentiment: positive]

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

## Main file
- conig.py                  # fast testing config
- cross_val.py              # cross validation
- dataload.py               # torch dataloader 
- engine.py                 # loop for batch 
- error_analyze.py          # error analyzer
- inference.py              # inference testing set
- model.py                  # basic model
- moldel_gcnn.py            # gcnn structure model
- model_linear.py           # linear gcnn structure
- ner.py                    # ner testing (from another kaggle)
- sentencepiece_pb2.py      # utils for tokenizer
- train.py                  # train model
- unsupervise.py            # amazing unsupervise solution (from another kaggler) 
- utils.py                  # utils


## Evaluation Metric
### Jaccard score 
A Python implementation based on the links above, and matched with the output of the C# implementation on the back end, is provided below.

![Jaccard]()

## Final Score
- Public  score : 0.71691 (144th/2227)
= Private score : 0.71558 (625th/2227)