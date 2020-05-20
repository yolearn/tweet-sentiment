import pandas as pd
import spacy
from spacy.util import minibatch, compounding
import random
from pathlib import Path
from tqdm import tqdm
from cross_val import CrossValidation
import config

#NER Input format
TRAIN_DATA = [
        ("Uber blew through $1 million a week", {"entities": [(0, 4, "ORG")]}),
        ("Google rebrands its business apps", {"entities": [(0, 6, "ORG")]})]


def get_training_data(df):
    ner_data = []
    for index, row in df.iterrows():
        selected_text = row.selected_text
        text = row.text
        start = text.find(selected_text)
        end = start + len(selected_text)
        ner_data.append((text, {"entities": [[start, end, 'selected_text']]}))

    return ner_data

def train(trn_data, model=None, output_dir='../output', n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in trn_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in tqdm(range(n_iter)):
            random.shuffle(trn_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(trn_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            #print("Losses", losses)


    # save model to output directory
    # if output_dir is not None:
    #     output_dir = Path(output_dir)
    #     if not output_dir.exists():
    #         output_dir.mkdir()
    #     nlp.to_disk(output_dir)
    #     print("Saved model to", output_dir)

    return nlp

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text



if __name__ == "__main__":
    df = pd.read_csv('../input/train.csv')
    df = df[df.sentiment == 'positive']
    cv = CrossValidation(df, config.SPLIT_TYPE, config.SEED, config.NFOLDS, config.SHUFFLE)

    cv_score = []
    num = []
    for fold, (trn_idx, val_idx) in enumerate(cv.split()):
        trn_df = df.iloc[trn_idx]
        val_df = df.iloc[trn_idx]
        trn_df = get_training_data(trn_df)
        model = train(trn_df, model=None, n_iter=10)
        jaccard_score = 0
        for index, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
            text = row.text
            jaccard_score += jaccard(predict_entities(text, model), row.selected_text)
        
        print(f'fold {fold+1} is : {jaccard_score / val_df.shape[0]}') 
        cv_score.append(jaccard_score )
        num.append(val_df.shape[0])
    print(f"cv score is {sum(cv_score) / sum(num)}")

    df = pd.read_csv('../input/train.csv')
    df = df[df.sentiment == 'negative']
    cv = CrossValidation(df, config.SPLIT_TYPE, config.SEED, config.NFOLDS, config.SHUFFLE)
    
    for fold, (trn_idx, val_idx) in enumerate(cv.split()):
        trn_df = df.iloc[trn_idx]
        val_df = df.iloc[trn_idx]
        trn_df = get_training_data(trn_df)
        model = train(trn_df, model=None, n_iter=10)
        jaccard_score = 0
        for index, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
            text = row.text
            jaccard_score += jaccard(predict_entities(text, model), row.selected_text)
        
        print(f'fold {fold+1} is : {jaccard_score / val_df.shape[0]}') 
        cv_score.append(jaccard_score )
        num.append(val_df.shape[0])
    print(f"cv score is {sum(cv_score) / sum(num)}")
