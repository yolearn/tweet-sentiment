import tokenizers
TRAIN_FILE = '../input/train.csv'
DEVICE = 'cuda'                    
import transformers

MAX_LEN = 128
MODEL_VERSION = 'roberta-base'
BATCH_SIZE = 32
MODEL_PATH = f'../input/{MODEL_VERSION}/'
MODEL_CONF = transformers.RobertaConfig.from_pretrained(f'../input/{MODEL_VERSION}/config.json')
MODEL_CONF.output_hidden_states = True

EMBEDDING_SIZE = 768
CNN_OUTPUT_CHANNEL = 1
CNN_KERNEL_WIDTH = 1
DROPOUT_RATE = 0.1
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
                vocab_file=f"../input/roberta-base-squad2/vocab.json",
                merges_file=f"../input/roberta-base-squad2/merges.txt",
                lowercase=True,
                add_prefix_space=True
            )

                
