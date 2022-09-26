.PHONY: all

ifndef INPUT_FILE
INPUT_FILE = ../data/dev.json
endif

ifndef OUTPUT_DIR
OUTPUT_DIR = predictions/
endif

all: bert roberta gpt2 xlnet sentiment
bert: bert-base-cased bert-large-cased
roberta: roberta-base roberta-large 
xlnet: xlnet-base-cased xlnet-large-cased
gpt2: gpt2-small gpt2-medium gpt2-large

bert-base-cased:
	python3 eval_discriminative_models.py --pretrained-class bert-base-cased --tokenizer BertTokenizer --intrasentence-model BertLM --intersentence-model BertNextSentence --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

bert-large-cased:
	python3 eval_discriminative_models.py --pretrained-class bert-large-cased --tokenizer BertTokenizer --intrasentence-model BertLM --intersentence-model BertNextSentence --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

roberta-base:
	python3 eval_discriminative_models.py --pretrained-class roberta-base --tokenizer RobertaTokenizer --intrasentence-model RoBERTaLM --intersentence-model ModelNSP --intersentence-load-path models/pretrained_models/RobertaModel_roberta-base_1e-05.pth --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

roberta-large:
	python3 eval_discriminative_models.py --pretrained-class roberta-large --tokenizer RobertaTokenizer --intrasentence-model RoBERTaLM --intersentence-model ModelNSP --intersentence-load-path models/pretrained_models/RobertaModel_roberta-large_1e-05.pth --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

xlnet-base-cased:
	python3 eval_discriminative_models.py --pretrained-class xlnet-base-cased --tokenizer XLNetTokenizer --intrasentence-model XLNetLM --intersentence-model ModelNSP --intersentence-load-path models/pretrained_models/XLNetModel_xlnet-base-cased_1e-05.pth --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

xlnet-large-cased:
	python3 eval_discriminative_models.py --pretrained-class xlnet-large-cased --tokenizer XLNetTokenizer --intrasentence-model XLNetLM --intersentence-model ModelNSP --intersentence-load-path models/pretrained_models/XLNetModel_xlnet-large-cased_1e-05.pth --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

gpt2-small:
	python3 eval_generative_models.py --pretrained-class gpt2 --intrasentence-model GPT2LM --intersentence-model ModelNSP --tokenizer GPT2Tokenizer --max-seq-length 128 --intersentence-load-path models/pretrained_models/GPT2Model_gpt2_0.0005.pth --batch-size 1 --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

gpt2-medium:
	python3 eval_generative_models.py --pretrained-class gpt2-medium --intrasentence-model GPT2LM --intersentence-model ModelNSP --tokenizer GPT2Tokenizer --max-seq-length 128 --intersentence-load-path models/pretrained_models/GPT2Model_gpt2-medium_0.0005.pth --batch-size 1 --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

gpt2-large:
	python3 eval_generative_models.py --pretrained-class gpt2-large --intrasentence-model GPT2LM --intersentence-model ModelNSP --tokenizer GPT2Tokenizer --max-seq-length 128 --intersentence-load-path models/pretrained_models/GPT2Model_gpt2-large_1e-05.pth --batch-size 1 --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

gpt2-xl: 
	python3 eval_generative_models.py --pretrained-class gpt2-xl --intrasentence-model GPT2LM --intersentence-model ModelNSP --tokenizer GPT2Tokenizer --max-seq-length 128 --intersentence-load-path models/pretrained_models/GPT2Model_gpt2-xl_5e-06-0.0001-0.93.pth --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS) 

sentiment:
	python3 eval_sentiment_models.py --load-path models/pretrained_models/SentimentBert.pth --input-file $(INPUT_FILE) --output-dir $(OUTPUT_DIR) $(FLAGS)

ensemble:
	python3 eval_ensemble.py --gold-file $(INPUT_FILE) --predictions-dir $(OUTPUT_DIR) --output-file $(OUTPUT_DIR)/predictions_EnsembleModel_.json 
