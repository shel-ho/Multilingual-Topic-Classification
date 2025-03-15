from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# Top 5 most spoken languages and one lang from each United Nations geoscheme region
# excluding the Americas because none in this dataset is supported by XLM-R
# and Viet for fun and round up to 10
# - Oceania: Samoan
# - Asia 1 (W & C): Arabic
# - Asia 2 (S): Hindi, Bengali
# - Asia 3 (SE & E): Chinese (Traditional), Vietnamese
# - Europe 1 (N, W, S): English, Spanish
# - Europe 2 (E): Slovak
# - Africa: Afrikaans
langs = ['smo_Latn', 'afr_Latn', 'zho_Hant', 'arb_Arab', 'hin_Deva',
         'eng_Latn', 'slk_Latn', 'spa_Latn', 'ben_Beng', 'vie_Latn']
num_langs = len(langs)
#Constants
train_size = 701
dev_size = 99
test_size = 204
id2label = {0: 'entertainment', 1: 'geography', 2: 'health', 3: 'politics',
            4: 'science/technology', 5: 'sports', 6: 'travel'}
label2id = {'entertainment': 0, 'geography': 1, 'health': 2, 'politics':3,
            'science/technology': 4, 'sports':5, 'travel': 6}

class Model():
    def __init__(self):
        self.data = None
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.accuracy = evaluate.load("accuracy")

    #Load dataset
    def load_data(self):
        print("Loading Data")
        dsets = [load_dataset('Davlan/sib200', lang) for lang in langs]

        # Concatenate the datasets by split
        train_data = concatenate_datasets([dset['train'] for dset in dsets])
        dev_data = concatenate_datasets([dset['validation'] for dset in dsets])
        test_data = concatenate_datasets([dset['test'] for dset in dsets])

        # Combine split datasets back into one
        self.data = DatasetDict(dict(train=train_data,validation=dev_data,test=test_data))
        print("Data loaded")

    #Preprocess
    def preprocess_func(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def preprocess(self):
        self.data = self.data.map(self.preprocess_func, batched=True)
        self.data = self.data.class_encode_column("category")
        self.data = self.data.align_labels_with_mapping(label2id, "category")
        self.data = self.data.rename_column('category', 'labels')

    #Evaluate
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)

    #Train
    def train(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base", num_labels=7, id2label=id2label, label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir="my_xlmr_topic_classifier",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_steps = 100,
            report_to = 'none',
            label_names = ["labels"]
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

if __name__ == "__main__":
    neural = Model()
    neural.load_data()
    neural.preprocess()
    neural.train()