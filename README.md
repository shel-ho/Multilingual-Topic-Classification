# Multilingual Topic Classification 
## Author: Shel Ho
### Overview
This project performs the task of topic classification
on a subset of the languages offered in the [SIB-200](https://aclanthology.org/2024.eacl-long.14/)
dataset with two fully supervised models: logistic regression classifier and XLM-R language model. 
Both models achieved an accuracy level comparable to the results found in the SIB-200 paper. 
Specifically, logistic regression achieved **60%** accuracy while XLM-R achieved **85%**. 

### The Languages
| Language                       | Region   |  Language  | Region   | 
|--------------------------------|----------|------------|----------|
| Mandarin Chinese (Traditional) | Asia 3   | Afrikaans  | Africa   |
| Bengali                        | Asia 2   | Samoan     | Oceania  |
| English                        | Europe 1 | Slovak     | Europe 2 |
| Arabic (MSA)                   | Asia 1   | Bengali    | Asia 2   |
| Hindi                          | Asia 2   | Vietnamese | Asia 3   |


This rationale for the language selection for this multilingual project
was to have both high and low resourced languages with various scripts
across all United Nation regions (excluding the Americas because none
in the dataset is supported by XLM-R). 

### The Models
The two models used to perform the task are each in their own .py file, and each file loads the data,
trains, and evaluates. 
- **classical-model.py**: Implementation of Logistic Regression classifier
with scikit-learn
- **neural-model.py**: Implementation of the fine-tuned XLM-R language model