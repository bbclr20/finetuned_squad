# finetuned_squad

## Download SQuAD

    download.sh

## Finetune model

Fine-tune the QA model with transformers. The answer in the paragraph of SQuAD is given by the start and end indices. 

    cd demo
    ./run_squad1.sh

Fine-tune the QA model with masked transformer. The answer in the paragraph of SQuAD is given by a mask which is labeled as 1.

    python3 train_masked_squad1.py
