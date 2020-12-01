import argparse
import torch
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process
import os
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from model.masked_qa import BertForMaskedQuestionAnswering

logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss = 0.0
    loss_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            model.zero_grad()

            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            print(f"Loss: {loss.item():.6f}, step: {step}/{len(train_dataloader)}")
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            global_step += 1

        print(f"Epoch: {epoch}, loss:{tr_loss/len(train_dataloader)}")
        loss_list.append(tr_loss/len(train_dataloader))
    print(f"loss_list: {loss_list}")
       
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate(args, model, tokenizer):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    total_loss = 0.0
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    print(f"Eval total loss: {total_loss/len(eval_dataloader)}")

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    input_dir = "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        processor = SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
        else:
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, help="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="./finetuned_squad/", type=str, help="The output directory where the model checkpoints and predictions will be written.",)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--max_seq_length", default=384, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.",)
    parser.add_argument("--doc_stride", default=128, type=int, help="When splitting up a long document into chunks, how much stride to take between chunks.",)
    parser.add_argument("--max_query_length", default=64, type=int, help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.",)
    parser.add_argument("--train_file", default="train-v1.1.json", type=str, help="The input training file. If a data dir is specified, will look for the file there If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument("--predict_file", default="dev-v1.1.json", type=str, help="The input evaluation file. If a data dir is specified, will look for the file there If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument("--data_dir", default="./squad/squad1", type=str, help="The input evaluation file. If a data dir is specified, will look for the file there If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--train_batch_size", type=int, default=8, help="multiple threads for converting example to features")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="multiple threads for converting example to features")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--num_train_epochs", type=int, default=2, help="multiple threads for converting example to features")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    # parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    # model = AutoModelForQuestionAnswering.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     cache_dir=None,
    # )
    
    model = BertForMaskedQuestionAnswering.from_pretrained(args.model_name_or_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # train
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    train(args, train_dataset, model, tokenizer)

    # eval
    evaluate(args, model, tokenizer)