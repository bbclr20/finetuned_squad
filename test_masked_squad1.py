from model.masked_qa import BertForMaskedQuestionAnswering
from transformers import AutoConfig, AutoTokenizer
from train_masked_squad1 import load_and_cache_examples
import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, help="")
    parser.add_argument("--model_name_or_path", default="finetuned_squad", type=str, help="")
    parser.add_argument("--checkpoint", default="", type=str, help="")
    parser.add_argument("--train_file", default="train-v1.1.json", type=str, help="The input training file. If a data dir is specified, will look for the file there If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument("--predict_file", default="dev-v1.1.json", type=str, help="The input evaluation file. If a data dir is specified, will look for the file there If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument("--data_dir", default="./squad/squad1", type=str, help="The input evaluation file. If a data dir is specified, will look for the file there If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument("--max_seq_length", default=384, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.",)
    parser.add_argument("--doc_stride", default=128, type=int, help="When splitting up a long document into chunks, how much stride to take between chunks.",)
    parser.add_argument("--max_query_length", default=64, type=int, help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.",)
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    
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

    model = BertForMaskedQuestionAnswering.from_pretrained(args.model_name_or_path)
    # state_dict = torch.load(args.checkpoint)
    # model.load_state_dict(state_dict)
    model.eval()

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    data = eval_dataset[30]

    # check answer
    input_ids = data[0]
    attention_mask = data[1]
    token_type_ids = data[2]
    start_positions = data[3].item()
    end_positions = data[4].item()

    s, e = start_positions, end_positions
    ans = tokenizer.decode(input_ids[s:e+1])
    print(ans)
    
    res = model(input_ids=input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0), token_type_ids=token_type_ids.unsqueeze(dim=0), start_positions=[start_positions], end_positions=[end_positions])
    print(s, e)
    print(res.logits.shape)
    print(torch.where(res.logits.squeeze(-1)>0.5))