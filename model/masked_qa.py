from transformers import BertPreTrainedModel, BertModel, BertForQuestionAnswering
from torch import nn
from torch.nn import BCELoss
import torch


class MaskedQuestionAnswering:
    
    def __init__(self, loss, logits, targets):
        self.loss = loss
        self.logits = logits
        self.targets = targets

    # def __init__(self, loss, logits, targets, hidden_states, attentions):
    #     self.loss = loss
    #     self.logits = logits
    #     self.targets = targets
    #     self.hidden_states = hidden_states
    #     self.attentions = attentions

class BertForMaskedQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, 1)
        self.activation = nn.Sigmoid()
        self.init_weights()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="bert-base-uncased",
    #     output_type=QuestionAnsweringModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None,
        end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None,):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        masked_answer = logits.squeeze(-1)
        masked_answer = self.activation(masked_answer)

        loss_fct = BCELoss()
        targets = torch.empty_like(masked_answer)
        for n, (s, e) in enumerate(zip(start_positions, end_positions)):
            targets[n][s:e+1] = 1
        loss = loss_fct(masked_answer, targets)
  
        return MaskedQuestionAnswering(
            loss=loss,
            logits=logits,
            targets=targets,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

if __name__ == "__main__":
    from os import sys
    sys.path.append("..")
    from train_masked_squad1 import load_and_cache_examples
    from transformers import AutoTokenizer
    import argparse

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default="bert", type=str, help="distilbert")
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
    parser.add_argument("--data_dir", default="../squad/squad1", type=str, help="The input evaluation file. If a data dir is specified, will look for the file there If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--train_batch_size", type=int, default=4, help="multiple threads for converting example to features")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedQuestionAnswering.from_pretrained("bert-base-uncased", return_dict=True)
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)

    data = train_dataset[3]
    input_ids, attention_mask, token_type_ids, start_positions, end_positions = data[:5]
    res = model(input_ids=input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0), token_type_ids=token_type_ids.unsqueeze(dim=0), start_positions=[start_positions], end_positions=[end_positions])