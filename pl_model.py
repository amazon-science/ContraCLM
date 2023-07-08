import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from deepspeed.ops.adam import FusedAdam
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

from utils import get_inputs_and_labels, load_model_and_tokenizer


class LitContraCLM(pl.LightningModule):
    def __init__(self, trainer_args, loss_func_tok=None, loss_func_seq=None, 
                 loss_func_tok_word=None, num_nodes=1):
        super(LitContraCLM, self).__init__()
        self.save_hyperparameters(trainer_args)
        # Load Model and Tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            trainer_args.model_name, 
            pad_token_id=trainer_args.pad_token_id,
            dropout_layers=trainer_args.dropout_layers,
            dropout_p=trainer_args.dropout_p,
            functional_dropout=trainer_args.functional_dropout
        )
        self.trainer_args = trainer_args
        self.loss_func_tok = loss_func_tok
        self.loss_func_seq = loss_func_seq
        self.mle_loss = torch.nn.CrossEntropyLoss()
        self.vocab_size = self.model.config.vocab_size
        self.embed_dim = self.model.config.hidden_size
        self.num_nodes = num_nodes


    def setup(self, stage):
        if stage == 'fit':
            # Hyperparamters and Configuration
            self.dropout_p = self.trainer_args.dropout_p
            self.functional_dropout = self.trainer_args.functional_dropout
            self.pad_token_id = self.trainer_args.pad_token_id

            self.lr = self.trainer_args.lr
            self.weight_decay = self.trainer_args.weight_decay
            self.num_warmup_steps = self.trainer_args.warmup_steps
            self.num_epochs = self.trainer_args.max_epochs
            self.train_batch_size = self.trainer_args.train_batch_size
            self.num_train_examples = self.trainer_args.num_training_examples
            self.num_gpu_per_node = self.trainer_args.devices
            self.accumulate_grad_batches = self.trainer_args.accumulate_grad_batches

            if self.trainer_args.max_steps == -1:
                num_steps_per_epoch = self.num_train_examples // (self.num_gpu_per_node * self.num_nodes * self.accumulate_grad_batches)
                self.num_training_steps = self.num_epochs * num_steps_per_epoch
                print(f"steps_per_epoch: {num_steps_per_epoch}\t total_training_steps: {self.num_training_steps}.")
            else:
                self.num_training_steps = self.trainer_args.max_steps

            self.no_scheduling = self.trainer_args.no_scheduling
            self.world_size = self.trainer_args.devices * self.num_nodes
            # Loss Configuration
            self.loss = self.trainer_args.loss
            assert self.loss in ["MLE_Only", "ContraCLM", "ContraCLMTok", "ContraCLMSeq"], \
                f"Loss: `{self.loss}` is not supported!"


    def forward(self, input_ids, attention_mask=None):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        return logits, outputs.hidden_states


    def training_step(self, batch, batch_idx):
        token_ids = batch['input_ids']
        input_ids, labels, attention_mask = get_inputs_and_labels(
            token_ids, pad_token_id=self.pad_token_id, mask_pad=True
        )
        uniq_tokens = torch.unique(input_ids)
        all_tokens = torch.sum(attention_mask)
        self.log("all_tokens_per_gpu", all_tokens, sync_dist=True)
        self.log("unique_tokens_per_gpu", len(uniq_tokens), sync_dist=True)

        # first forward pass
        logits, hidden_states = self(input_ids, attention_mask=attention_mask)
        last_hidden_states = hidden_states[-1]

        # compute the MLE loss on all devices independently
        loss = self.mle_loss(logits.view(-1, self.vocab_size), labels.view(-1))
        self.log("Train/Loss/MLE", loss, sync_dist=True, on_step=True, prog_bar=True)

        # Original MLE
        if self.loss == "MLE_Only":
            return loss

        # get the dropout based augmentation either via the second forwarding pass or functional dropout
        if self.functional_dropout:
            last_hidden_states_orig = last_hidden_states
            last_hidden_states = F.dropout(last_hidden_states_orig, p=self.dropout_p)
            last_hidden_states_2 = F.dropout(last_hidden_states_orig, p=self.dropout_p)
        else:
            _, hidden_states_2 = self(input_ids, attention_mask=attention_mask)
            last_hidden_states_2 = hidden_states_2[-1]

        # Token-level loss
        if self.loss == "ContraCLMTok" or self.loss == "ContraCLM":
            loss_tok = self.loss_func_tok(last_hidden_states, last_hidden_states_2, attention_mask)
            loss += loss_tok
            self.log(f"Train/Loss/TokCL", loss_tok, sync_dist=True, on_step=True, prog_bar=True)

        # Sequence-level loss
        if self.loss == "ContraCLMSeq" or self.loss == "ContraCLM":
            # We use all_gather to gather representations from all GPUs. Since all_gather results are not part of
            # computational graph, we replace the current process's corresponding embeddings with original tensors
            if self.world_size > 1:
                all_attention_mask = self.all_gather(attention_mask).flatten(start_dim=0, end_dim=1)
                all_hidden_feature_1 = self.all_gather(last_hidden_states)
                all_hidden_feature_1[self.global_rank] = last_hidden_states
                all_hidden_feature_1 = all_hidden_feature_1.flatten(start_dim=0, end_dim=1)

                all_hidden_feature_2 = self.all_gather(last_hidden_states_2)
                all_hidden_feature_2[self.global_rank] = last_hidden_states_2
                all_hidden_feature_2 = all_hidden_feature_2.flatten(start_dim=0, end_dim=1)
            else:
                all_attention_mask = input_ids
                all_hidden_feature_1 = last_hidden_states
                all_hidden_feature_2 = last_hidden_states_2
            loss_seq = self.loss_func_seq(all_hidden_feature_1, all_hidden_feature_2, 
                                          all_attention_mask)
            loss += loss_seq
            self.log(f"Train/Loss/SeqCL", loss_seq, rank_zero_only=True, on_step=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        eval_fct = torch.nn.CrossEntropyLoss()
        token_ids = batch['token_ids']
        input_ids, labels, attention_mask = get_inputs_and_labels(
            token_ids, pad_token_id=self.pad_token_id, mask_pad=True
        )
        logits, _ = self(input_ids, attention_mask=attention_mask)
        loss = eval_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        return loss


    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack(validation_step_outputs).mean()
        perplexity = torch.exp(val_loss)
        self.log("Valid/Loss/MLE", val_loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("Valid/Loss/Perplexity", perplexity, sync_dist=True, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optim_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = FusedAdam(optim_groups, lr=self.lr)
        # optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.no_scheduling:
            return optimizer
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.num_warmup_steps,
                                                    num_training_steps=self.num_training_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
