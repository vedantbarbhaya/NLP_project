import click
import torch
from trainers import RewardModelTrainer, AcceleratorRewardModelTrainer
from configs import get_configs
from gpt import GPTRewardModel
from dataset import DahoasRMStaticDataset


def train_accelerate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_configs("gpt2-xl")
    rm = GPTRewardModel.from_pretrained(cfg)
    cfg.batch_size = 1
    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=20,
                                     tokenizer_name="tiktoken/gpt2")
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=20,
                                    tokenizer_name="tiktoken/gpt2")
    trainer = AcceleratorRewardModelTrainer(cfg,
                                            device,
                                            rm,
                                            train_ds,
                                            test_ds,
                                            total_epochs=1,
                                            finetune_method=False)
    trainer.fit()


def train(pretrain, batch_size, exp_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_configs("gpt2-medium/lora")
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    cfg.total_epochs = 1
    cfg.exp_name = exp_name

    if pretrain == "huggingface":
        rm = GPTRewardModel.from_pretrained(cfg)
    else:
        rm = GPTRewardModel.from_backbone_checkpoint(cfg, pretrain)

    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=None,
                                     tokenizer_name="tiktoken/gpt2")
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=None,
                                    tokenizer_name="tiktoken/gpt2")
    trainer = RewardModelTrainer(cfg, device, rm, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s', default="naive")
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
def main(strategy, pretrain, batch_size, exp_name):
    torch.manual_seed(1234)

    if strategy == "accelerate":
        train_accelerate()
    elif strategy == "naive":
        train(pretrain, batch_size, exp_name)


if __name__ == "__main__":
    main()
