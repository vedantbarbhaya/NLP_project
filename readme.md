Short description of what the code does:

1. main.py - 
   1. gpt: Loads a GPT model using predefined configurations and generates 10 text completions for a given prompt.
   2. unwrap_gpt: Loads a specific GPT model checkpoint, generates text completions, and saves a state dictionary with an inner (unwrapped) model state.
   3. gpt_sft: Loads a GPT model checkpoint, generates 10 text completions using the model trained with supervised fine-tuning (SFT).
   4. reward: Loads a GPT reward model, evaluates a given input prompt, and prints the model score for that prompt.
   5. reward_sft: Loads a GPT reward model from a fine-tuned checkpoint, evaluates a given prompt, and prints the model score.
   6. dataset: Loads the "Anthropic/hh-rlhf" dataset using the datasets library and prints the first training sample.
   7. test_loss: Tests a pairwise loss function on given scores to compute and print the loss.
   8. load_fsdp: Loads a GPT model using the Fully Sharded Data Parallel (FSDP) checkpoint.
   9. test_tokenizer: Tests different tokenizers, including TiktokenTokenizer and GPT2Tokenizer, to show how text is processed into tokenized tensors.

2. gpt.py
   1. MaskedMultiheadSelfAttention: Implements a multi-head self-attention layer with causal masking for GPT-style autoregressive text generation.
   2. FeedForwardNetworks: Implements the two-layer feed-forward network used in transformers.
      Expands the input dimension to a larger intermediate size and then projects it back down.
      Supports the use of LoRA layers for lightweight adaptation.
   3. TransformerDecoderBlock: Represents a single block in the transformer decoder. 
      Applies a self-attention layer and a feed-forward network, each followed by layer normalization and residual connections.
   4. TransformerDecoder: Implements a full decoder-only transformer model using multiple TransformerDecoderBlock instances.
   5. GPT:The main GPT-style language model that includes a TransformerDecoder architecture.
      Contains a final linear head for token predictions.
      Provides methods for autoregressive text generation (generate and batch_generate).
      Supports loading pre-trained checkpoints or Hugging Face models.
   6. HFGPTRewardModel: A reward model based on Hugging Face's GPT-2.
      Uses a value head to output a single score representing the reward.
   7. GPTRewardModel: Similar to HFGPTRewardModel but uses a custom GPT backbone.
      Provides methods to freeze weights, load from checkpoints, and manage adaptation through LoRA.
   8. GPTCritic: A critic model for reinforcement learning tasks based on the GPTRewardModel.
   9. GPTActor: A specialized version of the GPT model designed to act as an "actor" in reinforcement learning.
      Contains a forward_actor method to compute action log probabilities.

3. train_ppo.py
   1. Sets up the training configurations, loads actor and critic weights, and prepares the training dataset.
      Creates an instance of PPOTrainer to coordinate training with the provided models and data.

4. train_rm.py
   1. train_accelerate Function:
   This function configures a reward model training session using the AcceleratorRewardModelTrainer.
   Loads a pre-trained GPT model, prepares training and testing datasets, and fits the model using efficient training strategies.
   2. train Function:
   This function sets up the reward model training session based on user input for checkpoint loading and batch size.
   Uses the RewardModelTrainer to coordinate training of the GPT reward model.

5. train_sft.py 
   1.train class: Sets up the training configuration using parameters like batch size and pre-training source.
   Loads a pre-trained GPT model using the specified configuration.
   Prepares training and testing datasets.
   Creates an SFTTrainer instance and fits the model to the training data.

6. trainers.py
   1. trainer class - Base class that provides common utility functions for all trainers.
   2. PPOTrainer: Trains models using the PPO (Proximal Policy Optimization) algorithm.
   3. SFTTrainer: Trains models using supervised fine-tuning (SFT).
   4. RewardModelTrainer: Trains a reward model using pairwise loss.
   5. AcceleratorRewardModelTrainer: Trains a reward model using Hugging Face's Accelerate library for faster distributed training.

To test the final trained model, you can use this colab notebook - https://colab.research.google.com/drive/1bGVmT91qW221-GbtIMXq2ygpO9UVQCp7?usp=sharing
RUNNING THE CODE:
1. Install [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/#getting-started)
2. Install dependencies with
```bash
pip install -r requirements.txt
```

3. The first step is to traing a SFT model, inside `src` directory, run this command. You can change batch size via `-b`. The bigger VRAM you have the larger batch size you can afford.
```bash
python train_sft.py --n experiment_name -b 2`
```

4. Once you finished SFT stage, you can start to train the reward model. You should have a directory started with `sft_` in your `runs` directory. Find the final model weights and run this. This should start a reward model training for 1 epoch and generate a directory started with `rm_` with weights in it.
```bash
python train_rm.py -b 2 -n experiment_name -p "./runs/path/to/your/weights"
```

5. Finally, you can start the RLHF with the reward model and SFT model you get from previous two steps. Run this command. Because the training is not stable sometimes, I stopped early around 12K steps with a batch size of 1. The final weights will be in a directory started with `ppo_`.
```bash
python train_rm.py -b 2 -n experiment_name -a "./runs/path/to/sft/weights" -c "./runs/path/to/reward_model/weights" -s naive
```

6. ## Evaluate
0. You need to have an OpenAI account with credential key
1. Put your key into a file called "openai.key" JSON file. It should be a dictionary with a key called "OPENAI_API_KEY" and the value is your key.
2. Inside `src`, run this:
```bash
python evaluate.py -s "/path/to/sft/model" -p "/path/to/ppo/model"
```
