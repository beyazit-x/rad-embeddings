![Rad Emeddings Logo](https://rad-embeddings.github.io/assets/logo.svg)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://rad-embeddings.github.io/assets/splash.svg">
  <img alt=Rad Emeddings overview" src="https://rad-embeddings.github.io/assets/splash_light.png">
</picture>

This repository contains the code for the paper: [**Compositional Automata Embeddings for Goal-Conditioned Reinforcement Learning**](https://rad-embeddings.github.io/), to appear in NeurIPS 2024.

## Setup

To set up the repository, follow the steps given below.

```bash
# Create a new conda environment with Python 3.9.19
conda create --name rad-embeddings python=3.9.19

# Activate the environment
conda activate rad-embeddings

# Downgrade pip to version 24.0
pip install pip==24.0

# Install requirements
pip install -r requirements.txt

# Install Safety Gym
pip install -e src/envs/safety/safety-gym/
```

## Pretraining
To pretrain a GNN, run the command given below. This command runs the pretraining in a dummy MDP enviroment called `Dummy-MDP-v0` and saves it to the `storage` directory. It trains a GATv2 model, but if the `--gnn` flag is not given, then it uses an RGCN model. To train the model, it samples RAD cDFAs. The sampling space can be changed through the `--dfa-sampler` flag. See below for other sampler options.

```
python train_agent.py \
	--algo ppo \
	--env Dummy-MDP-v0 \
	--log-interval 1 \
	--save-interval 20 \
	--frames-per-proc 512 \
	--batch-size 1024 \
	--frames 10000000 \
	--dumb-ac \
	--discount 0.9 \
	--dfa-sampler CompositionalReachAvoidDerived \
	--lr 0.001 \
	--clip-eps 0.1 \
	--gae-lambda 0.5 \
	--epochs 2 \
	--gnn GATv2Conv \
	--dfa
```

### Evaluation
To evaluate a pretrained GNN, run the command given below. You can change the sampler to evaluate using different task classes.

```
python evaluator.py \
    --dfa-sampler CompositionalReachAvoidDerived \
    --model-path storage/GATv2Conv-dumb_ac_CompositionalReachAvoidDerived_Dummy-MDP-v0_seed:1_epochs:2_bs:1024_fpp:512_dsc:0.9_lr:0.001_ent:0.01_clip:0.1_prog:full_dfa:True \
    --dumb-ac \
    --discount 0.94 \
    --eval-episodes 100 \
    --env Dummy-MDP-v0 \
    --gnn GATv2Conv \
    --dfa
```


## Training
To train a policy with no pretrained GNN in the discrete Letterworld MDP, run the command given below.

```
python train_agent.py \
	--algo ppo \
	--env Letter-7x7-v3 \
	--log-interval 1 \
	--save-interval 20 \
	--frames 10000000 \
	--discount 0.94 \
	--dfa-sampler CompositionalReachAvoidDerived \
	--epochs 4 \
	--lr 0.0003 \
	--seed 1 \
	--gnn GATv2Conv \
	--dfa
```

To train a policy with no pretrained GNN in the continuous Zones MDP, run the command given below.

```
python train_agent.py \
	--algo ppo \
	--env Zones-5-v0 \
	--log-interval 1 \
	--save-interval 2 \
	--dfa-sampler CompositionalReachAvoidDerived \
	--frames-per-proc 4096 \
	--batch-size 2048 \
	--lr 0.0003 \
	--discount 0.998 \
	--entropy-coef 0.003 \
	--epochs 10 \
	--frames 40000000 \
	--gnn GATv2Conv \
	--dfa
```

If you have a GNN pretrained on a specific task class, include the `--pretrained-gnn` flag in your command and also pass the name of the sampler as an option to the script using `--pretrained-gnn-sampler`. The script will find the pretrained model in the `storage` directory. To freeze the GNN, also add `--freeze` to your command.

### Evaluation
To evaluate a trained policy, run the command given below.

```
python evaluator.py \
    --dfa-sampler CompositionalReachAvoidDerived \
    --model-path storage/GATv2Conv_CompositionalReachAvoidDerived_Letter-7x7-v3_seed:1_epochs:4_bs:256_fpp:None_dsc:0.94_lr:0.0003_ent:0.01_clip:0.2_prog:full_dfa:True \
    --discount 0.94 \
    --eval-episodes 100 \
    --env Letter-7x7-v3 \
    --gnn GATv2Conv \
    --dfa
```

## Samplers
We have various DFA samplers implemented in the repository. We can divide them into seven major categories as given below.

1. `ReachAvoidDerived` samples RAD DFAs, see the paper for details.
2. `Reach_a_b_c_d` samples `n ~ Uniform(c, d)` DFAs, each representing a sequence of goals with length `m_i ~ Uniform(a, b)`.
3. `ReachAvoid_a_b_c_d` samples `n ~ Uniform(c, d)` DFAs, each representing a sequence of reach-avoid tasks with length `m_i ~ Uniform(a, b)`.
4. `ReachAvoidRedemption_a_b_c_d` samples `n ~ Uniform(c, d)` DFAs, each representing a sequence of reach-avoid-redemption tasks with length `m_i ~ Uniform(a, b)`.
5. `Parity_a_b_c_d` samples `n ~ Uniform(c, d)` DFAs, each representing a sequence of parity tasks with length `m_i ~ Uniform(a, b)`.
6. `Until_a_b_c_d` samples DFAs from the avoidance task class of LTL2Action.
7. `Eventually_a_b_c_d` samples DFAs from the partially-ordered task class of LTL2Action.

These task classes return DFAs, not cDFAs. To sample tasks in the cDFA format, add the `Compositional` prefix to the sampler names.

## Acknowledgements
This repository originated as a fork of [the LTL2Action repository](https://github.com/LTL2Action/LTL2Action). After substantial modifications implementing our compositional automata framework, it has evolved into its current form.
We express our gratitude to the authors of [the LTL2Action paper](https://arxiv.org/pdf/2102.06858) for their transparent, reproducible, and well-maintained implementation. In our paper, we provide a detailed comparison and discussion of our approach in relation to theirs.
