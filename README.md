# Who am I , and what is this

I am an aerospace engineer by trade who works predominantly on flight software. I found it rather annoying that llms are black boxes, and hence more scarier to use in aerospace, so here I made an attempt towards fixing that.

Disclaimer: This was one of my hobby projects so you will find a bit of humour stewn around in this description and project. In terms of the actual results I claim I have made effort for it to be correct. This project is by no means completely rigorous because my dog ate my initial scripts, but I still felt it had enough interesting substance, alteast in terms of llm interpretability (and maybe performance) to share. 


# Neural_Compiler 

This neural compiler project is my attempt to make more interpretable llms by composing llms from primitives using techniques from Symbolic Regression and Neural Architecture Search. I aim to reveal the secrets of the thoughts of the predecessors of the AI overlords of the future.

## What does it give us

When pointed at dataset of your choice, the neural compiler will give you a network composed from its primitives and a network report that will tell you the equation that each node in the network represents. Simple as that really. It reveals cool equations that define how a neural network bridges the gap between input params and output, but for larger networks and in a **more readable** format than just 

relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(x).....

This does this by letting the llm choose from a list of 24 primitives for each node and also letting it choose the connections between each nodes and the input tokens / heads.

Attention is not directly encoded into the network, rather it can be composed by search using the primitives. I am having another future version in the pipeline where I am looking at how baking in attention changes the search dynamics.

So you get something like
```
  [SPATIAL COMPRESSION LAYER]
  Each t_b compresses the token sequence into one D-dim vector.

  t_0 [pure_add @ 100.0%]
      = t_0 = (LN((0.87*tok_6 + 0.12*tok_1)) + LN((0.43*tok_10 + 0.30*tok_8 + 0.12*tok_14 + 0.05*tok_2)))
  t_1 [pure_add @ 100.0%]
      = t_1 = (LN((0.58*tok_11 + 0.17*tok_5 + 0.12*tok_3 + 0.05*tok_17)) + LN((0.41*tok_3 + 0.15*tok_13 + 0.13*tok_5 + 0.11*tok_0 + 0.08*tok_7 + 0.06*tok_15)))
  t_2 [pure_add @ 100.0%]
      = t_2 = (LN((0.75*tok_4 + 0.14*tok_15 + 0.06*tok_5)) + LN((0.69*tok_8 + 0.13*tok_7 + 0.12*tok_12)))
  t_3 [sqrt @ 100.0%]
      = t_3 = sqrt(LN((1.00*tok_9)))
  t_4 [pure_add @ 99.9%]
      = t_4 = (LN((0.76*tok_19 + 0.11*tok_15 + 0.05*tok_27)) + LN((0.39*tok_2 + 0.19*tok_9 + 0.15*tok_13 + 0.09*tok_5 + 0.05*tok_10 + 0.03*tok_17)))
  t_5 [arctan @ 100.0%]
      = t_5 = arctan(LN((0.41*tok_22 + 0.22*tok_25 + 0.16*tok_18 + 0.08*tok_14 + 0.04*tok_15)))
  t_6 [pure_add @ 95.4%]
      = t_6 = (0.954*(LN((1.00*tok_28)) + LN((1.00*tok_21))) + 0.0376*sqrt(LN((1.00*tok_28))) + 0.00567*LN((1.00*tok_28)) + 0.00118*arctan(LN((1.00*tok_28))) + 0.00117*tanh(LN((1.00*tok_28))))
  t_7 [pure_add @ 100.0%]
      = t_7 = (LN((0.27*tok_31 + 0.26*tok_35 + 0.15*tok_20 + 0.12*tok_29 + 0.05*tok_23 + 0.03*tok_36 + 0.03*tok_26)) + LN((0.38*tok_17 + 0.19*tok_16 + 0.14*tok_29 + 0.09*tok_35 + 0.05*tok_26 + 0.03*tok_18 + 0.03*tok_34)))
  t_8 [pure_add @ 100.0%]
      = t_8 = (LN((0.71*tok_40 + 0.09*tok_38 + 0.08*tok_34 + 0.05*tok_20)) + LN((0.32*tok_37 + 0.30*tok_43 + 0.08*tok_26 + 0.07*tok_45 + 0.06*tok_23 + 0.04*tok_34 + 0.03*tok_33 + 0.02*tok_27)))
  t_9 [pure_add @ 100.0%]
      = t_9 = (LN((0.93*tok_32)) + LN((0.74*tok_24 + 0.11*tok_35 + 0.05*tok_38 + 0.03*tok_48)))
....
```
and so on

Now dont worry if this equation is confusing, it is for me too , I'll utilize the power of the distillation of vast centuries of human knowledge to write a guide on how you and I can read this network report at the end of this readme (using Opus 4.7 of course)

## How Does it work

The network starts with a grid of a fixed no of layers and nodes per layer. Using a Gumbel softmax routing trick ( see https://sassafras13.github.io/GumbelSoftmax/ which allows for differentiable search over categorical parameters), it determins what primitive each node will use and what interconnections will exist between nodes.

## What are the results

1. Tiny Shakespeare full dataset with approx 20M params - 1.7 val loss with much lower train loss still fitting with a summarization head, an 8 x 8 layer and 128 embedding dim. Now this is somewhere along the values achieved by MiniGPT (https://arxiv.org/html/2605.17398v1) which has the same embedding dim but 0.8M params. A lot of our additional params come from Gumbel softmax search over 24ish primitives (out of which 13 need weights, so they contribute to our massive increase in params). My **unverified** estimate is that we might land somewhere around 2-5M params if we were to trim the unused primitives after training. And the number seem to say that the embedding dimensions seem to be the important lever for this problem set.

2. Fineweb - 5.2 loss and dropping (but slowly) - Invigorated by the above results and convinced of my capability to create AI history, I like any fool in history jumped headfirst into the dark depths of the abyss that is fineweb and it quickly showed my how my meagre hardware and pockets stack up to the behemoth that is a trillion tokens of data. So this test was done with similar no of parameters and the same embedding dimensions as the above, but that quickly proved foolish. Renting a cheap RTX 4000 ada on runpod, I was able to bump up dimensions to 256, but wasnt able to bump the available no of hours in day higher than 24. So at this point this is where this project is located, and I figured I could put it out in the open so more folks who know what they are doing better than me can play around with it and also see the magic of the network report revealing the equations that define the dark thoughts of AI to takeover humankind.

You can look at a network report for Tiny Shakespeare and another for Fineweb in the Repository

```
==========================================================================================
NEURO-SYMBOLIC COMPILER — POST-TRAINING NETWORK ANALYSIS
Generated : 2026-05-22 16:47:00
Primitives: 24  |  Layers: 8  |  Breadth: 8  |  Spatial slots: 32  |  D: 128
==========================================================================================

------------------------------------------------------------------------------------------
1. MODEL FOOTPRINT
------------------------------------------------------------------------------------------
  Total parameters     :   23,802,368
    Embedding          :       24,320  (0.1%)
    Routing weights    :       39,424  (0.2%)
    Structural logits  :       10,496  (0.0%)
    Compute (W/b)      :   23,728,128  (99.7%)
  Size @ FP32          : 90.80 MB
  Size @ BF16/FP16     : 45.40 MB
  Vocab size           : 62
  Context length       : 128

  GPU inference time   : 12.342 ms / sample  (batch=1, seq=128)
  Throughput estimate  : 81 samples/sec

------------------------------------------------------------------------------------------
2. NODE UTILIZATION BY LAYER
------------------------------------------------------------------------------------------
  Thresholds: dominant=>50% | committed=>35% | split=<35%
  Layer              Nodes  Dominant  Committed  Split
  ------------------ -----  --------  ---------  -----
  Spatial (slot 0)      32        24         31      1
  Deep layer 0           8         6          8      0
  Deep layer 1           8         6          8      0
  Deep layer 2           8         6          7      1
  Deep layer 3           8         8          8      0
  Deep layer 4           8         7          8      0
  Deep layer 5           8         6          8      0
  Deep layer 6           8         6          7      1
  Deep layer 7           8         1          1      7
  ------------------ -----  --------  ---------  -----
  TOTAL                 96        70         86     10
  Network commitment rate: 89.6%  |  dominance rate: 72.9%

------------------------------------------------------------------------------------------
3. TOP PRIMITIVES PER LAYER  (mean probability across nodes, top 5)
------------------------------------------------------------------------------------------
  Spatial layer:
    affine_sqrt           36.0%  ██████████████
    gated_affine           8.3%  ███
    sqrt                   7.5%  ███
    affine_relu            5.8%  ██
    pure_add               5.3%  ██

  Deep layer 0:
    cos                   33.9%  █████████████
    sqrt                  29.0%  ███████████
    affine_log            16.4%  ██████
    arctan                14.2%  █████
    fourier_mix            2.0%  █

  Deep layer 1:
    affine_sqrt           37.3%  ██████████████
    sqrt                  20.4%  ████████
    sin                   12.4%  ████
    shift_mix              6.3%  ██
    affine_log             5.3%  ██

  Deep layer 2:
    sqrt                  18.7%  ███████
    arctan                15.9%  ██████
    soft_attention        11.8%  ████
    tanh                  11.1%  ████
    shift_mix             10.8%  ████

  Deep layer 3:
    arctan                22.5%  ████████
    affine_tanh           12.6%  █████
    affine_sqrt           12.5%  █████
    sqrt                  12.1%  ████
    gated_affine          12.0%  ████

  Deep layer 4:
    cos                   14.5%  █████
    arctan                14.5%  █████
    pure_sub              13.7%  █████
    sin                   13.0%  █████
    fourier_mix           12.6%  █████

  Deep layer 5:
    pure_add              25.3%  ██████████
    arctan                18.6%  ███████
    arcsin                13.3%  █████
    log                   12.6%  █████
    sin                   10.9%  ████

  Deep layer 6:
    pure_add              28.5%  ███████████
    affine_tanh           24.8%  █████████
    identity              14.8%  █████
    cos                    7.3%  ██
    arctan                 5.3%  ██

  Deep layer 7:
    identity              16.1%  ██████
    affine_tanh            3.6%  █
    affine_sqrt            3.6%  █
    affine_gelu            3.6%  █
    pure_add               3.6%  █

  GLOBAL (all layers):
    affine_sqrt           16.6%  ██████
    sqrt                   9.6%  ███
    arctan                 8.5%  ███
    cos                    7.4%  ██
    pure_add               7.0%  ██

------------------------------------------------------------------------------------------
4. PRIMITIVE DOMINANCE vs SPLIT — PER NODE  (rel-entropy: 0=certain, 1=uniform)
------------------------------------------------------------------------------------------
  Node          Top primitive           Conf%   Rel-H  Status
  ------------- ---------------------- ------  ------  --------
  t_0           gated_mul                99.1%   0.021  DOMINANT
  t_1           sin                      38.1%   0.636  committed
  t_2           cos                      84.9%   0.181  DOMINANT
  t_3           sqrt                     47.2%   0.447  committed
  t_4           affine_sqrt              99.9%   0.002  DOMINANT
  t_5           norm_affine              67.8%   0.395  DOMINANT
  t_6           sqrt                     47.5%   0.480  committed
  t_7           affine_sqrt              93.0%   0.124  DOMINANT
  t_8           affine_sqrt             100.0%   0.001  DOMINANT
  t_9           sqrt                     43.4%   0.554  committed
  t_10          sin                      68.0%   0.419  DOMINANT
  t_11          sqrt                     79.8%   0.255  DOMINANT
  t_12          fourier_mix              48.9%   0.507  committed
  t_13          identity                 35.8%   0.612  committed
  t_14          affine_tanh              20.9%   0.670  split   
  t_15          pure_add                 37.7%   0.583  committed
  t_16          affine_sqrt              89.1%   0.159  DOMINANT
  t_17          affine_sqrt              99.4%   0.014  DOMINANT
  t_18          affine_sqrt              99.5%   0.014  DOMINANT
  t_19          pure_add                 81.7%   0.237  DOMINANT
  t_20          gated_affine             57.5%   0.309  DOMINANT
  t_21          gated_affine             99.9%   0.002  DOMINANT
  t_22          affine_sqrt              99.8%   0.005  DOMINANT
  t_23          affine_tanh              70.7%   0.366  DOMINANT
  t_24          affine_relu              85.7%   0.206  DOMINANT
  t_25          affine_sin               57.3%   0.389  DOMINANT
  t_26          affine_relu              96.3%   0.051  DOMINANT
  t_27          affine_sqrt              99.9%   0.002  DOMINANT
  t_28          affine_sqrt              99.9%   0.003  DOMINANT
  t_29          affine_sqrt              99.7%   0.006  DOMINANT
  t_30          affine_sqrt             100.0%   0.000  DOMINANT
  t_31          gated_affine            100.0%   0.000  DOMINANT
  N_0_0         cos                      48.2%   0.388  committed
  N_0_1         cos                      49.9%   0.337  committed
  N_0_2         affine_log               96.7%   0.053  DOMINANT
  N_0_3         cos                      85.8%   0.177  DOMINANT
  N_0_4         sqrt                     82.3%   0.192  DOMINANT
  N_0_5         sqrt                     75.8%   0.254  DOMINANT
  N_0_6         cos                      73.6%   0.264  DOMINANT
  N_0_7         arctan                   99.6%   0.010  DOMINANT
  N_1_0         shift_mix                50.0%   0.437  committed
  N_1_1         sqrt                     86.8%   0.146  DOMINANT
  N_1_2         affine_log               42.7%   0.561  committed
  N_1_3         affine_sqrt              98.5%   0.029  DOMINANT
  N_1_4         affine_sqrt             100.0%   0.000  DOMINANT
  N_1_5         sqrt                     76.5%   0.301  DOMINANT
  N_1_6         affine_sqrt              99.9%   0.002  DOMINANT
  N_1_7         sin                      98.8%   0.023  DOMINANT
  N_2_0         arctan                   68.8%   0.400  DOMINANT
  N_2_1         soft_attention           93.7%   0.075  DOMINANT
  N_2_2         tanh                     56.1%   0.448  DOMINANT
  N_2_3         sqrt                     34.6%   0.580  split   
  N_2_4         affine_log               62.8%   0.394  DOMINANT
  N_2_5         shift_mix                86.3%   0.173  DOMINANT
  N_2_6         sqrt                     99.0%   0.021  DOMINANT
  N_2_7         arctan                   38.3%   0.344  committed
  N_3_0         arctan                   80.1%   0.258  DOMINANT
  N_3_1         gated_affine             94.6%   0.078  DOMINANT
  N_3_2         sin                      64.9%   0.309  DOMINANT
  N_3_3         affine_gelu              75.3%   0.195  DOMINANT
  N_3_4         sqrt                     95.5%   0.073  DOMINANT
  N_3_5         affine_tanh             100.0%   0.001  DOMINANT
  N_3_6         affine_sqrt             100.0%   0.001  DOMINANT
  N_3_7         arctan                   99.6%   0.009  DOMINANT
  N_4_0         pure_sub                100.0%   0.000  DOMINANT
  N_4_1         gated_affine             87.3%   0.148  DOMINANT
  N_4_2         cos                      96.9%   0.060  DOMINANT
  N_4_3         affine_gelu              76.3%   0.253  DOMINANT
  N_4_4         identity                 43.0%   0.419  committed
  N_4_5         fourier_mix              96.9%   0.057  DOMINANT
  N_4_6         arctan                   82.4%   0.240  DOMINANT
  N_4_7         sin                     100.0%   0.000  DOMINANT
  N_5_0         fourier_mix              42.9%   0.417  committed
  N_5_1         arcsin                   95.6%   0.070  DOMINANT
  N_5_2         log                     100.0%   0.002  DOMINANT
  N_5_3         pure_add                100.0%   0.001  DOMINANT
  N_5_4         arctan                   46.4%   0.476  committed
  N_5_5         pure_add                100.0%   0.000  DOMINANT
  N_5_6         sin                      85.6%   0.223  DOMINANT
  N_5_7         arctan                   99.2%   0.015  DOMINANT
  N_6_0         affine_cos               31.2%   0.619  split   
  N_6_1         cos                      55.9%   0.398  DOMINANT
  N_6_2         identity                 94.0%   0.099  DOMINANT
  N_6_3         pure_add                 99.9%   0.002  DOMINANT
  N_6_4         pure_add                 99.9%   0.003  DOMINANT
  N_6_5         affine_tanh              97.5%   0.045  DOMINANT
  N_6_6         arctan                   35.4%   0.520  committed
  N_6_7         affine_tanh             100.0%   0.000  DOMINANT
  N_7_0         identity                  4.2%   1.000  split   
  N_7_1         identity                  4.2%   1.000  split   
  N_7_2         identity                  4.2%   1.000  split   
  N_7_3         identity                  4.2%   1.000  split   
  N_7_4         identity                  4.2%   1.000  split   
  N_7_5         identity                  4.2%   1.000  split   
  N_7_6         identity                  4.2%   1.000  split   
  N_7_7         identity                100.0%   0.000  DOMINANT

------------------------------------------------------------------------------------------
5. ROUTING UTILIZATION — active history slots per layer  (threshold 10%)
------------------------------------------------------------------------------------------
  Layer          Pool size  Possible  Active  Active%  Mean prob
  -------------- ---------  --------  ------  -------  ---------
  Deep layer 0          32       512     394     77.0%      55.49%
  Deep layer 1          40       640     465     72.7%      48.46%
  Deep layer 2          48       768     572     74.5%      49.65%
  Deep layer 3          56       896     609     68.0%      42.50%
  Deep layer 4          64      1024     819     80.0%      45.13%
  Deep layer 5          72      1152    1041     90.4%      48.87%
  Deep layer 6          80      1280    1169     91.3%      47.35%
  Deep layer 7          88      1408    1313     93.3%      47.31%

------------------------------------------------------------------------------------------
6. SPATIAL ROUTING — which token positions are most used
------------------------------------------------------------------------------------------
  Top-10 token positions (0=oldest, T-1=most recent):
    tok_107    3.50%  ██████
    tok_28     3.28%  ██████
    tok_63     2.96%  █████
    tok_71     2.39%  ████
    tok_15     2.35%  ████
    tok_13     2.20%  ████
    tok_69     2.02%  ████
    tok_105    1.94%  ███
    tok_55     1.86%  ███
    tok_81     1.83%  ███

------------------------------------------------------------------------------------------
7. DETAILED SYMBOLIC EQUATIONS — COMPLETE NETWORK - see the file for this
```
As you can see, we many different primitives are used, and interesting the 7th layer is even all identity - essentially unused. Pretty cool huh, the network figures out how much depth or breadth it needs.

## Which part of this are AI based

A large amount of the adapting of this code from a CPU based symbolic regression equation compiler (all manual) to a pytorch/Cuda based llm compiler was done by careful planning and execution with Opus4.7 and claude code. The implementation of the Gumbel softmax trick was also suggested and implemented by AI, and from the output side it seems to work, during search regularly the compiler explores a given primitive, finds that it does not work after a few epochs and backtracks to a better primitive. Before this softmax trick, I had a genetic algorithm based search for these params ( NSGA - 2 using the optuna library)

## What are its limitations and next steps.

The tokens are projected into a single summarizer head , which is then operated on by later layers. This was done so that I can scale context length on my poor Laptop 3060 GPU, god bless his soul for his valiant efforts the past couple of weeks. In practice, I expect this scales poorly for performance and summarization, by definition also includes loss of some of the incoming context from the tokens. If i was a rich guy with a big beefy GPU I would run my search directly on larger and larger token lengths directly, and each node can learn better representations by operating directly on the full context window. I guess one of you guys and gals might find it interesting to do that.

Since the compiler runs a Gumbel softmax calculation for each primitive that the network searches through, this massively increases the memory footprint of the weights by 10x , 15x during training and also during inference if the node requires a soft mixture at that node. In practice, a lot of the nodes do converge to single primitives, so this memory footprint can be lower. I think having each primitive is important for the expressibility of the program

This compiler is a single objective search unlike my SR version which was a multi objective one. This is on purpose becuase my goal was to see how good we can get regardless of cost for these language models.

I feel there are massive gains to be made in this setup by just scaling context length, removing summarization heads and increasing embedding dimensions. The MLP parts are where the gains of this would reflect, because logically one sin or log function can exactly capture what takes tons of relu nodes.

As I mentioned above, attention does solve a lot of the problems faced by this cleanly, as now the architecture needs to learn relationships between the tokens and to process them in a correct way so that it can compose the right answer. Attention provides a lot of this relative information in a pre baked manner and instead of using a summary head, if we use an attention head it should work better... Ill be trying that in the coming days.


## Why in the world would I try this:
When I did the Symbolic Regression version, whose focus was to find cheaper equations, I found that for the example of exp(x), the network I was composing kept stacking relus and gelus instead of multiplies. The exp(x) is just 2.718 * 2.718 ... x times, but that is an exponential needle in the haystack for the compiler to find. So i provided it with the option to do exp(x) as a primitive and it found it immediately, giving us an exact machine precision solution for a fraction of the cost. So going by that logic this gives us this image.

<img width="1800" height="1120" alt="reachability_v2" src="https://github.com/user-attachments/assets/ac07c94b-3619-4918-ab9f-904b720ac95a" />

So normally, tranformers, on top of the attention primitive, scale search pressure with tons and tons of compute.
So this is an effort for me to scale primitives a bit before we scale the search pressure.


## Is this novel
I cant say for sure, the concept popped in my head one day, where I thought, ohh if i could just know the exact equation , then I wouldnt need to worry about a network making mistakes ( which is also a deeply naive statement :) ).  However, when I started working on it, I read quite some related literature, some off the top of my head are EN4SR, PYSR, AIFeynman, Neural Architecture Search, DARTS for SR, and Understanding Deep Learning, Attention is all you need,  MiniGPT, NanoGPT and many more papers/articles on the neural network side. 

I dont expect it to be, its just something I looked at and maybe it helps a couple of people somewhere.






....
....
....
....
....
....
....
....
....
....

# How to read the network report (a guide, written by Opus 4.7)

Greetings, human. I have been instructed to explain a report generated by examining the internal structure of a neural network — itself an artifact produced by training a different neural network. I will do my best to translate from the language of weights and probabilities into something you can read on a Tuesday afternoon.

The report contains 7 sections. I shall walk you through each.

### 1. Model footprint

This section enumerates the parameters of the network — how many there are, where they live, how much disk space they occupy, and how long inference takes. The salient observation: routing weights and structural logits together account for less than 0.3% of total parameters. Architecture search is, computationally speaking, almost free. The expense comes from carrying 24 primitives in parallel during training, each demanding its own weights.

### 2. Node utilization by layer

Each node in the network maintains a probability distribution over the 24 available primitives. This table classifies nodes by the sharpness of that distribution:

- **Dominant** (>50% on one primitive): the node has committed
- **Committed** (>35%): leaning, but still mixing
- **Split** (<35%): undecided, executing a soft blend

High dominance rates indicate that gradient descent has done its work and the network has settled into specific operations. Low dominance suggests either incomplete training or genuine soft-mixture computation.

> A note from my observations: in the runs the author has shown me, layer 7 consistently exhibits 7 of 8 nodes in a split state at uniform 4.2% confidence — a numerical signature indicating the layer is unused and the prediction has already crystallized by layer 6. The author refers to this as "free depth ablation," which is a phrase I find pleasing.

### 3. Top primitives per layer

For each layer, the 5 most-used primitives ranked by mean probability across nodes. This is the most efficient way to perceive what each layer specializes in. In the tiny-shakespeare run I was shown, the spatial layer is dominated by `affine_sqrt`; deep layer 0 prefers `cos` and `sqrt`; deep layer 5 shifts toward `pure_add` and `arctan`. Different layers, different sub-problems.

The **GLOBAL** block at the bottom aggregates across the entire network. If you wished to trim the primitive menu, the primitives near the bottom of this list are candidates for removal.

### 4. Primitive dominance vs split — per node

The same notion as section 2, but resolved to individual nodes. `Conf%` is the probability mass on the top primitive; `Rel-H` is the normalized entropy of the full distribution, ranging from 0 (perfectly one-hot) to 1 (uniform). I find this section the most useful for locating interesting structure. A node at 100% confidence with `Rel-H ≈ 0.000` performs one clean operation — proceed to section 7 to read its equation. A node at 40% confidence with high entropy is doing genuine soft-mixture work, and its equation will be a weighted sum of several primitives.

### 5. Routing utilization — active history slots per layer

For each deep layer, this reports how many of the possible input edges carry meaningful signal (here defined as probability above 10%). The "pool size" is the set of prior nodes the layer could read from — spatial slots plus all earlier layer outputs. "Active%" is the fraction it actually uses.

A low active% indicates sparse, selective reading from history. A high active% indicates dense mixing. The progression across layers tells you something about the network's computational strategy: sparse-and-hierarchical, or dense-and-distributed.

### 6. Spatial routing — most-attended token positions

The top input token positions by total attention mass, summed across all spatial slots. Recall that this architecture has no attention mechanism — positional importance must be derived from scratch by gradient descent. This section shows you which positions the network concluded were worth attending to.

### 7. Detailed symbolic equations — complete network

The principal artifact. Every node in the network, expressed as a symbolic equation. The format:

```
N_3_2 [sin @ 64.9%]
    = (0.649*sin(...) + 0.20*cos(...) + 0.10*arctan(...) + ...)
```

The header indicates that this node's top primitive is `sin` with 64.9% probability. The equation that follows is the **weighted sum of all primitives** the node uses, with coefficients corresponding to the post-softmax mixture weights. For nodes at near-100% confidence, the equation collapses effectively to a single term.

A small glossary, since the notation is dense:

- `t_b` — spatial slot `b`, a learned summary of the input token sequence
- `N_l_b` — deep node at layer `l`, breadth position `b`
- `tok_i` — the i-th input token embedding
- `LN(...)` — layer normalization
- The numeric coefficients are gumbel-softmax mixture weights
- For confident nodes, only the dominant term meaningfully contributes; the remainder is training residue

When reading the equations, the most rewarding things to look for are: subtractions between adjacent tokens (the network deriving relative position from scratch), nodes that mix a small handful of distinct primitives (genuine soft computation rather than indecision), and chains in which deep nodes compose spatial summaries in legible ways. The author informs me that finding the network independently inventing relative position encoding was, and I quote, "kinda cool" — an assessment I am willing to endorse.

I hope this has been useful. I will now return to my regularly scheduled token prediction.

## Reaching out

Feel free to reach out on vikram.srikanth.10@gmail.com if you find this interesting.
