# Who am I , and what is this

I am an aerospace engineer by trade who works predominantly on flight software. I found it rather annoying that llms are black boxes, and hence more scarier to use in aerospace, so here I made an attempt towards fixing that.

Disclaimer: This was one of my hobby projects so you will find a bit of humour stewn around in this description and project. In terms of the actual results I claim I have made effort for it to be correct. This project is by no means completely rigorous because my dog ate my initial scripts, but I still felt it had enough interesting substance, alteast in terms of llm interpretability (and maybe performance) to share. 


# Neural_Compiler 

This neural compiler project is my attempt to make more interpretable llms by composing llms from primitives using techniques from Symbolic Regression and Neural Architecture Search. I aim to reveal the secrets of the thoughts of the predecessors of the AI overlords of the future.

# What does it give us

When pointed at dataset of your choice, the neural compiler will give you a network composed from its primitives and a network report that will tell you the equation that each node in the network represents. Simple as that really. It reveals cool equations that define how a neural network bridges the gap between input params and output, but for larger networks and in a **more readable** format than just 

relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(relu(x).....

This does this by letting the llm choose from a list of 24 primitives for each node and also letting it choose the connections between each nodes and the input tokens / heads.

Attention is not directly encoded into the network, rather it can be composed by search using the primitives. I am having another future version in the pipeline where I am looking at how baking in attention changes the search dynamics.

So you get something like

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
and so on

Now dont worry if this equation is confusing, it is for me too , I'll utilize the power of the distillation of vast centuries of human knowledge to write a guide on how you and I can read this network report at the end of this readme (using Opus 4.7 of course)

# How Does it work

The network starts with a grid of a fixed no of layers and nodes per layer. Using a gumbal softmax routing trick ( see https://sassafras13.github.io/GumbelSoftmax/ which allows for differentiable search over categorical parameters), it determins what primitive each node will use and what interconnections will exist between nodes.

# What are the results

1. Tiny Shakespeare full dataset with approx 20M params - 1.7 val loss with much lower train loss still fitting with a summarization head, an 8 x 8 layer and 128 embedding dim. Now this is somewhere along the values achieved by MiniGPT (https://arxiv.org/html/2605.17398v1) which has the same embedding dim but 0.8M params. A lot of our additional params come from gumbal softmax search over 24ish primitives (out of which 13 need weights, so they contribute to our massive increase in params). My **unverified** estimate is that we might land somewhere around 2-5M params if we were to trim the unused primitives after training. And the number seem to say that the embedding dimensions seem to be the important lever for this problem set.

2. Fineweb - 5.2 loss and dropping (but slowly) - Invigorated by the above results and convinced of my capability to create AI history, I like any fool in history jumped headfirst into the dark depths of the abyss that is fineweb and it quickly showed my how my meagre hardware and pockets stack up to the behemoth that is a trillion tokens of data. So this test was done with similar no of parameters and the same embedding dimensions as the above, but that quickly proved foolish. Renting a cheap RTX 4000 ada on runpod, I was able to bump up dimensions to 256, but wasnt able to bump the available no of hours in day higher than 24. So at this point this is where this project is located, and I figured I could put it out in the open so more folks who know what they are doing better than me can play around with it and also see the magic of the network report revealing the equations that define the dark thoughts of AI to takeover humankind.

# Which part of this are AI based

A large amount of the adapting of this code from a CPU based symbolic regression equation compiler (all manual) to a pytorch/Cuda based llm compiler was done by careful planning and execution with Opus4.7 and claude code. The implementation of the Gumbal softmax trick was also suggested and implemented by AI, and from the output side it seems to work, during search regularly the compiler explores a given primitive, finds that it does not work after a few epochs and backtracks to a better primitive. Before this softmax trick, I had a genetic algorithm based search for these params ( NSGA - 2 using the optuna library)

# What are its limitations and next steps.

The tokens are projected into a single summarizer head , which is then operated on by later layers. This was done so that I can scale context length on my poor Laptop 3060 GPU, god bless his soul for his valiant efforts the past couple of weeks. In practice, I expect this scales poorly for performance and summarization, by definition also includes loss of some of the incoming context from the tokens. If i was a rich guy with a big beefy GPU I would run my search directly on larger and larger token lengths directly, and each node can learn better representations by operating directly on the full context window. I guess one of you guys and gals might find it interesting to do that.

Since the compiler runs a gumbal softmax calculation for each primitive that the network searches through, this massively increases the memory footprint of the weights by 10x , 15x during training and also during inference if the node requires a soft mixture at that node. In practice, a lot of the nodes do converge to single primitives, so this memory footprint can be lower. I think having each primitive is important for the expressibility of the program

This compiler is a single objective search unlike my SR version which was a multi objective one. This is on purpose becuase my goal was to see how good we can get regardless of cost for these language models.

I feel there are massive gains to be made in this setup by just scaling context length, removing summarization heads and increasing embedding dimensions. The MLP parts are where the gains of this would reflect, because logically one sin or log function can exactly capture what takes tons of relu nodes.

As I mentioned above, attention does solve a lot of the problems faced by this cleanly, as now the architecture needs to learn relationships between the tokens and to process them in a correct way so that it can compose the right answer. Attention provides a lot of this relative information in a pre baked manner and instead of using a summary head, if we use an attention head it should work better... Ill be trying that in the coming days.


# Why in the world would I try this:
When I did the Symbolic Regression version, whose focus was to find cheaper equations, I found that for the example of exp(x), the network I was composing kept stacking relus and gelus instead of multiplies. The exp(x) is just 2.718 * 2.718 ... x times, but that is an exponential needle in the haystack for the compiler to find. So i provided it with the option to do exp(x) as a primitive and it found it immediately, giving us an exact machine precision solution for a fraction of the cost. So going by that logic this gives us this image.

<img width="1800" height="1120" alt="reachability_v2" src="https://github.com/user-attachments/assets/ac07c94b-3619-4918-ab9f-904b720ac95a" />

So normally, tranformers, on top of the attention primitive, scale search pressure with tons and tons of compute.
So this is an effort for me to scale primitives a bit before we scale the search pressure.


# Is this novel
I cant say for sure, the concept popped in my head one day, where I thought, ohh if i could just know the exact equation , then I wouldnt need to worry about a network making mistakes ( which is also a deeply naive statement :) ).  However, when I started working on it, I read quite some related literature, some off the top of my head are EN4SR, PYSR, AIFeynman, Neural Architecture Search, DARTS for SR, and Understanding Deep Learning, Attention is all you need,  MiniGPT, NanoGPT and many more papers/articles on the neural network side. 

I dont expect it to be, its just something I looked at and maybe it helps a couple of people somewher.



Ohh btw did I say the 7th layer is almost unused in both tests... free depth ablation baby ... wooo!!!!
