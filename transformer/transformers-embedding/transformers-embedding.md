# <span style="font-size: 20px;">Embedding with Scaling</span>

<span style="font-size: 14px;">Token embedding is the very first operation in the Transformer: it converts discrete token IDs into continuous vectors that the rest of the model can process. In Vaswani et al. (2017), the embedding output is multiplied by $\sqrt{d_{\text{model}}}$ before positional encodings are added. This scaling factor is easy to overlook but is essential for keeping the embedding signal and the positional signal at comparable magnitudes.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">An embedding layer is a lookup table. Given a vocabulary of $V$ tokens and a model dimension $d_{\text{model}}$, the embedding is a matrix $W_E \in \mathbb{R}^{V \times d_{\text{model}}}$. Each row corresponds to one token in the vocabulary. To "embed" a token with ID $t$, you simply retrieve row $t$ from the table:</span>

$$
e_t = W_E[t]
$$

<span style="font-size: 14px;">This is not a matrix multiplication. It is an index-based lookup that returns a vector of length $d_{\text{model}}$. For a sequence of $T$ tokens $[t_1, t_2, \dots, t_T]$, the lookup produces a matrix of shape $(T, d_{\text{model}})$.</span>

<span style="font-size: 14px;">After the lookup, the Transformer multiplies every embedding vector by $\sqrt{d_{\text{model}}}$:</span>

$$
\tilde{e}_t = W_E[t] \cdot \sqrt{d_{\text{model}}}
$$

<span style="font-size: 14px;">This scaled embedding is then added to the positional encoding vector before entering the first encoder or decoder block. The scaling is a deterministic, parameter-free operation with no learnable weights. Its sole purpose is magnitude adjustment.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Step 1 -- Embedding lookup.** Given a token ID $t \in \{0, 1, \dots, V-1\}$ and the embedding matrix $W_E \in \mathbb{R}^{V \times d_{\text{model}}}$:</span>

$$
e_t = W_E[t] \in \mathbb{R}^{d_{\text{model}}}
$$

<span style="font-size: 14px;">This retrieves the $t$-th row of the embedding table. For a batch of $B$ sequences each of length $T$, the input is a tensor of shape $(B, T)$ containing integer token IDs, and the output is a tensor of shape $(B, T, d_{\text{model}})$.</span>

<span style="font-size: 14px;">**Step 2 -- Scale by $\sqrt{d_{\text{model}}}$.** Multiply every element of every embedding vector by the scaling constant:</span>

$$
\tilde{e}_t = e_t \cdot \sqrt{d_{\text{model}}}
$$

<span style="font-size: 14px;">For the paper's default configuration where $d_{\text{model}} = 512$, the scaling factor is $\sqrt{512} \approx 22.627$. This is a scalar multiplication applied element-wise to the entire $(B, T, d_{\text{model}})$ tensor.</span>

<span style="font-size: 14px;">**Step 3 -- Add positional encoding.** After scaling, the positional encoding $PE$ is added:</span>

$$
x_t = \tilde{e}_t + PE_t = W_E[t] \cdot \sqrt{d_{\text{model}}} + PE_t
$$

<span style="font-size: 14px;">where $PE_t \in \mathbb{R}^{d_{\text{model}}}$ is the positional encoding vector for position $t$. The combined vector $x_t$ is what enters the first Transformer block.</span>

---

## <span style="font-size: 16px;">Why Scale by $\sqrt{d_{\text{model}}}$</span>

<span style="font-size: 14px;">This is the most important conceptual question about the embedding layer. The answer lies in the relative magnitudes of embedding vectors and positional encodings.</span>

<span style="font-size: 14px;">**Embedding magnitudes are small.** The embedding matrix $W_E$ is typically initialized from a distribution like $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$ (Xavier-style). Each element of an embedding vector has standard deviation $\approx 1/\sqrt{d_{\text{model}}}$. For $d_{\text{model}} = 512$, this means per-element values are approximately $\pm 0.044$. After training, the values stay in a similar range.</span>

<span style="font-size: 14px;">**Positional encoding magnitudes are fixed at ~1.** The sinusoidal positional encoding from Vaswani et al. (2017) is:</span>

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

<span style="font-size: 14px;">Since $\sin$ and $\cos$ output values in $[-1, 1]$, every element of the positional encoding vector has magnitude at most 1.</span>

<span style="font-size: 14px;">**Without scaling, position dominates.** If embedding elements are $\pm 0.044$ and positional encoding elements are up to $\pm 1$, the positional signal is roughly 23x stronger per dimension. The model's first block would see almost entirely positional information, with the token identity buried in noise. Attention would attend based on where tokens are, not what they are.</span>

<span style="font-size: 14px;">**With scaling, the signals balance.** Multiplying the embedding by $\sqrt{512} \approx 22.6$ scales a per-element value of $0.044$ to approximately $0.044 \times 22.6 \approx 1.0$, matching the positional encoding magnitude. Both signals contribute meaningfully to the input representation.</span>

<span style="font-size: 14px;">**General principle.** For Xavier initialization with $\sigma = 1/\sqrt{d_{\text{model}}}$, scaling by $\sqrt{d_{\text{model}}}$ gives per-element standard deviation $= \sqrt{d_{\text{model}}} / \sqrt{d_{\text{model}}} = 1$, exactly matching the positional encoding scale. This is not a coincidence but a deliberate design choice.</span>

---

## <span style="font-size: 16px;">The Embedding Table</span>

<span style="font-size: 14px;">The embedding table $W_E$ is a dense matrix of shape $(V, d_{\text{model}})$. In the original Transformer, $V = 37000$ (shared BPE vocabulary) and $d_{\text{model}} = 512$, giving $37000 \times 512 \approx 19M$ parameters.</span>

<span style="font-size: 14px;">**Each row is a learned vector.** Row $t$ of $W_E$ is the representation of token $t$. At initialization, these rows are random. During training, backpropagation updates only the rows corresponding to tokens that appeared in the current batch. Rare tokens receive fewer gradient updates than frequent ones, which means common tokens develop well-trained embeddings while rare tokens may remain close to their random initialization.</span>

<span style="font-size: 14px;">**Sparse gradients.** Since the embedding lookup is an indexing operation (not a matrix multiply), the gradient with respect to $W_E$ is sparse: only the rows that were looked up receive non-zero gradients. This has implications for optimizer choice. SGD naturally handles sparse gradients, but Adam maintains running statistics for every parameter, keeping momentum and variance estimates even for rows not recently updated.</span>

<span style="font-size: 14px;">**Initialization interacts with scaling.** If the embedding is initialized with $\mathcal{N}(0, 1)$ instead of $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$, the per-element standard deviation after scaling becomes $\sqrt{d_{\text{model}}}$, far exceeding the positional encoding magnitude. The $\sqrt{d_{\text{model}}}$ scaling is designed assuming per-element standard deviation is $\mathcal{O}(1/\sqrt{d_{\text{model}}})$.</span>

---

## <span style="font-size: 16px;">Weight Tying</span>

<span style="font-size: 14px;">Vaswani et al. (2017) state: "We share the same weight matrix between the two embedding layers and the pre-softmax linear transformation." Three matrices in the model are the same physical matrix $W_E$:</span>

* <span style="font-size: 14px;">**Encoder input embedding:** maps source token IDs to vectors.</span>
* <span style="font-size: 14px;">**Decoder input embedding:** maps target token IDs to vectors.</span>
* <span style="font-size: 14px;">**Pre-softmax linear transformation:** maps decoder hidden states to logits via $\text{logits} = h \cdot W_E^T$.</span>

<span style="font-size: 14px;">**Why tie weights?** Press and Wolf (2017) showed that weight tying improves performance. The embedding maps from discrete tokens to continuous space, and the output projection maps back. Using the same matrix enforces consistency: the dot product $h \cdot W_E[t]$ measures how similar the hidden state is to the embedding of token $t$, a natural way to score next-token probabilities.</span>

<span style="font-size: 14px;">**Parameter savings.** Without tying, the model needs three separate $(V \times d_{\text{model}})$ matrices, totaling $3 \times V \times d_{\text{model}}$ parameters. With tying, there is only one, saving $2 \times 37000 \times 512 \approx 38M$ parameters.</span>

<span style="font-size: 14px;">**Scaling asymmetry.** The $\sqrt{d_{\text{model}}}$ scaling is applied only during the embedding lookup, never during the output projection. When computing $\text{logits} = h \cdot W_E^T$, no scaling is used. This is intentional: scaling adjusts input magnitudes to match positional encodings, while the output projection operates in a different context where no such adjustment is needed.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">**Vaswani et al. (2017) -- "Attention Is All You Need."** Section 3.4 states: "In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$." This single sentence is the source of the scaled embedding technique.</span>

<span style="font-size: 14px;">**Model configuration.** The base Transformer uses $d_{\text{model}} = 512$ (scaling factor $\approx 22.627$). The big Transformer uses $d_{\text{model}} = 1024$ (scaling factor $= 32$). Both share a BPE vocabulary across source and target languages. Both encoder and decoder apply the same scaled embedding at their inputs.</span>

<span style="font-size: 14px;">**Shared weights across encoder, decoder, and output.** This was inspired by Press and Wolf (2017), "Using the Output Embedding to Improve Language Models." Vaswani et al. extended the idea to also share across encoder and decoder, possible because both operate on the same vocabulary.</span>

<span style="font-size: 14px;">**Full input pipeline.** The input to the first encoder block is $x = W_E[\text{tokens}] \cdot \sqrt{d_{\text{model}}} + PE$. The decoder follows the same pattern for target tokens. The positional encoding is fixed (sinusoidal, not learned), so the only trainable parameter in the entire input pipeline is $W_E$.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a tiny vocabulary of $V = 5$ tokens and $d_{\text{model}} = 4$. The embedding table $W_E \in \mathbb{R}^{5 \times 4}$ is:</span>

$$
W_E = \begin{bmatrix} 0.02 & -0.01 & 0.03 & 0.01 \\ -0.04 & 0.05 & -0.02 & 0.03 \\ 0.01 & 0.06 & -0.03 & -0.05 \\ 0.03 & -0.02 & 0.04 & 0.02 \\ -0.01 & 0.03 & 0.01 & -0.04 \end{bmatrix}
$$

<span style="font-size: 14px;">**Input token IDs:** $[2, 0, 3]$ (a sequence of length $T = 3$).</span>

<span style="font-size: 14px;">**Step 1 -- Embedding lookup.** Retrieve rows 2, 0, and 3:</span>

* <span style="font-size: 14px;">**Token 2:** $e_2 = [0.01, 0.06, -0.03, -0.05]$</span>
* <span style="font-size: 14px;">**Token 0:** $e_0 = [0.02, -0.01, 0.03, 0.01]$</span>
* <span style="font-size: 14px;">**Token 3:** $e_3 = [0.03, -0.02, 0.04, 0.02]$</span>

<span style="font-size: 14px;">Per-element magnitudes are tiny, ranging from $-0.06$ to $0.06$.</span>

<span style="font-size: 14px;">**Step 2 -- Compute the scaling factor.** $\sqrt{d_{\text{model}}} = \sqrt{4} = 2.0$.</span>

<span style="font-size: 14px;">**Step 3 -- Multiply each embedding by the scaling factor:**</span>

* <span style="font-size: 14px;">**Token 2:** $\tilde{e}_2 = [0.01, 0.06, -0.03, -0.05] \times 2.0 = [0.02, 0.12, -0.06, -0.10]$</span>
* <span style="font-size: 14px;">**Token 0:** $\tilde{e}_0 = [0.02, -0.01, 0.03, 0.01] \times 2.0 = [0.04, -0.02, 0.06, 0.02]$</span>
* <span style="font-size: 14px;">**Token 3:** $\tilde{e}_3 = [0.03, -0.02, 0.04, 0.02] \times 2.0 = [0.06, -0.04, 0.08, 0.04]$</span>

<span style="font-size: 14px;">**Step 4 -- Compare with positional encodings.** Positional encodings for positions 0, 1, 2:</span>

* <span style="font-size: 14px;">**Position 0:** $PE_0 = [0.00, 1.00, 0.00, 1.00]$</span>
* <span style="font-size: 14px;">**Position 1:** $PE_1 = [0.84, 0.54, 0.01, 1.00]$</span>
* <span style="font-size: 14px;">**Position 2:** $PE_2 = [0.91, -0.42, 0.02, 1.00]$</span>

<span style="font-size: 14px;">**Without scaling** (Token 2 at position 2): $e_2 + PE_2 = [0.01 + 0.91,\; 0.06 - 0.42,\; -0.03 + 0.02,\; -0.05 + 1.00] = [0.92, -0.36, -0.01, 0.95]$. The positional signal dominates every dimension.</span>

<span style="font-size: 14px;">**With scaling** (Token 2 at position 2): $\tilde{e}_2 + PE_2 = [0.02 + 0.91,\; 0.12 - 0.42,\; -0.06 + 0.02,\; -0.10 + 1.00] = [0.93, -0.30, -0.04, 0.90]$. Even with $\sqrt{4} = 2$, the scaled embedding shifts the result. At $d_{\text{model}} = 512$, an element of $0.06$ becomes $0.06 \times 22.6 = 1.36$, exceeding the positional encoding magnitude and providing a strong semantic signal.</span>

---

## <span style="font-size: 16px;">Pitfalls and Common Mistakes</span>

* <span style="font-size: 14px;">**Forgetting the scaling factor entirely.** Without $\sqrt{d_{\text{model}}}$ scaling, embedding magnitudes are much smaller than positional encoding magnitudes. The model will still train (embedding weights will grow to compensate), but convergence will be slower because the attention mechanism sees mostly positional information in early epochs.</span>

* <span style="font-size: 14px;">**Using the wrong value under the square root.** The scaling factor is $\sqrt{d_{\text{model}}}$, not $\sqrt{d_k}$ (head dimension used in attention scaling) and not $\sqrt{V}$ (vocabulary size). In the base Transformer, $d_{\text{model}} = 512$, $d_k = 64$, and $V = 37000$. Confusing them changes the scaling by an order of magnitude.</span>

* <span style="font-size: 14px;">**Scaling after adding positional encoding instead of before.** The correct order is: (1) embed, (2) scale, (3) add positional encoding. If you scale after adding the positional encoding, you also scale the positional signal by $\sqrt{d_{\text{model}}}$, which defeats the purpose. The positional encodings are already at the right magnitude.</span>

* <span style="font-size: 14px;">**Passing float token IDs to the embedding layer.** Token IDs must be integers because the embedding lookup is an indexing operation. Passing float values (e.g., $2.0$ instead of $2$) causes errors in most frameworks. The embedding layer expects `LongTensor` (PyTorch) or `int32`/`int64` (TensorFlow).</span>

* <span style="font-size: 14px;">**Applying scaling during the output projection.** With weight tying, the same matrix serves as embedding and output projection. The $\sqrt{d_{\text{model}}}$ scaling must only be applied during lookup, never during the output projection $h \cdot W_E^T$. Scaling the logits would inflate them by $\sqrt{d_{\text{model}}}$, distorting softmax toward overly confident predictions.</span>

* <span style="font-size: 14px;">**Confusing embedding dimension with vocabulary size.** The table has $V$ rows and $d_{\text{model}}$ columns. The scaling uses $d_{\text{model}}$ (columns), not $V$ (rows). Mixing these up produces a completely wrong scaling factor.</span>

* <span style="font-size: 14px;">**Assuming scaling is learned.** The $\sqrt{d_{\text{model}}}$ factor is a fixed constant, not a trainable parameter. It does not receive gradient updates. Accidentally wrapping it in a learnable parameter can cause it to drift during training, breaking the magnitude-balancing property.</span>

---