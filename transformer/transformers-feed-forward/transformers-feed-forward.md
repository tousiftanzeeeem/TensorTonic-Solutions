# <span style="font-size: 20px;">Position-wise Feed-Forward Network</span>

<span style="font-size: 14px;">The position-wise feed-forward network (FFN) is the other major sublayer in each Transformer block, paired with multi-head self-attention. It consists of two linear transformations with a ReLU activation in between: $\text{FFN}(x) = \max(0,\, xW_1 + b_1)W_2 + b_2$. The FFN is applied to each position independently and identically, acting as a per-token two-layer MLP that expands the representation to a higher dimension, applies a nonlinearity, and contracts back. In Vaswani et al. (2017), the model dimension is $d_{\text{model}} = 512$ and the inner dimension is $d_{ff} = 2048$, giving a 4x expansion ratio.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The position-wise FFN is a two-layer fully connected network applied independently to the representation at each sequence position. "Position-wise" means the same weight matrices $W_1$, $b_1$, $W_2$, $b_2$ are shared across all positions in the sequence, but there is no information flow between positions within this sublayer. Each token's $d_{\text{model}}$-dimensional vector is processed in isolation.</span>

<span style="font-size: 14px;">The architecture is straightforward: the first linear layer projects from $d_{\text{model}}$ to a larger dimension $d_{ff}$, a ReLU activation introduces nonlinearity, and the second linear layer projects back from $d_{ff}$ to $d_{\text{model}}$. This expand-activate-contract pattern gives each token access to a wider computational workspace before compressing the result back to the model's working dimension.</span>

<span style="font-size: 14px;">In the Transformer block, the FFN always follows the multi-head attention sublayer. Both sublayers are wrapped with residual connections and layer normalization: $\text{LayerNorm}(x + \text{Sublayer}(x))$. The FFN receives attention-processed representations and refines them token by token.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">The FFN Formula</span>

<span style="font-size: 14px;">Given an input vector $x \in \mathbb{R}^{d_{\text{model}}}$ at a single position, the feed-forward network computes:</span>

$$
\text{FFN}(x) = \max(0,\, xW_1 + b_1)\, W_2 + b_2
$$

<span style="font-size: 14px;">Breaking this into its three stages:</span>

* <span style="font-size: 14px;">**Expand:** Compute $h = xW_1 + b_1$, where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ and $b_1 \in \mathbb{R}^{d_{ff}}$. This projects the input from dimension $d_{\text{model}}$ to $d_{ff}$.</span>
* <span style="font-size: 14px;">**ReLU:** Compute $a = \max(0, h)$, applying the rectified linear unit element-wise. Every negative component is zeroed out; positive components pass through unchanged.</span>
* <span style="font-size: 14px;">**Contract:** Compute $\text{FFN}(x) = aW_2 + b_2$, where $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ and $b_2 \in \mathbb{R}^{d_{\text{model}}}$. This projects back from $d_{ff}$ to $d_{\text{model}}$.</span>

### <span style="font-size: 14px;">Dimensions in the Original Transformer</span>

<span style="font-size: 14px;">Vaswani et al. set $d_{\text{model}} = 512$ and $d_{ff} = 2048$. The parameter count for one FFN sublayer: $W_1$ has $512 \times 2048 = 1{,}048{,}576$ parameters, $b_1$ has $2048$, $W_2$ has $2048 \times 512 = 1{,}048{,}576$ parameters, $b_2$ has $512$. Total: $2{,}099{,}712$ parameters per sublayer. With 6 encoder and 6 decoder layers, the FFN sublayers alone account for roughly $25.2$ million parameters.</span>

### <span style="font-size: 14px;">Batch and Sequence Dimensions</span>

<span style="font-size: 14px;">In practice, the input is a tensor $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ where $B$ is the batch size and $T$ is the sequence length. The FFN applies identically across both the $B$ and $T$ dimensions. This is equivalent to reshaping to $(BT) \times d_{\text{model}}$, applying a standard two-layer MLP, and reshaping back.</span>

---

## <span style="font-size: 16px;">Why 4x Expansion</span>

### <span style="font-size: 14px;">Increased Computational Capacity</span>

<span style="font-size: 14px;">The expansion from $d_{\text{model}}$ to $4 \cdot d_{\text{model}}$ gives the FFN a much richer intermediate space. With 2048 hidden units, the network can represent a far more complex function of the 512-dimensional input. Each hidden unit detects a different linear pattern, and ReLU selectively activates a subset, effectively choosing which features are relevant for each token.</span>

### <span style="font-size: 14px;">Bottleneck Architecture</span>

<span style="font-size: 14px;">The expand-contract design is a bottleneck: the first projection scatters the input into a diverse set of feature detectors, ReLU sparsifies the representation by zeroing out roughly half the dimensions, and the second projection combines the surviving features back into $d_{\text{model}}$ space. This forces the network to learn an efficient encoding rather than simply memorizing the expanded representation.</span>

### <span style="font-size: 14px;">The 4x Ratio Is Empirical</span>

<span style="font-size: 14px;">The ratio $d_{ff} / d_{\text{model}} = 4$ was chosen by Vaswani et al. based on experimental results, not derived from theory. GPT-3 maintains this convention. Models using gated activations like SwiGLU (e.g., LLaMA) typically use $d_{ff} = \frac{8}{3} \cdot d_{\text{model}}$ to keep parameter count similar despite the additional gating projection. The key insight is that the FFN needs a meaningfully larger hidden dimension to provide sufficient capacity, but the exact multiplier is tunable.</span>

---

## <span style="font-size: 16px;">Position-Wise Application</span>

### <span style="font-size: 14px;">Same Weights at Every Position</span>

<span style="font-size: 14px;">The FFN uses exactly the same parameters $W_1, b_1, W_2, b_2$ at every sequence position. Position 0 and position 511 are processed by the identical function. Vaswani et al. describe this as equivalent to two $1 \times 1$ convolutions: one with 2048 output channels, then one with 512 output channels. The kernel size of 1 means each position is processed independently, with no spatial receptive field.</span>

### <span style="font-size: 14px;">No Cross-Position Interaction</span>

<span style="font-size: 14px;">Within the FFN sublayer, position $i$ has absolutely no access to positions $j \neq i$. All cross-position communication happens exclusively in the multi-head attention sublayer. This clean separation is a defining architectural choice: attention handles token-to-token interaction (the "mixing" step), and the FFN handles per-token transformation (the "processing" step).</span>

### <span style="font-size: 14px;">Computational Parallelism</span>

<span style="font-size: 14px;">Because positions are independent within the FFN, all $T$ positions can be processed simultaneously as a single batched matrix multiplication. There is no sequential dependency, making the FFN perfectly parallelizable on GPUs. This contrasts with recurrent architectures where each timestep depends on the previous hidden state.</span>

---

## <span style="font-size: 16px;">Why ReLU</span>

### <span style="font-size: 14px;">Simplicity and Effectiveness</span>

<span style="font-size: 14px;">ReLU ($\max(0, x)$) is the simplest widely-used nonlinear activation. It has zero computational overhead beyond a comparison, introduces sparsity by zeroing out negative values, and its gradient is trivially 0 or 1. At the time of publication (2017), ReLU was the dominant activation function in deep learning, making it a natural default choice.</span>

### <span style="font-size: 14px;">The Role of Nonlinearity</span>

<span style="font-size: 14px;">Without the ReLU between the two linear layers, the FFN would collapse to a single linear transformation: $(xW_1 + b_1)W_2 + b_2 = x(W_1 W_2) + (b_1 W_2 + b_2)$. This is just $xW + b$, so the two-layer network would have no more expressive power than a single linear layer. ReLU breaks this linearity, enabling the FFN to approximate nonlinear functions. The nonlinearity is what makes the expansion to $d_{ff}$ meaningful.</span>

### <span style="font-size: 14px;">Later Replacements</span>

<span style="font-size: 14px;">Subsequent models replaced ReLU with smoother alternatives. GPT-2 uses GELU (Hendrycks and Gimpel, 2016), which provides a smooth probabilistic gate instead of a hard cutoff at zero. LLaMA and Mistral use SwiGLU (Shazeer, 2020), a gated activation that multiplies two projections element-wise with a SiLU nonlinearity. The FFN structure (expand, activate, contract) remains the same; only the activation function changes.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

### <span style="font-size: 14px;">Vaswani et al. (2017)</span>

<span style="font-size: 14px;">In "Attention Is All You Need," Section 3.3 states: "In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between." The paper specifies $d_{\text{model}} = 512$, $d_{ff} = 2048$ for the base model, and $d_{\text{model}} = 1024$, $d_{ff} = 4096$ for the big model.</span>

### <span style="font-size: 14px;">Analogy to 1x1 Convolutions</span>

<span style="font-size: 14px;">The paper notes that the position-wise FFN "can also be described as two convolutions with kernel size 1." A $1 \times 1$ convolution over a sequence treats each position independently and transforms the channel dimension, exactly like a shared linear layer applied at every position. This connection was intuitive for researchers from the CNN literature, where $1 \times 1$ convolutions (Network-in-Network, Lin et al. 2013) were already well established.</span>

### <span style="font-size: 14px;">Different Parameters Across Layers</span>

<span style="font-size: 14px;">While the FFN shares parameters across positions within a layer, different Transformer layers have different FFN parameters. The paper states: "While the linear transformations are the same across different positions, they use different parameters from layer to layer." Each layer learns a different nonlinear transformation of the per-token representation.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a minimal example with $d_{\text{model}} = 4$ and $d_{ff} = 8$ to trace the full computation for a single position.</span>

### <span style="font-size: 14px;">Setup</span>

<span style="font-size: 14px;">Input: $x = [1.0,\; -0.5,\; 0.3,\; 0.8]$ (shape $1 \times 4$). Let $b_1 = \mathbf{0}$ for clarity. After the first linear layer $h = xW_1$:</span>

$$
h = [2.1,\; -0.7,\; 1.4,\; -1.2,\; 0.0,\; 0.9,\; -0.3,\; 1.8]
$$

### <span style="font-size: 14px;">Apply ReLU</span>

<span style="font-size: 14px;">$a = \max(0, h)$ element-wise:</span>

$$
a = [2.1,\; 0.0,\; 1.4,\; 0.0,\; 0.0,\; 0.9,\; 0.0,\; 1.8]
$$

<span style="font-size: 14px;">Four of eight dimensions survive (indices 0, 2, 5, 7). The other four are zeroed out -- a 50% sparsity rate, typical for ReLU on zero-mean inputs.</span>

### <span style="font-size: 14px;">Contract Back</span>

<span style="font-size: 14px;">Multiply by $W_2$ (shape $8 \times 4$) and add $b_2$. Only the rows of $W_2$ at indices 0, 2, 5, 7 contribute. Suppose the output is:</span>

$$
\text{FFN}(x) = [0.7,\; -0.2,\; 1.1,\; 0.4]
$$

### <span style="font-size: 14px;">Residual Connection</span>

<span style="font-size: 14px;">The Transformer adds the residual: $\text{output} = \text{LayerNorm}(x + \text{FFN}(x)) = \text{LayerNorm}([1.7,\; -0.7,\; 1.4,\; 1.2])$. The FFN refines the token's representation while the skip connection preserves the original signal.</span>

### <span style="font-size: 14px;">Key Observations</span>

* <span style="font-size: 14px;">The input was 4-dimensional, expanded to 8-dimensional, then compressed back to 4-dimensional.</span>
* <span style="font-size: 14px;">ReLU created sparsity, selecting which expanded features survive.</span>
* <span style="font-size: 14px;">The entire computation was local to one position -- no other token was involved.</span>
* <span style="font-size: 14px;">The zero entries in $a$ mean only a subset of $W_2$'s rows contributed, making the output a position-dependent linear combination of selected learned features.</span>

---

## <span style="font-size: 16px;">The FFN's Role in the Transformer</span>

### <span style="font-size: 14px;">Attention Mixes, FFN Processes</span>

<span style="font-size: 14px;">The Transformer block alternates between two fundamentally different operations. Multi-head attention allows every position to gather information from all other positions, producing a context-aware representation. The FFN then applies a nonlinear transformation to each token independently. Attention is about communication between tokens; the FFN is about computation within each token.</span>

### <span style="font-size: 14px;">The FFN as a Key-Value Memory</span>

<span style="font-size: 14px;">Interpretability research (Geva et al., 2021) has shown that FFN layers function as implicit key-value memories. Each row of $W_1$ acts as a "key" that matches certain input patterns (triggering a high pre-ReLU activation), and the corresponding column of $W_2$ acts as a "value" added to the output when that key matches. ReLU determines which keys are active. This explains why FFNs store factual knowledge: specific $W_1$ rows detect patterns like "the capital of France is" and the corresponding $W_2$ columns push the representation toward "Paris."</span>

### <span style="font-size: 14px;">Parameter Distribution</span>

<span style="font-size: 14px;">In the base model, the FFN sublayers account for roughly two-thirds of total parameters. Attention has $4 \cdot d_{\text{model}}^2$ parameters per layer (for $Q, K, V$, and output projections), while the FFN has $2 \cdot d_{\text{model}} \cdot d_{ff} = 8 \cdot d_{\text{model}}^2$. The FFN is the larger component, aligning with its role as the primary site for storing learned knowledge.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Placing ReLU after the second linear layer instead of between the two layers.** The correct structure is Linear-ReLU-Linear. If ReLU is applied after $W_2$ instead of after $W_1$, everything up to the activation is a single linear function ($x W_1 W_2$), and the subsequent ReLU merely clips the output. This eliminates the benefit of the expanded hidden dimension.</span>

* <span style="font-size: 14px;">**Using the wrong expansion ratio.** Implementing $d_{ff} = d_{\text{model}}$ (ratio 1) instead of $d_{ff} = 4 \cdot d_{\text{model}}$ dramatically reduces capacity. The standard ratio is 4 for ReLU-based FFNs and $\frac{8}{3}$ for gated variants like SwiGLU. Always check which convention the target architecture uses.</span>

* <span style="font-size: 14px;">**Confusing the FFN with an attention mechanism.** The FFN has no query-key-value structure, no softmax, and no interaction between positions. It is a simple two-layer MLP applied per token. Conflating the two sublayers leads to implementations that accidentally violate position-wise independence.</span>

* <span style="font-size: 14px;">**Forgetting the bias terms.** The original Transformer FFN includes bias vectors $b_1$ and $b_2$. Some modern architectures (PaLM, LLaMA) remove biases, but Vaswani et al. explicitly includes them. Omitting biases changes the function class, especially for inputs near the origin where $b_1$ shifts the ReLU activation boundary.</span>

* <span style="font-size: 14px;">**Applying the FFN across positions instead of independently per position.** The FFN must treat each position as a separate sample. A common error is reshaping so the sequence dimension gets mixed into the feature dimension. The correct implementation applies linear layers to the last dimension only, broadcasting over batch and sequence.</span>

* <span style="font-size: 14px;">**Forgetting the residual connection and layer normalization.** The FFN is always wrapped in a residual: $\text{output} = \text{LayerNorm}(x + \text{FFN}(x))$. Without the skip connection, gradient flow degrades in deep models and training becomes unstable.</span>

---