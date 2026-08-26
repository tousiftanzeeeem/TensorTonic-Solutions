# <span style="font-size: 20px;">Layer Normalization</span>

<span style="font-size: 14px;">Layer Normalization (Ba et al., 2016) normalizes activations across the feature dimension for each individual sample, stabilizing training by controlling the distribution of inputs to each layer. In the original Transformer (Vaswani et al., 2017), it appears after every residual connection, making it one of the most frequently executed operations in the architecture.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Layer Normalization takes a single input vector and transforms it to have zero mean and unit variance across the feature dimension, then applies a learned affine transformation (scale and shift). It operates on each sample independently, with no dependence on other samples in the batch.</span>

<span style="font-size: 14px;">In the Transformer, Layer Normalization appears inside every encoder and decoder block. Each block contains two sub-layers (self-attention and feed-forward network), and each sub-layer is wrapped with a residual connection followed by Layer Normalization. For a 6-layer encoder with 2 sub-layers per layer, that is 12 LayerNorm operations. The decoder adds a third sub-layer (cross-attention) per block, bringing the total to 18 across 6 decoder layers, plus 12 in the encoder: 30 LayerNorm calls per forward pass in the full Transformer.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Given an input vector $x \in \mathbb{R}^d$ (one token's hidden state, where $d = d_{\text{model}}$):</span>

<span style="font-size: 14px;">**Step 1: Compute the mean** across all $d$ features:</span>

$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

<span style="font-size: 14px;">**Step 2: Compute the variance** across all $d$ features:</span>

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

<span style="font-size: 14px;">**Step 3: Normalize** each element to zero mean and unit variance:</span>

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

<span style="font-size: 14px;">where $\epsilon$ is a small constant (typically $10^{-5}$) for numerical stability.</span>

<span style="font-size: 14px;">**Step 4: Scale and shift** using learned parameters:</span>

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$x \in \mathbb{R}^d$: input activation vector. In the original Transformer, $d = d_{\text{model}} = 512$.</span>
* <span style="font-size: 14px;">$\mu$: scalar mean of $x$ across the feature dimension. Computed independently per token.</span>
* <span style="font-size: 14px;">$\sigma^2$: scalar variance of $x$ across the feature dimension. Computed independently per token.</span>
* <span style="font-size: 14px;">$\epsilon$: small constant for numerical stability. Standard value is $10^{-5}$.</span>
* <span style="font-size: 14px;">$\gamma \in \mathbb{R}^d$: learnable scale parameter, initialized to ones.</span>
* <span style="font-size: 14px;">$\beta \in \mathbb{R}^d$: learnable shift parameter, initialized to zeros.</span>
* <span style="font-size: 14px;">$\odot$: element-wise (Hadamard) multiplication.</span>

---

## <span style="font-size: 16px;">Post-Norm in the Original Transformer</span>

<span style="font-size: 14px;">The original Transformer paper (Vaswani et al., 2017) uses what is now called the **post-norm** architecture. The paper states: "We employ a residual connection around each of the two sub-layers, followed by layer normalization." The computation for each sub-layer is:</span>

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

<span style="font-size: 14px;">Here, $x$ is the input to the sub-layer, $\text{Sublayer}(x)$ is the output of either the self-attention or the feed-forward network, and LayerNorm is applied to the sum. The term "post-norm" refers to the fact that normalization happens after the residual addition.</span>

<span style="font-size: 14px;">The residual branch and the skip connection are added first, producing a potentially large and unnormalized sum, and then LayerNorm brings the combined result back to a controlled distribution. In the encoder block, the full computation is:</span>

$$
h = \text{LayerNorm}(x + \text{MultiHeadAttention}(x, x, x))
$$

$$
\text{out} = \text{LayerNorm}(h + \text{FFN}(h))
$$

<span style="font-size: 14px;">Each sub-layer gets its own LayerNorm with its own $\gamma$ and $\beta$ parameters. The two LayerNorm instances do not share parameters.</span>

---

## <span style="font-size: 16px;">Why Normalize</span>

<span style="font-size: 14px;">During training, the distribution of inputs to each layer shifts as the parameters of all preceding layers update. This phenomenon, described as **internal covariate shift** (Ioffe and Szegedy, 2015), forces each layer to continuously adapt to a moving target. Normalization fixes the input distribution to zero mean and unit variance, removing this moving target and stabilizing the optimization landscape.</span>

<span style="font-size: 14px;">Normalization enables higher learning rates because the gradients are better conditioned. Without normalization, the loss surface has sharp valleys where a slightly too-large learning rate causes divergence. With normalization, the surface is smoother, tolerating larger steps. Training without LayerNorm at all is extremely difficult for deep Transformers, even with learning rate warmup.</span>

<span style="font-size: 14px;">Normalization also improves gradient flow. In a post-norm architecture, the gradient passes through LayerNorm at every layer. Each LayerNorm rescales the gradient to a controlled magnitude, preventing both vanishing and exploding gradients. This is why Transformers can be stacked to 6, 12, or even 96 layers while remaining trainable.</span>

---

## <span style="font-size: 16px;">The Learnable Parameters</span>

<span style="font-size: 14px;">Layer Normalization has two learnable parameter vectors: $\gamma \in \mathbb{R}^d$ (scale) and $\beta \in \mathbb{R}^d$ (shift). Together, they form an affine transformation applied element-wise to the normalized output.</span>

<span style="font-size: 14px;">$\gamma$ is initialized to all ones and $\beta$ is initialized to all zeros. At initialization, LayerNorm acts as pure normalization. During training, the network adjusts $\gamma$ and $\beta$ through backpropagation to learn the optimal distribution for each feature dimension.</span>

<span style="font-size: 14px;">These parameters exist because raw normalization is too restrictive. Forcing all hidden states to zero mean and unit variance constrains representational capacity. If the network learns $\gamma_i = \sigma$ and $\beta_i = \mu$ for the original pre-normalization statistics, it recovers the unnormalized activation exactly, so LayerNorm never reduces model capacity.</span>

<span style="font-size: 14px;">For the original Transformer with $d_{\text{model}} = 512$, each LayerNorm instance has $2 \times 512 = 1{,}024$ parameters. With 30 LayerNorm instances across the full encoder-decoder, that is $30{,}720$ learnable parameters total, tiny compared to attention and FFN weights but critical for model quality.</span>

---

## <span style="font-size: 16px;">LayerNorm vs BatchNorm</span>

<span style="font-size: 14px;">Batch Normalization (Ioffe and Szegedy, 2015) was the dominant normalization before LayerNorm. The fundamental difference is the normalization dimension:</span>

* <span style="font-size: 14px;">**Batch Normalization** computes mean and variance across the batch dimension: for each feature, aggregate statistics over all samples in the mini-batch.</span>
* <span style="font-size: 14px;">**Layer Normalization** computes mean and variance across the feature dimension: for each sample, aggregate statistics over all features.</span>

<span style="font-size: 14px;">For Transformers, LayerNorm is strongly preferred:</span>

* <span style="font-size: 14px;">**No batch dependence.** BatchNorm introduces a dependency between samples. LayerNorm computes everything per-sample, so each token is normalized independently.</span>
* <span style="font-size: 14px;">**Variable sequence lengths.** BatchNorm at position $t$ averages across all sequences that reach position $t$, creating inconsistent statistics. LayerNorm normalizes each token by its own features, unaffected by sequence length.</span>
* <span style="font-size: 14px;">**Train-test consistency.** BatchNorm maintains running statistics during training and uses them at inference, creating a gap. LayerNorm has no running statistics and behaves identically at train and test time.</span>
* <span style="font-size: 14px;">**Small batch sizes.** BatchNorm with batch size 1 produces degenerate statistics (variance is 0). LayerNorm works perfectly with any batch size, including single-sample inference.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The Transformer paper (Vaswani et al., 2017) adopts Layer Normalization from Ba, Kiros, and Hinton (2016), who proposed it as a batch-independent alternative to Batch Normalization. The paper does not devote much discussion to the choice, treating it as a known technique. The relevant passage states: "We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is $\text{LayerNorm}(x + \text{Sublayer}(x))$, where $\text{Sublayer}(x)$ is the function implemented by the sub-layer itself."</span>

<span style="font-size: 14px;">The paper uses dropout with rate 0.1 applied to sub-layer outputs before the residual addition, meaning the actual computation is $\text{LayerNorm}(x + \text{Dropout}(\text{Sublayer}(x)))$. The combination of residual connections and LayerNorm follows a pattern established by He et al. (2016) in ResNets. Without LayerNorm, stacking 6 encoder and 6 decoder layers would be difficult to train, as residual additions would accumulate and activation magnitudes would grow without bound.</span>

---

## Numerical Example ($d = 4$)

<span style="font-size: 14px;">Consider a single token with a 4-dimensional hidden state.</span>

<span style="font-size: 14px;">**Input:** $x = (2.0, \; 6.0, \; -2.0, \; 4.0)$, $\gamma = (1, 1, 1, 1)$, $\beta = (0, 0, 0, 0)$, $\epsilon = 0$.</span>

<span style="font-size: 14px;">1. **Mean:**</span>

$$
\mu = \frac{1}{4}(2.0 + 6.0 + (-2.0) + 4.0) = \frac{10.0}{4} = 2.5
$$

<span style="font-size: 14px;">2. **Variance:**</span>

$$
\sigma^2 = \frac{1}{4}\left((2.0 - 2.5)^2 + (6.0 - 2.5)^2 + (-2.0 - 2.5)^2 + (4.0 - 2.5)^2\right)
$$

$$
= \frac{1}{4}(0.25 + 12.25 + 20.25 + 2.25) = \frac{35.0}{4} = 8.75
$$

<span style="font-size: 14px;">3. **Standard deviation:** $\sqrt{8.75} \approx 2.9580$</span>

<span style="font-size: 14px;">4. **Normalize** by subtracting the mean and dividing by the standard deviation:</span>

$$
\hat{x} = \frac{(-0.5, \; 3.5, \; -4.5, \; 1.5)}{2.9580} \approx (-0.1690, \; 1.1832, \; -1.5213, \; 0.5071)
$$

<span style="font-size: 14px;">Verification: the mean of $\hat{x}$ is approximately 0 and the variance is approximately 1.</span>

<span style="font-size: 14px;">5. **Scale and shift** (with $\gamma = 1$, $\beta = 0$, output equals $\hat{x}$):</span>

$$
\text{LayerNorm}(x) \approx (-0.1690, \; 1.1832, \; -1.5213, \; 0.5071)
$$

<span style="font-size: 14px;">**With non-trivial parameters.** Suppose $\gamma = (0.5, \; 2.0, \; 1.0, \; 0.8)$ and $\beta = (0.1, \; -0.5, \; 0.0, \; 0.3)$:</span>

$$
\text{LayerNorm}(x) = \gamma \odot \hat{x} + \beta \approx (0.0155, \; 1.8664, \; -1.5213, \; 0.7057)
$$

<span style="font-size: 14px;">Feature 2 was amplified by $\gamma_2 = 2.0$, feature 1 was compressed by $\gamma_1 = 0.5$ and shifted up by $\beta_1 = 0.1$, and feature 4 was shifted up by $\beta_4 = 0.3$. This demonstrates how the learned parameters reshape the normalized distribution.</span>

---

## <span style="font-size: 16px;">Post-Norm vs Pre-Norm</span>

<span style="font-size: 14px;">The original Transformer uses **post-norm**, where LayerNorm is applied after the residual addition:</span>

$$
\text{Post-Norm: } \quad x' = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

<span style="font-size: 14px;">GPT-2 (Radford et al., 2019) introduced **pre-norm**, where LayerNorm is applied before the sub-layer:</span>

$$
\text{Pre-Norm: } \quad x' = x + \text{Sublayer}(\text{LayerNorm}(x))
$$

<span style="font-size: 14px;">The difference has major consequences for training deep models:</span>

* <span style="font-size: 14px;">**Gradient flow in post-norm:** The gradient must pass through LayerNorm at every layer. In very deep networks (24+ layers), this repeated modification can attenuate gradients. The original Transformer mitigates this with careful learning rate warmup.</span>
* <span style="font-size: 14px;">**Gradient flow in pre-norm:** The residual connection provides a clean identity path. The gradient includes a direct $\frac{\partial x'}{\partial x} = 1$ term bypassing the sub-layer entirely, keeping gradients well-conditioned regardless of depth.</span>
* <span style="font-size: 14px;">**Warmup sensitivity:** Post-norm requires careful warmup; without it, training often fails. Pre-norm is significantly less sensitive.</span>
* <span style="font-size: 14px;">**Final quality:** Some research (Xiong et al., 2020) suggests post-norm can achieve slightly higher final quality when it converges, but it is harder to get working and the quality gap is small.</span>

<span style="font-size: 14px;">The modern consensus is clear: LLaMA, GPT-2, GPT-3, PaLM, Mistral, and virtually all decoder-only LLMs use pre-norm. The original Transformer's post-norm is historically important but has been superseded. Encoder-only models like BERT still use post-norm as originally designed.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Normalizing over the wrong dimension.** LayerNorm normalizes across the feature (last) dimension, not the batch or sequence dimension. Computing statistics over `dim=0` (batch) or `dim=1` (sequence) produces silently incorrect outputs that degrade model quality without raising errors.</span>
* <span style="font-size: 14px;">**Setting epsilon too small.** Using $\epsilon = 10^{-12}$ causes numerical instability when the variance is near zero, particularly in float16 or bfloat16 mixed-precision training. The gradient through $1/\sqrt{\sigma^2 + \epsilon}$ explodes when $\sigma^2 + \epsilon$ is tiny. The standard value of $10^{-5}$ balances stability and accuracy.</span>
* <span style="font-size: 14px;">**Confusing post-norm and pre-norm placement.** The original Transformer uses $\text{LayerNorm}(x + \text{Sublayer}(x))$ (post-norm). Most modern models use $x + \text{Sublayer}(\text{LayerNorm}(x))$ (pre-norm). Implementing one when intending the other changes gradient flow and can cause training instability.</span>
* <span style="font-size: 14px;">**Forgetting gamma and beta.** Omitting the learnable parameters $\gamma$ and $\beta$ forces all activations to exactly zero mean and unit variance. The network loses the ability to learn per-feature scaling, degrading quality significantly. Custom implementations often forget them.</span>
* <span style="font-size: 14px;">**Confusing LayerNorm with BatchNorm.** In a tensor of shape $(B, T, d)$, BatchNorm normalizes over dimension 0 ($B$) while LayerNorm normalizes over dimension 2 ($d$). Swapping the two produces completely wrong statistics and breaks training. BatchNorm also has running mean/variance that differ between training and inference, while LayerNorm has no such distinction.</span>
* <span style="font-size: 14px;">**Applying LayerNorm to the wrong tensor in the residual block.** In post-norm, LayerNorm takes $x + \text{Sublayer}(x)$ as input. In pre-norm, LayerNorm takes $x$ alone. Normalizing the sub-layer output by itself, without the residual, breaks the intended architecture.</span>
* <span style="font-size: 14px;">**Initializing gamma to zeros instead of ones.** If $\gamma$ starts at zero, all LayerNorm outputs are zero (before the bias), effectively disabling the sub-layer at initialization. Only the residual path carries signal, causing training instability. The correct initialization is $\gamma = 1$, $\beta = 0$.</span>

---