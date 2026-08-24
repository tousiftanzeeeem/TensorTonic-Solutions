# <span style="font-size: 20px;">Scaled Dot-Product Attention</span>

<span style="font-size: 14px;">Scaled Dot-Product Attention is the foundational attention mechanism introduced in "Attention Is All You Need" (Vaswani et al., 2017). It computes a weighted sum of value vectors where the weights come from query-key similarity, scaled by $1/\sqrt{d_k}$ to prevent softmax saturation. This single operation is the core building block of every Transformer model.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Scaled Dot-Product Attention computes a **weighted sum of value vectors**, where the weight on each value is determined by the similarity between a **query** vector and the corresponding **key** vector. It answers: given a query, which values are most relevant, and how should they be combined?</span>

<span style="font-size: 14px;">The mechanism takes three matrix inputs: queries $Q$, keys $K$, and values $V$. It measures query-key similarity via dot products, scales the result to control variance, applies softmax to get a probability distribution over keys, and uses those probabilities to produce a weighted combination of value vectors. The output has the same number of rows as $Q$ and the same column dimension as $V$.</span>

<span style="font-size: 14px;">The "scaled" refers to the division by $\sqrt{d_k}$ before softmax. Without this factor, dot products between high-dimensional vectors grow large in magnitude, pushing softmax into regions where gradients are extremely small. The scaling is what makes the mechanism trainable. Vaswani et al. write: "We call our particular attention 'Scaled Dot-Product Attention'."</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The complete attention function:</span>

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

<span style="font-size: 14px;">Breaking this into constituent operations:</span>

<span style="font-size: 14px;">**Raw scores via dot product:**</span>

$$
S = QK^T \in \mathbb{R}^{n \times m}
$$

<span style="font-size: 14px;">where $Q \in \mathbb{R}^{n \times d_k}$ has $n$ query vectors, $K \in \mathbb{R}^{m \times d_k}$ has $m$ key vectors, and $S_{ij} = q_i^T k_j$ measures query-key compatibility.</span>

<span style="font-size: 14px;">**Scale by $\sqrt{d_k}$:**</span>

$$
S_{\text{scaled}} = \frac{QK^T}{\sqrt{d_k}}
$$

<span style="font-size: 14px;">Element-wise division ensures scores have approximately unit variance regardless of $d_k$.</span>

<span style="font-size: 14px;">**Softmax to obtain weights:**</span>

$$
W = \text{softmax}(S_{\text{scaled}}) \in \mathbb{R}^{n \times m}
$$

<span style="font-size: 14px;">Applied **row-wise**: each row of $W$ sums to 1. Entry $W_{ij}$ is the attention weight query $i$ places on key $j$.</span>

<span style="font-size: 14px;">**Weighted sum of values:**</span>

$$
O = WV \in \mathbb{R}^{n \times d_v}
$$

<span style="font-size: 14px;">where $V \in \mathbb{R}^{m \times d_v}$. Each output row $o_i = \sum_{j=1}^{m} W_{ij} v_j$ is a convex combination of value vectors.</span>

<span style="font-size: 14px;">**Dimension summary:**</span>

* <span style="font-size: 14px;">**$Q$:** $n \times d_k$ (queries)</span>
* <span style="font-size: 14px;">**$K$:** $m \times d_k$ (keys, must match query dimension)</span>
* <span style="font-size: 14px;">**$V$:** $m \times d_v$ (values, rows must match keys)</span>
* <span style="font-size: 14px;">**$S$:** $n \times m$ (score matrix)</span>
* <span style="font-size: 14px;">**$W$:** $n \times m$ (attention weights, rows sum to 1)</span>
* <span style="font-size: 14px;">**$O$:** $n \times d_v$ (output, one vector per query)</span>

---

## Why Scale by $\sqrt{d_k}$?

<span style="font-size: 14px;">This is the central design insight, and the paper provides an explicit statistical argument.</span>

<span style="font-size: 14px;">**The problem:** Assume query and key components are independent with mean 0 and variance 1. The dot product $q^T k = \sum_{i=1}^{d_k} q_i k_i$ is a sum of $d_k$ independent terms, each with mean 0 and variance $\text{Var}(q_i k_i) = E[q_i^2]E[k_i^2] = 1$. Therefore:</span>

$$
E[q^T k] = 0, \quad \text{Var}(q^T k) = d_k
$$

<span style="font-size: 14px;">The standard deviation grows as $\sqrt{d_k}$. For $d_k = 64$ (per-head dimension in the base Transformer), the standard deviation is 8. Dot products become increasingly spread out as dimension grows.</span>

<span style="font-size: 14px;">**Why this breaks softmax:** When inputs have large magnitude, softmax output becomes extremely peaked, approaching a one-hot vector. In this saturated regime, the Jacobian entries become vanishingly small, causing gradient flow to stall during backpropagation.</span>

<span style="font-size: 14px;">**The fix:** Dividing by $\sqrt{d_k}$ normalizes variance back to 1:</span>

$$
\text{Var}\!\left(\frac{q^T k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

<span style="font-size: 14px;">After scaling, scores have unit standard deviation regardless of $d_k$, keeping softmax in a regime with smooth gradients.</span>

<span style="font-size: 14px;">**Vaswani et al. (Section 3.2.1):** "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $1/\sqrt{d_k}$." They contrast this with additive attention, which does not suffer from scaling issues because its $\tanh$ output is inherently bounded.</span>

<span style="font-size: 14px;">**Concrete illustration:** With $d_k = 64$, a moderately aligned pair might produce a raw dot product around 20. Feeding $[20, 1, 1, 1]$ into softmax gives $[0.9999, 0.00003, 0.00003, 0.00003]$. After scaling by $\sqrt{64} = 8$, the scores become $[2.5, 0.125, 0.125, 0.125]$, and softmax produces $[0.74, 0.087, 0.087, 0.087]$, a much smoother distribution with healthy gradients.</span>

---

## <span style="font-size: 16px;">Step by Step</span>

<span style="font-size: 14px;">Given $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$:</span>

<span style="font-size: 14px;">1. **Compute raw scores:** $S = QK^T \in \mathbb{R}^{n \times m}$. Each $S_{ij}$ is the dot product of query $i$ with key $j$. Cost: $O(n \cdot m \cdot d_k)$.</span>

<span style="font-size: 14px;">2. **Scale:** $S_{\text{scaled}} = S / \sqrt{d_k}$. Cheap element-wise division normalizing score variance to ~1.</span>

<span style="font-size: 14px;">3. **Mask (optional):** For autoregressive decoding, add mask $M$ where $M_{ij} = 0$ for allowed positions and $M_{ij} = -\infty$ (practically $-10^9$) for forbidden ones. This forces softmax to assign zero weight to masked positions.</span>

<span style="font-size: 14px;">4. **Softmax row-wise:** $W = \text{softmax}(S_{\text{scaled}})$. Each row independently becomes a probability distribution summing to 1.</span>

<span style="font-size: 14px;">5. **Weighted sum:** $O = WV \in \mathbb{R}^{n \times d_v}$. Each $o_i = \sum_j W_{ij} v_j$ is a weighted average of value vectors. Cost: $O(n \cdot m \cdot d_v)$.</span>

<span style="font-size: 14px;">Total complexity: $O(n \cdot m \cdot (d_k + d_v))$. In self-attention ($n = m$), this is $O(n^2 \cdot d)$, quadratic in sequence length.</span>

---

## <span style="font-size: 16px;">Softmax Function</span>

<span style="font-size: 14px;">**Definition:** For $z = [z_1, \ldots, z_m]$:</span>

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{m} e^{z_j}}
$$

<span style="font-size: 14px;">**Key properties for attention:**</span>

* <span style="font-size: 14px;">**Output range:** Each element in $(0, 1)$, sum is exactly 1 (valid probability distribution).</span>
* <span style="font-size: 14px;">**Monotonicity:** Larger inputs produce larger outputs; ordering is preserved.</span>
* <span style="font-size: 14px;">**Temperature sensitivity:** Large-magnitude inputs concentrate mass on the maximum (approaches argmax); small inputs approach uniform. The $1/\sqrt{d_k}$ scaling acts as temperature control.</span>
* <span style="font-size: 14px;">**Translation invariance:** $\text{softmax}(z + c) = \text{softmax}(z)$ for any scalar $c$. Critical for numerical stability.</span>

<span style="font-size: 14px;">**Row-wise application:** In the $n \times m$ score matrix, softmax is applied independently to each of the $n$ rows. Each row is one query's distribution over all $m$ keys.</span>

<span style="font-size: 14px;">**Numerical stability:** Computing $e^{z_i}$ directly overflows for large $z_i$ ($e^{88.7} \to \infty$ in float32). The standard trick exploits translation invariance:</span>

$$
\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j} e^{z_j - \max(z)}}
$$

<span style="font-size: 14px;">After subtraction, the largest exponent is $e^0 = 1$ and all others are $\leq 1$, eliminating overflow. Every production implementation uses this trick.</span>

---

## <span style="font-size: 16px;">Attention as Soft Lookup</span>

<span style="font-size: 14px;">Attention is a differentiable, soft version of a dictionary lookup. In a hard key-value store, you provide a query, find the exact matching key, and return one value. Attention relaxes this:</span>

* <span style="font-size: 14px;">**Query ($q$):** What information are you looking for?</span>
* <span style="font-size: 14px;">**Keys ($k_1, \ldots, k_m$):** Index entries describing what each position holds.</span>
* <span style="font-size: 14px;">**Values ($v_1, \ldots, v_m$):** The actual content at each position.</span>

<span style="font-size: 14px;">Instead of returning one value from an exact match, attention scores the query against every key, converts scores to probabilities, and returns a weighted average of all values. Values whose keys best match the query dominate the output.</span>

<span style="font-size: 14px;">Because the operation is differentiable, the model learns what queries to ask ($W^Q$), what keys to advertise ($W^K$), and what values to store ($W^V$) through backpropagation. This makes attention a learned, adaptive retrieval mechanism.</span>

<span style="font-size: 14px;">**Self-attention** is the special case where $Q$, $K$, $V$ come from the same sequence (after projection). Each token asks "which tokens are relevant to me?", with keys answering and values providing information to aggregate. This is how Transformers build contextual representations without recurrence or convolution.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Vaswani et al. introduced this mechanism in "Attention Is All You Need" (NeurIPS 2017), Section 3.2.1, the first architecture built entirely on attention with no recurrence or convolution.</span>

<span style="font-size: 14px;">**Dot-product vs additive attention:** Additive attention (Bahdanau et al., 2015) uses a feed-forward network: $\text{score}(q, k) = v^T \tanh(W_1 q + W_2 k)$. Dot-product computes $q^T k$ directly. Vaswani et al.: "dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code." The $1/\sqrt{d_k}$ scaling makes it match additive attention in quality while being significantly faster on GPUs/TPUs.</span>

<span style="font-size: 14px;">**Role in Multi-Head Attention:** Rather than one attention pass with $d_{\text{model}}$-dimensional vectors, the model projects $Q$, $K$, $V$ into $h$ separate $d_k$-dimensional subspaces ($d_k = d_{\text{model}} / h$), runs scaled dot-product attention on each in parallel, concatenates, and applies a final projection. Base model: $d_{\text{model}} = 512$, $h = 8$, $d_k = d_v = 64$.</span>

<span style="font-size: 14px;">**Three uses in the Transformer:** (1) Encoder self-attention. (2) Decoder self-attention with causal mask. (3) Encoder-decoder cross-attention, where queries come from the decoder and keys/values from the encoder.</span>

---

## Numerical Example ($n = 3$, $d_k = 2$, $d_v = 2$)

<span style="font-size: 14px;">Three tokens with key/query dimension 2 and value dimension 2:</span>

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{pmatrix}, \quad V = \begin{pmatrix} 10 & 0 \\ 0 & 10 \\ 5 & 5 \end{pmatrix}
$$

### <span style="font-size: 14px;">Step 1: Raw Scores $S = QK^T$</span>

$$
S = \begin{pmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ 1 & 1 & 1 \end{pmatrix}
$$

<span style="font-size: 14px;">Query 1 aligns with Key 1 (score 1.0) and partially with Key 3 (0.5). Query 3 attends equally to all keys.</span>

### <span style="font-size: 14px;">Step 2: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$</span>

$$
S_{\text{scaled}} = \frac{S}{\sqrt{2}} = \begin{pmatrix} 0.707 & 0 & 0.354 \\ 0 & 0.707 & 0.354 \\ 0.707 & 0.707 & 0.707 \end{pmatrix}
$$

### <span style="font-size: 14px;">Step 3: Row-wise Softmax</span>

<span style="font-size: 14px;">**Row 1:** $e^{0.707} = 2.028$, $e^{0} = 1.000$, $e^{0.354} = 1.425$. Sum $= 4.453$. Weights: $[0.456, 0.225, 0.320]$.</span>

<span style="font-size: 14px;">**Row 2:** By symmetry: $[0.225, 0.456, 0.320]$.</span>

<span style="font-size: 14px;">**Row 3:** All entries equal, so uniform: $[0.333, 0.333, 0.333]$.</span>

$$
W = \begin{pmatrix} 0.456 & 0.225 & 0.320 \\ 0.225 & 0.456 & 0.320 \\ 0.333 & 0.333 & 0.333 \end{pmatrix}
$$

### <span style="font-size: 14px;">Step 4: Output $O = WV$</span>

$$
O = \begin{pmatrix} 0.456(10) + 0.225(0) + 0.320(5) & 0.456(0) + 0.225(10) + 0.320(5) \\ 0.225(10) + 0.456(0) + 0.320(5) & 0.225(0) + 0.456(10) + 0.320(5) \\ 0.333(10) + 0.333(0) + 0.333(5) & 0.333(0) + 0.333(10) + 0.333(5) \end{pmatrix} = \begin{pmatrix} 6.16 & 3.85 \\ 3.85 & 6.16 \\ 5.00 & 5.00 \end{pmatrix}
$$

<span style="font-size: 14px;">Token 1 (query $[1,0]$) attends mostly to Key 1, pulling output toward $(10, 0)$. Token 2 mirrors this toward $(0, 10)$. Token 3 (query $[1,1]$) attends uniformly, producing the exact average $(5, 5)$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Forgetting the Scaling Factor</span>

<span style="font-size: 14px;">Computing $\text{softmax}(QK^T)V$ without dividing by $\sqrt{d_k}$ is the most common mistake. For small $d_k$ in toy examples the model may still learn, masking the bug. For $d_k = 64$, unscaled dot products have standard deviation 8, pushing softmax into near-one-hot outputs. Symptoms: loss plateaus early, attention maps look like permutation matrices, gradients vanish.</span>

### <span style="font-size: 14px;">Wrong Matrix Multiplication Order</span>

<span style="font-size: 14px;">The correct product is $QK^T$ ($n \times m$). $Q^T K$ produces $d_k \times d_k$ (feature correlations, not token scores). $KQ^T$ produces $m \times n$ (transposed scores); in self-attention where $n = m$, this silently gives wrong results. Always verify: score matrix shape = (num queries, num keys).</span>

### <span style="font-size: 14px;">Softmax on the Wrong Axis</span>

<span style="font-size: 14px;">Softmax must be row-wise (over the key dimension, last axis). Column-wise softmax makes keys compete for queries rather than queries selecting among keys. In PyTorch: `softmax(scores, dim=-1)` is correct, `dim=-2` is wrong. The error is insidious because shapes remain valid and training proceeds, but attention patterns are meaningless.</span>

### <span style="font-size: 14px;">Numerical Overflow in Softmax</span>

<span style="font-size: 14px;">Even with scaling, anomalous inputs can overflow $e^{z_i}$ ($e^{88.7} \to \infty$ in float32, $e^{11.1}$ in float16). Always subtract the row maximum before exponentiation. Major frameworks handle this internally, but custom kernels must include it explicitly.</span>

### <span style="font-size: 14px;">Confusing $d_k$ with $d_{\text{model}}$</span>

<span style="font-size: 14px;">In Multi-Head Attention, scaling uses $d_k = d_{\text{model}} / h$ (per-head dimension), not $d_{\text{model}}$. With $d_{\text{model}} = 512$, $h = 8$: correct is $\sqrt{64} = 8$, not $\sqrt{512} \approx 22.6$. Over-scaling makes attention weights too uniform.</span>

### <span style="font-size: 14px;">Masking After Softmax Instead of Before</span>

<span style="font-size: 14px;">Masks must be applied before softmax (add $-\infty$ to forbidden positions). Zeroing weights after softmax breaks the sum-to-one property: $[0.3, 0.5, 0.2]$ zeroed to $[0.3, 0.5, 0]$ has total weight 0.8, systematically underweighting the output.</span>

---