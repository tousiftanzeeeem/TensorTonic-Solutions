# <span style="font-size: 20px;">Sinusoidal Positional Encoding</span>

<span style="font-size: 14px;">Transformers process all tokens in parallel, so unlike RNNs, they have no built-in notion of token order. Sinusoidal positional encoding, introduced in Vaswani et al. (2017), solves this by adding a fixed, deterministic signal to each token embedding that encodes its position in the sequence. The encoding uses sine and cosine functions at different frequencies, producing a unique positional fingerprint for every position. The output is a matrix of shape (seq_len, d_model) with all values in [-1, 1].</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Sinusoidal positional encoding is a fixed signal added element-wise to token embeddings before they enter the Transformer encoder or decoder stack. It requires no learned parameters. For a given position and dimension, the encoding value is computed by evaluating either a sine or cosine function at a specific frequency.</span>

<span style="font-size: 14px;">Each position gets a unique vector in $\mathbb{R}^{d_{\text{model}}}$, computed once from a deterministic formula. Given a token embedding $e \in \mathbb{R}^{d_{\text{model}}}$ at position $pos$, the input to the Transformer is:</span>

$$
\text{input}(pos) = e(pos) + PE(pos)
$$

<span style="font-size: 14px;">where $PE(pos) \in \mathbb{R}^{d_{\text{model}}}$ is the positional encoding vector. This is element-wise addition, so both vectors share the same dimensionality. The encoding adds zero learnable parameters and is identical across every forward pass, every batch, every epoch.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The positional encoding for position $pos$ and dimension index $i$ is defined by two formulas:</span>

$$
PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

<span style="font-size: 14px;">where $pos$ is the 0-indexed sequence position, $i$ is the dimension pair index ranging from $0$ to $d_{\text{model}}/2 - 1$, and $d_{\text{model}}$ is the model dimension. Even-indexed dimensions ($0, 2, 4, \ldots$) use sine; odd-indexed dimensions ($1, 3, 5, \ldots$) use cosine.</span>

<span style="font-size: 14px;">The shared argument for each (sin, cos) pair is:</span>

$$
\theta_i = \frac{pos}{10000^{2i / d_{\text{model}}}}
$$

<span style="font-size: 14px;">Dimensions $2i$ and $2i+1$ form a pair evaluated at the same angle $\theta_i$. Together they trace a point on the unit circle for that frequency, encoding position as a rotation. The output matrix $PE \in \mathbb{R}^{L \times d_{\text{model}}}$ has all values in $[-1, 1]$.</span>

---

## <span style="font-size: 16px;">The Frequency Schedule</span>

<span style="font-size: 14px;">The denominator $10000^{2i / d_{\text{model}}}$ creates a geometric progression of wavelengths across dimensions. Rewriting as the wavelength for dimension pair $i$:</span>

$$
\lambda_i = 2\pi \cdot 10000^{2i / d_{\text{model}}}
$$

<span style="font-size: 14px;">When $i = 0$ (dimensions 0 and 1), $\lambda_0 = 2\pi \approx 6.28$. This is the shortest wavelength and highest frequency. The sin/cos values change rapidly from one position to the next, encoding fine-grained positional differences.</span>

<span style="font-size: 14px;">When $i = d_{\text{model}}/2 - 1$ (the last pair), $\lambda_{\max} = 2\pi \cdot 10000 \approx 62{,}832$. This is the longest wavelength and lowest frequency, encoding coarse, large-scale positional information that changes slowly across positions.</span>

<span style="font-size: 14px;">Between these extremes, wavelengths grow geometrically. For the original Transformer with $d_{\text{model}} = 512$, the ratio between consecutive wavelengths is $10000^{2/512} \approx 1.036$, so each successive pair has a wavelength about 3.6% longer. This multi-scale structure is analogous to a clock: the seconds digit changes fastest, the hours digit slowest, and their combination uniquely identifies a time.</span>

---

## <span style="font-size: 16px;">Why Sinusoidal</span>

<span style="font-size: 14px;">Vaswani et al. chose sinusoidal functions for several specific reasons:</span>

<span style="font-size: 14px;">**No learning required.** The encoding is computed from a fixed formula, adds zero parameters, and cannot overfit to the training data's position distribution.</span>

<span style="font-size: 14px;">**Generalization to longer sequences.** The sinusoidal functions are defined for any real-valued input, so the encoding can be computed for positions beyond those seen during training. A model trained on length 512 can produce valid encodings for position 5000. Values remain in [-1, 1] and the multi-frequency structure is preserved. This extrapolation property was a primary motivation for choosing sinusoidal over learned encodings.</span>

<span style="font-size: 14px;">**Relative position via linear combination.** For any fixed offset $k$, $PE(pos + k)$ can be expressed as a linear function of $PE(pos)$. For each (sin, cos) pair at frequency index $i$:</span>

$$
\begin{pmatrix} \sin(\theta_{pos+k}) \\ \cos(\theta_{pos+k}) \end{pmatrix} = \begin{pmatrix} \cos(\theta_k) & \sin(\theta_k) \\ -\sin(\theta_k) & \cos(\theta_k) \end{pmatrix} \begin{pmatrix} \sin(\theta_{pos}) \\ \cos(\theta_{pos}) \end{pmatrix}
$$

<span style="font-size: 14px;">where $\theta_{pos} = pos / 10000^{2i/d_{\text{model}}}$ and $\theta_k = k / 10000^{2i/d_{\text{model}}}$. The rotation matrix depends only on $k$, not on $pos$. The model can learn to attend to relative positions by learning a linear transformation, since the relationship between positions separated by the same offset is always the same rotation.</span>

<span style="font-size: 14px;">**Unique encoding per position.** The combination of sinusoids at different frequencies produces a unique vector for each position, giving the model an unambiguous positional signal.</span>

---

## <span style="font-size: 16px;">Even/Odd Pairing</span>

<span style="font-size: 14px;">The encoding assigns sine to even-indexed dimensions and cosine to odd-indexed dimensions. Dimensions $2i$ and $2i+1$ form a pair sharing the same frequency but using complementary trigonometric functions.</span>

<span style="font-size: 14px;">This pairing is not arbitrary. Together, $\sin(\theta)$ and $\cos(\theta)$ encode a position as a point on the unit circle. Given both values, the angle is uniquely determined (up to $2\pi$ periodicity). If only sine were used, positions at $\theta$ and $\pi - \theta$ would be indistinguishable. The cosine resolves this ambiguity.</span>

<span style="font-size: 14px;">The (sin, cos) pair at each frequency also enables the linear combination property. The rotation matrix that maps $PE(pos)$ to $PE(pos + k)$ operates on 2D blocks of (sin, cos) pairs. If both dimensions used the same function, the rotation would not work as a matrix multiplication.</span>

<span style="font-size: 14px;">In implementation, the encoding is constructed by creating a position vector $pos = [0, 1, \ldots, L-1]^T$ of shape $(L, 1)$ and a frequency vector $\text{div\_term} = 10000^{-2i / d_{\text{model}}}$ of shape $(1, d_{\text{model}}/2)$. Their outer product gives angles $\Theta$ of shape $(L, d_{\text{model}}/2)$. Applying sin and cos then interleaving produces the final $(L, d_{\text{model}})$ matrix.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Vaswani et al. introduced sinusoidal positional encoding in "Attention Is All You Need" (2017), Section 3.5. The paper states: "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence."</span>

<span style="font-size: 14px;">The authors also tested learned positional embeddings and report in Table 3, row (E), that they produced nearly identical results on the base model. They chose sinusoidal for the final model because it "would allow the model to extrapolate to sequence lengths longer than the ones encountered during training."</span>

<span style="font-size: 14px;">The original Transformer used $d_{\text{model}} = 512$ (base) and $d_{\text{model}} = 1024$ (big), with maximum sequence length of 512 tokens. The positional encoding was added at the bottom of both encoder and decoder stacks.</span>

<span style="font-size: 14px;">Since 2017, most large language models have moved away from sinusoidal encoding. BERT (2019) and GPT-2 (2019) use learned positional embeddings. RoPE (Su et al., 2021), used in LLaMA and similar models, rotates query and key vectors by position-dependent angles so the attention dot product directly encodes relative position $(i - j)$. ALiBi (Press et al., 2022) adds a position-dependent bias to attention scores. Sinusoidal encoding remains foundational because it introduced the insight that position can be encoded via multi-frequency periodic functions.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider $d_{\text{model}} = 4$ and $\text{seq\_len} = 3$. The output is a $3 \times 4$ matrix. Dimension pairs: $(i=0)$ for columns 0,1 and $(i=1)$ for columns 2,3.</span>

<span style="font-size: 14px;">**Frequency computation.** Denominators for each pair:</span>

$$
\text{div}_0 = 10000^{0/4} = 10000^0 = 1, \quad \text{div}_1 = 10000^{2/4} = 10000^{0.5} = 100
$$

<span style="font-size: 14px;">**Position 0.** Angles: $\theta_0 = 0/1 = 0$ and $\theta_1 = 0/100 = 0$.</span>

$$
PE(0, 0) = \sin(0) = 0.0000, \quad PE(0, 1) = \cos(0) = 1.0000
$$

$$
PE(0, 2) = \sin(0) = 0.0000, \quad PE(0, 3) = \cos(0) = 1.0000
$$

<span style="font-size: 14px;">**Position 1.** Angles: $\theta_0 = 1/1 = 1$ and $\theta_1 = 1/100 = 0.01$.</span>

$$
PE(1, 0) = \sin(1) = 0.8415, \quad PE(1, 1) = \cos(1) = 0.5403
$$

$$
PE(1, 2) = \sin(0.01) = 0.0100, \quad PE(1, 3) = \cos(0.01) = 0.9999
$$

<span style="font-size: 14px;">**Position 2.** Angles: $\theta_0 = 2/1 = 2$ and $\theta_1 = 2/100 = 0.02$.</span>

$$
PE(2, 0) = \sin(2) = 0.9093, \quad PE(2, 1) = \cos(2) = -0.4161
$$

$$
PE(2, 2) = \sin(0.02) = 0.0200, \quad PE(2, 3) = \cos(0.02) = 0.9998
$$

<span style="font-size: 14px;">**Full positional encoding matrix:**</span>

$$
PE = \begin{pmatrix} 0.0000 & 1.0000 & 0.0000 & 1.0000 \\ 0.8415 & 0.5403 & 0.0100 & 0.9999 \\ 0.9093 & -0.4161 & 0.0200 & 0.9998 \end{pmatrix}
$$

<span style="font-size: 14px;">**Observations:**</span>

* <span style="font-size: 14px;">**Columns 0-1 (high frequency, $i=0$):** Values change rapidly. From pos 0 to 2, $\sin$ goes 0.0 to 0.91, $\cos$ goes 1.0 to -0.42. Fine-grained position signal.</span>
* <span style="font-size: 14px;">**Columns 2-3 (low frequency, $i=1$):** Values change slowly. $\sin$ goes 0.0 to 0.02, $\cos$ stays near 1.0. Coarse position signal.</span>
* <span style="font-size: 14px;">**All values in [-1, 1]:** Guaranteed by sine/cosine.</span>
* <span style="font-size: 14px;">**Position 0 has a distinctive pattern:** All sin values are 0, all cos values are 1, giving $(0, 1, 0, 1)$.</span>
* <span style="font-size: 14px;">**Each row is unique:** No two positions share the same encoding vector.</span>

---

## <span style="font-size: 16px;">Sinusoidal vs Learned vs RoPE</span>

<span style="font-size: 14px;">Three major approaches to positional encoding have emerged in the Transformer literature:</span>

<span style="font-size: 14px;">**Sinusoidal (Vaswani et al., 2017).** Fixed, deterministic, added to token embeddings. Zero learnable parameters. Can extrapolate to positions beyond training length. However, the fixed nature prevents task-specific adaptation. Encodes absolute position: each position always gets the same vector regardless of context.</span>

<span style="font-size: 14px;">**Learned positional embeddings (BERT, GPT-2).** A lookup table of shape $(L_{\max}, d_{\text{model}})$ with learnable vectors. More flexible since the model can learn task-specific positional patterns. The downside is a hard maximum sequence length: positions beyond $L_{\max}$ have no embedding, preventing extrapolation. Adds $L_{\max} \times d_{\text{model}}$ parameters (393,216 for BERT with $L_{\max} = 512$, $d_{\text{model}} = 768$).</span>

<span style="font-size: 14px;">**RoPE (Su et al., 2021).** Does not add a vector to embeddings. Instead, rotates query and key vectors by position-dependent angles before the dot product, so $q_i^T k_j$ naturally encodes relative position $(i - j)$. No learnable parameters. Can theoretically extrapolate, though practical long-context use requires scaling techniques. Directly encodes relative position, which is what attention mechanisms actually need.</span>

<span style="font-size: 14px;">The progression reflects deepening understanding: sinusoidal showed periodic functions can encode position; learned embeddings showed flexibility helps; RoPE showed encoding relative position directly in attention is more principled than adding absolute position to embeddings.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong dimension index.** The formula uses $2i$ in the exponent, not $i$. For $d_{\text{model}} = 512$, the pair index $i$ ranges 0 to 255 and the exponent is $2i/512$, not $i/512$. Using $i/512$ halves all frequencies.</span>
* <span style="font-size: 14px;">**Wrong base constant.** The base is 10000, not 1000 or 100000. This value spans wavelengths from $2\pi$ (about 6.28 positions) to $2\pi \cdot 10000$ (about 62,832 positions). A different base changes the frequency range.</span>
* <span style="font-size: 14px;">**Forgetting the even/odd split.** Applying sine to all dimensions or cosine to all dimensions destroys the unique identification property and breaks the linear combination for relative positions. The correct pattern is sine for even (0, 2, 4, ...) and cosine for odd (1, 3, 5, ...).</span>
* <span style="font-size: 14px;">**Wrong division in the exponent.** The exponent is $2i / d_{\text{model}}$, producing values from 0 to roughly 1. Common errors: $2i \cdot d_{\text{model}}$ (multiplication instead of division) or $d_{\text{model}} / 2i$ (inverted fraction), both producing wildly incorrect frequencies.</span>
* <span style="font-size: 14px;">**Values outside [-1, 1].** Every entry is a sine or cosine output, so results must be in $[-1, 1]$. Values outside this range indicate an implementation error.</span>
* <span style="font-size: 14px;">**Concatenating instead of adding.** The positional encoding is added to the token embedding, not concatenated. If your output dimension is $2 \cdot d_{\text{model}}$, you are concatenating.</span>
* <span style="font-size: 14px;">**Off-by-one in position indexing.** Positions are 0-indexed. The first token is at position 0, not 1. Starting at 1 shifts all encodings and loses the distinctive anchor at position 0 where $\sin(0) = 0$ and $\cos(0) = 1$.</span>

---