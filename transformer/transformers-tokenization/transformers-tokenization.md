# <span style="font-size: 20px;">Word-Level Tokenization</span>

<span style="font-size: 14px;">Tokenization is the process of converting raw text into a sequence of discrete integer IDs that a neural network can process. It is the very first stage of every NLP pipeline, bridging the gap between human-readable strings and the numerical tensors that models operate on. Word-level tokenization, the simplest strategy, treats each whitespace-delimited word as an atomic token.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A word-level tokenizer performs two complementary operations. **Encoding** takes a raw string, normalizes it by lowercasing, splits it on whitespace, and maps each resulting word to a unique integer ID via a fixed vocabulary lookup. **Decoding** reverses the process, converting a sequence of integer IDs back into a human-readable string by looking up each ID in the reverse mapping and joining the words with spaces.</span>

<span style="font-size: 14px;">The vocabulary is built in advance from a training corpus. Every unique word receives a permanent integer ID. Words encountered at inference time that were never seen during training are mapped to a special unknown token. This is the core tradeoff: simplicity at the cost of a rigid vocabulary that cannot handle novel words. The tokenizer maintains a deterministic bijective mapping:</span>

$$
\texttt{word\_to\_id}: \text{string} \rightarrow \text{integer} \qquad \texttt{id\_to\_word}: \text{integer} \rightarrow \text{string}
$$

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The encoding function for a single word $w$, given vocabulary $V$, is:</span>

$$
\text{encode}(w) = \begin{cases} \texttt{word\_to\_id}[w] & \text{if } w \in V \\ 1 & \text{otherwise (UNK)} \end{cases}
$$

<span style="font-size: 14px;">The four special tokens occupy the first four IDs with fixed assignments:</span>

$$
\texttt{<PAD>} = 0, \quad \texttt{<UNK>} = 1, \quad \texttt{<BOS>} = 2, \quad \texttt{<EOS>} = 3
$$

<span style="font-size: 14px;">Regular vocabulary words receive IDs starting from $4$. If the unique words from the training corpus, sorted alphabetically, are $w_0, w_1, \ldots, w_{n-1}$, then:</span>

$$
\texttt{word\_to\_id}[w_i] = i + 4 \quad \text{for } i = 0, 1, \ldots, n-1
$$

<span style="font-size: 14px;">The total vocabulary size is $|V| = n + 4$, where $n$ is the count of unique training words and $4$ accounts for special tokens.</span>

<span style="font-size: 14px;">The decoding function for a single ID $k$ is simply the inverse lookup:</span>

$$
\text{decode}(k) = \texttt{id\_to\_word}[k]
$$

<span style="font-size: 14px;">For a full sentence, encoding maps the lowercased, whitespace-split word list through $\text{encode}$ element-wise, and decoding maps IDs through $\text{decode}$ then joins with spaces.</span>

---

## <span style="font-size: 16px;">Vocabulary Building</span>

<span style="font-size: 14px;">Building the vocabulary is a deterministic, multi-step procedure that must be followed in exact order to produce consistent ID assignments.</span>

<span style="font-size: 14px;">**Step 1 - Initialize special tokens.** Reserve the first four IDs: `<PAD>` = 0, `<UNK>` = 1, `<BOS>` = 2, `<EOS>` = 3. These are hardcoded constants, never derived from training data.</span>

<span style="font-size: 14px;">**Step 2 - Lowercase all text.** Every training sentence is converted to lowercase so that "The", "the", and "THE" all map to the same token. Without this, the vocabulary contains spurious duplicates differing only in capitalization.</span>

<span style="font-size: 14px;">**Step 3 - Split on whitespace.** Each lowercased sentence is split into words using whitespace as the delimiter. Punctuation attached to a word (like "hello,") remains part of that token.</span>

<span style="font-size: 14px;">**Step 4 - Collect unique words.** Iterate through every word across all training sentences and collect the set of unique words. Duplicates are eliminated.</span>

<span style="font-size: 14px;">**Step 5 - Sort alphabetically.** The unique words are sorted in lexicographic order. Sorting is critical for reproducibility; without it, hash randomization or insertion order could produce different ID assignments across runs.</span>

<span style="font-size: 14px;">**Step 6 - Assign IDs.** Walk through the sorted list and assign IDs starting from $4$. The first word alphabetically gets ID $4$, the second $5$, and so on.</span>

<span style="font-size: 14px;">The result is a forward dictionary `word_to_id` and a reverse dictionary `id_to_word`, both including the special tokens.</span>

---

## <span style="font-size: 16px;">Special Tokens</span>

<span style="font-size: 14px;">Special tokens serve structural roles that regular words cannot. Each exists for a specific engineering reason rooted in how neural networks process sequences.</span>

### <span style="font-size: 14px;">PAD (ID = 0)</span>

<span style="font-size: 14px;">Neural networks process batches of sequences simultaneously, but sequences rarely have the same length. **Padding** fills shorter sequences so all can be stacked into a single tensor. PAD is assigned ID $0$ by convention, which interacts cleanly with masking: a binary mask of non-padding positions is simply a check for non-zero IDs. During attention, padding positions are masked out.</span>

### <span style="font-size: 14px;">UNK (ID = 1)</span>

<span style="font-size: 14px;">The **unknown token** is a fallback for any word not in the vocabulary. Every out-of-vocabulary (OOV) word collapses to the same UNK ID, destroying all information about the original word. This information loss is the central weakness of word-level tokenization.</span>

### <span style="font-size: 14px;">BOS (ID = 2)</span>

<span style="font-size: 14px;">The **beginning-of-sequence** token marks where a sequence starts. In autoregressive models, BOS serves as the initial input token the model conditions on to generate the first real word. Without it, the model cannot distinguish "the first token of a new sequence" from a continuation token.</span>

### <span style="font-size: 14px;">EOS (ID = 3)</span>

<span style="font-size: 14px;">The **end-of-sequence** token marks where a sequence terminates. In autoregressive generation, predicting EOS tells the decoding algorithm to stop. In the original Transformer, EOS on the source side signals input completion, and EOS on the target side signals the end of translation.</span>

---

## <span style="font-size: 16px;">Encoding: Text to IDs</span>

<span style="font-size: 14px;">Encoding converts a raw text string into a list of integer IDs in three steps.</span>

<span style="font-size: 14px;">**Step 1 - Lowercase.** Convert the entire input to lowercase, matching the normalization applied during vocabulary building.</span>

<span style="font-size: 14px;">**Step 2 - Split.** Split the lowercased string on whitespace to produce a list of word strings.</span>

<span style="font-size: 14px;">**Step 3 - Lookup.** For each word, look it up in `word_to_id`. If found, use its ID. If not found, use the UNK ID ($1$).</span>

<span style="font-size: 14px;">The output is a list of integers with the same length as the number of words. BOS and EOS are not automatically added; whether to wrap the sequence with them depends on the downstream model.</span>

---

## <span style="font-size: 16px;">Decoding: IDs to Text</span>

<span style="font-size: 14px;">Decoding converts a list of integer IDs back into a human-readable string.</span>

<span style="font-size: 14px;">**Step 1 - Lookup.** For each ID, look it up in `id_to_word` to retrieve the corresponding string. Special token IDs map to their string representations (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).</span>

<span style="font-size: 14px;">**Step 2 - Join.** Concatenate all retrieved strings with a single space between each pair.</span>

<span style="font-size: 14px;">Decoding is not a perfect inverse of encoding. Any word that was mapped to UNK during encoding decodes as the literal string `<UNK>`, not the original word. The original word is irrecoverably lost.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The original Transformer paper, "Attention Is All You Need" (Vaswani et al., 2017), did not use word-level tokenization. The authors employed **Byte Pair Encoding (BPE)** with a shared source-target vocabulary of approximately 37,000 tokens for English-German translation and 32,000 tokens for English-French.</span>

<span style="font-size: 14px;">The choice was deliberate. Machine translation handles two languages simultaneously, and word-level vocabularies for both combined would be enormous. BPE compresses the vocabulary by representing rare words as sequences of common subword units, eliminating the OOV problem entirely.</span>

<span style="font-size: 14px;">Notably, Vaswani et al. used a shared vocabulary: a single BPE vocabulary learned from concatenated bilingual data. This allowed encoder and decoder to share the same embedding matrix, reducing parameters and exploiting cross-lingual subword overlap.</span>

<span style="font-size: 14px;">Word-level tokenization remains foundational as the conceptual baseline from which all subword methods depart. Every subword tokenizer still produces integer ID sequences, manages special tokens, and maintains encode/decode symmetry.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider two training sentences:</span>

* <span style="font-size: 14px;">**Sentence 1:** "The cat sat on the mat"</span>
* <span style="font-size: 14px;">**Sentence 2:** "The dog sat on the log"</span>

### <span style="font-size: 14px;">Building the Vocabulary</span>

<span style="font-size: 14px;">**Lowercase:** Both sentences become "the cat sat on the mat" and "the dog sat on the log".</span>

<span style="font-size: 14px;">**Collect unique words:** {cat, dog, log, mat, on, sat, the}. "the" appears four times but counts once.</span>

<span style="font-size: 14px;">**Sort alphabetically:** [cat, dog, log, mat, on, sat, the].</span>

<span style="font-size: 14px;">**Assign IDs starting from 4:**</span>

* <span style="font-size: 14px;">**`<PAD>`** = 0</span>
* <span style="font-size: 14px;">**`<UNK>`** = 1</span>
* <span style="font-size: 14px;">**`<BOS>`** = 2</span>
* <span style="font-size: 14px;">**`<EOS>`** = 3</span>
* <span style="font-size: 14px;">**cat** = 4</span>
* <span style="font-size: 14px;">**dog** = 5</span>
* <span style="font-size: 14px;">**log** = 6</span>
* <span style="font-size: 14px;">**mat** = 7</span>
* <span style="font-size: 14px;">**on** = 8</span>
* <span style="font-size: 14px;">**sat** = 9</span>
* <span style="font-size: 14px;">**the** = 10</span>

<span style="font-size: 14px;">Total vocabulary size: $7 + 4 = 11$.</span>

### <span style="font-size: 14px;">Encoding a Known Sentence</span>

<span style="font-size: 14px;">**Input:** "The cat sat on the mat"</span>

<span style="font-size: 14px;">**Lowercase:** "the cat sat on the mat"</span>

<span style="font-size: 14px;">**Split:** ["the", "cat", "sat", "on", "the", "mat"]</span>

<span style="font-size: 14px;">**Lookup each word:** [10, 4, 9, 8, 10, 7]</span>

<span style="font-size: 14px;">Every word exists in the vocabulary, so no UNK tokens appear.</span>

### <span style="font-size: 14px;">Encoding a Sentence With Unknown Words</span>

<span style="font-size: 14px;">**Input:** "The bird sat on the log"</span>

<span style="font-size: 14px;">**Lowercase:** "the bird sat on the log"</span>

<span style="font-size: 14px;">**Split:** ["the", "bird", "sat", "on", "the", "log"]</span>

<span style="font-size: 14px;">**Lookup:** "the" = 10, "bird" not in vocabulary = 1 (UNK), "sat" = 9, "on" = 8, "the" = 10, "log" = 6.</span>

<span style="font-size: 14px;">**Result:** [10, 1, 9, 8, 10, 6]</span>

<span style="font-size: 14px;">"bird" was never seen during training, so it collapses to UNK.</span>

### <span style="font-size: 14px;">Decoding</span>

<span style="font-size: 14px;">**Input IDs:** [10, 1, 9, 8, 10, 6]</span>

<span style="font-size: 14px;">**Lookup:** 10 = "the", 1 = "`<UNK>`", 9 = "sat", 8 = "on", 10 = "the", 6 = "log".</span>

<span style="font-size: 14px;">**Join with spaces:** "the `<UNK>` sat on the log"</span>

<span style="font-size: 14px;">"bird" has been replaced by `<UNK>`, demonstrating the irreversible information loss.</span>

---

## <span style="font-size: 16px;">Word-Level vs Subword Tokenization</span>

<span style="font-size: 14px;">Word-level and subword tokenization make fundamentally different tradeoffs along the granularity spectrum.</span>

### <span style="font-size: 14px;">Word-Level Tokenization</span>

* <span style="font-size: 14px;">**Vocabulary size:** Grows with the corpus. English has over 170,000 words in active use. A 100,000+ token vocabulary means the embedding matrix alone consumes hundreds of megabytes.</span>
* <span style="font-size: 14px;">**OOV problem:** Any word not in the vocabulary maps to UNK. Catastrophic for morphologically rich languages (Turkish, Finnish), proper nouns, and typos.</span>
* <span style="font-size: 14px;">**Token semantics:** Each token is a complete word, making the vocabulary human-interpretable.</span>
* <span style="font-size: 14px;">**Sequence length:** Produces the shortest sequences (one token per word), minimizing the $O(n^2)$ attention cost.</span>

### <span style="font-size: 14px;">Subword Tokenization (BPE, WordPiece, Unigram)</span>

* <span style="font-size: 14px;">**Vocabulary size:** Typically 30,000 to 50,000 tokens regardless of corpus size. GPT-2 uses 50,257 BPE tokens; BERT uses 30,522 WordPiece tokens.</span>
* <span style="font-size: 14px;">**No OOV problem:** Any word decomposes into known subword units. "transformerification" becomes ["transform", "##eri", "##fication"] rather than UNK.</span>
* <span style="font-size: 14px;">**Morphological sharing:** "run", "running", "runner" share the subword "run", enabling generalization across variants.</span>
* <span style="font-size: 14px;">**Sequence length:** Longer sequences than word-level because rare words split into multiple pieces, increasing the $n$ in $O(n^2)$ attention.</span>

<span style="font-size: 14px;">The Transformer paper chose BPE because translation demands robustness to rare words across two languages. A word-level tokenizer for English-German would need 200,000+ tokens, still suffer OOV on proper nouns, and waste capacity on rare words. BPE with 37,000 shared tokens solved all three problems.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to lowercase before vocabulary lookup.** If the vocabulary was built from lowercased text but encoding skips lowercasing, "The" will not match "the" and maps to UNK. This is a silent failure: the model runs but receives degraded input with inflated UNK rates.</span>

* <span style="font-size: 14px;">**Wrong sort order producing inconsistent IDs.** If unique words are not sorted before assigning IDs, the mapping becomes non-deterministic. Python's `set` has no guaranteed iteration order. Two runs on the same data could produce different vocabularies.</span>

* <span style="font-size: 14px;">**Not handling unknown words during encoding.** If the encoding function does a direct dictionary access without a fallback, it crashes with a KeyError on the first OOV word. The correct implementation uses `.get()` with the UNK ID as default.</span>

* <span style="font-size: 14px;">**Off-by-one in IDs after special tokens.** The first regular word must receive ID $4$, because IDs $0$ through $3$ are reserved. A common bug is starting enumeration at $0$, overwriting special token IDs. This causes collisions where a regular word shares an ID with PAD or UNK, silently corrupting all encoding.</span>

* <span style="font-size: 14px;">**Case sensitivity in the reverse mapping.** The `id_to_word` dictionary should map IDs to lowercased words. If built from original non-lowercased text, the round-trip property breaks: decode(encode(lowercase(text))) will not equal lowercase(text).</span>

* <span style="font-size: 14px;">**Treating special token strings as regular words.** If the training corpus contains literal strings like "`<PAD>`" or "`<UNK>`", they should not be added to the regular vocabulary a second time, or duplicate entries at different IDs make encoding ambiguous.</span>

* <span style="font-size: 14px;">**Assuming punctuation is handled.** Basic word-level tokenization splits only on whitespace. "hello, world!" produces ["hello,", "world!"], not ["hello", ",", "world", "!"]. Punctuation-attached words become distinct vocabulary entries, causing unexpected bloat.</span>

---