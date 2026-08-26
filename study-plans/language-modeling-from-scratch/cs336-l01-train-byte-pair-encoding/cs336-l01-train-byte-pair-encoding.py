def train_bpe(corpus, vocab_size):
  corpus = [list(s.encode('utf-8')) for s in corpus]

  vocabs = 256
  deliverable = {'vocab': [], 'merges': []}
  merge_list = {}

  while vocabs < vocab_size:
    merge = {}
    if len(corpus) == 0:
      break

    for seq in corpus:
      if len(seq) < 2:
        continue
      for pairs in zip(seq, seq[1:]):
        merge[pairs] = merge.get(pairs, 0) + 1

    if len(merge) == 0:
      break

    max_val = max(merge.values())
    max_freq_pairs = [k for k, v in merge.items() if v == max_val]

    def find_all_simple_token(token):
      if token < 256:
        return [token]
      tokens1 = find_all_simple_token(merge_list[token][0])
      tokens2 = find_all_simple_token(merge_list[token][1])
      return tokens1 + tokens2

    merged_tokens = max(
        max_freq_pairs,
        key=lambda p: (
            find_all_simple_token(p[0]),
            find_all_simple_token(p[1]),
        ),
    )
    merge.clear()

    updated_corpus = []
    for seq in corpus:
      updated_list = []
      i = 0
      new_val = vocabs
      while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == merged_tokens:
          updated_list.append(new_val)
          i += 2
        else:
          updated_list.append(seq[i])
          i += 1
      updated_corpus.append(updated_list)
    corpus = updated_corpus

    deliverable['merges'].append([merged_tokens[0], merged_tokens[1], new_val])
    merge_list.setdefault(new_val, []).extend(
        [merged_tokens[0], merged_tokens[1]]
    )

    all_token1 = find_all_simple_token(
        merged_tokens[0]
    ) + find_all_simple_token(merged_tokens[1])

    deliverable['vocab'].append([new_val, all_token1])
    vocabs += 1

  return deliverable