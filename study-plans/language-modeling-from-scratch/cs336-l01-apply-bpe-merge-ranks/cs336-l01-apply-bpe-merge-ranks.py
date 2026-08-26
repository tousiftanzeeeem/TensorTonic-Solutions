def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns: list[int] containing token IDs after applying the ordered merge rules
    """
    text = list(text.encode('utf-8'))
    for i in merges:
        left = i[0]
        right = i[1]
        new = i[2]
        updated_list = []
        iin=0
        while iin < len(text):
            if iin < len(text) - 1 and (text[iin], text[iin+1]) == (left,right):
                updated_list.append(new); iin+=2
            else:
                updated_list.append(text[iin])
                iin += 1
        text = updated_list
    return text

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns: the Unicode string reconstructed from token IDs and vocabulary bytes
    """
    vocab = sorted(vocab.items(), reverse=True)
    for i in vocab:
        updated_list = []
        for t in ids:
            if(t == i[0]):
                updated_list.extend(i[1])
            else:
                updated_list.append(t)
        ids = updated_list
    
    text = bytes(ids).decode('utf-8')
    return text
