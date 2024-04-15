from datasketch import MinHash, MinHashLSH
from nltk import ngrams


def shingle_document(doc, n=3):
    """
    Convert document to set of shingles (n-grams).
    """
    res = set(ngrams(doc, n))
    return res


def create_minhash(shingles, num_perm=128):
    """
    Create a MinHash from a set of shingles.
    """
    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update("".join(shingle).encode("utf-8"))
    return m


def remove_near_duplicates(docs, threshold=0.7, num_perm=128):
    """
    Remove near duplicate documents.
    """
    # Create MinHash for all docs
    minhashes = []
    for doc in docs:
        shingles = shingle_document(doc)
        minhash = create_minhash(shingles, num_perm=num_perm)
        minhashes.append(minhash)

    # LSH to find near duplicates
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, minhash in enumerate(minhashes):
        lsh.insert(i, minhash)

    # Collect the IDs of unique docs
    unique_doc_ids = set()
    seen_doc_ids = set()
    for i, minhash in enumerate(minhashes):
        if i not in seen_doc_ids:
            # Query LSH for the current minhash
            result = lsh.query(minhash)
            # Convert results back to integers
            result_ids = set(map(int, result))
            # Update seen_doc_ids
            seen_doc_ids.update(result_ids)
            # Add the first doc in the group as the unique representative
            unique_doc_ids.add(min(result_ids))
    return sorted(list(unique_doc_ids))
