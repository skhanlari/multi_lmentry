# Used Sampling Techniques

1. Random_Sampling --> Multiple seeds per ratios
     - 10 seeds for r ≤ 10%
     - 5 seeds for r > 10%

2. ClusteringBased_Sampling --> Deduplication bsed on cosin similarity t= 0.90

    - SentenceTransformer embeddings

    - Cosine similarity threshold (0.90)

    - Cluster representatives

    - Random sampling from deduplicated pool

3. Model-disagreement difficulty based on different modes. (p = mean correctness across models)
     - easy p = 1
        - pick examples with highest p_i (closest to 1)

     - hard p = 0
        - pick examples with lowest p_i (closest to 0)

     - mid p = 0.5
        - pick examples closest to 0.5

     - stratified (hard, mid, easy bins): Select proportional amounts from each bin.
        - hard: (p < 0.33)
        - Mid: (0.33–0.66)
        - Easy: (p > 0.66)
        - pick a mix across bins: easy + mid + hard, so the subset matches the full difficulty distribution (or is balanced).


# Not_Used Sampling Techniques

1. Clustering_based Samplinng
    - Topic modeling (LDA, NMF)
    - Spectral clustering
    - K-means
    - Embedding-based clustering (MTEB, BERT, SFR)

2. Quality-based Sampling
    - Spelling error minimization
    - Optimal word length
    - Lexical diversity
    - Flesch readability

3. Difficulty-based Sampling
    - Gunning Fog
    - SMOG
    - Flesch
    - Dale-Chall

