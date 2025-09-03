import numpy as np

def relevancy_scores(lang: np.ndarray,
                              query: np.ndarray,
                              canon: np.ndarray) -> np.ndarray:
    """
    Relevancy score w.r.t. neutral corpus as in LERF paper.

    Args:
        lang: normalized rendered features (n_lang, dim)
        query: normalized query features (n_query, dim)
        canon: normalized canonical / neutral features (n_canon, dim)

    Returns:
        (n_query, n_lang) array in [0, 1]
    """

    num_queries = query.shape[0]
    num_lang_feats = lang.shape[0]

    sim_lang_canon = lang @ canon.T # (n_lang, n_canon)

    scores = np.empty((num_queries, num_lang_feats), dtype=np.float32)
    for q in range(num_queries):
        sim_lang_query = lang @ query[q] # (n_lang,)
        exp_sim_query = np.exp(sim_lang_query)[:, None]   # (n_lang, 1)
        exp_sim_canon = np.exp(sim_lang_canon)           # (n_lang, n_canon)
        p_query = exp_sim_query / (exp_sim_query + exp_sim_canon)  # (n_lang, n_canon)
        scores[q] = p_query.min(axis=1).astype(np.float32)

    return scores