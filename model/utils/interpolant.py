def trans_vector_field(t, trans_1, trans_t):
    return (trans_1 - trans_t) / (1 - t)

def plm_embed_vector_field(t, plm_embed_1, plm_embed_t):
    return (plm_embed_1 - plm_embed_t) / (1 - t)
