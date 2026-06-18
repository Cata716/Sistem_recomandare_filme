import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BGE_PREFIX = 'Represent this sentence: '
MODEL_PATH = 'sbert_v6b'
DATA_PATH  = 'movies_final_v6.csv'
EMB_PATH   = 'doc_embeddings_v6.npy'

st.set_page_config(page_title='Sistem de recomandare de filme', layout='wide')

@st.cache_resource(show_spinner='Se încarcă modelul...')
def load_model():
    return SentenceTransformer(MODEL_PATH)

@st.cache_resource(show_spinner='Se încarcă datele...')
def load_data():
    df  = pd.read_csv(DATA_PATH)
    emb = np.load(EMB_PATH)
    return df, emb

model    = load_model()
df, emb  = load_data()

st.title(' Sistem de Recomandare Filme')
st.caption('Fine-tuned BGE + Mixed Hard Negative Mining · 40,197 filme · Hit@10 = 44.3%')

st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        'Descrie filmul pe care îl cauți în limba engleză:',
        placeholder='e.g. a hero saves the world from aliens',
        label_visibility='visible'
    )
with col2:
    top_k = st.selectbox('Număr rezultate:', [1, 3, 5, 10, 20], index=1)

if query and query.strip():
    with st.spinner('Se caută...'):
        q_emb  = model.encode([BGE_PREFIX + query.strip()], normalize_embeddings=True)
        scores = cosine_similarity(q_emb, emb)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]

    st.subheader(f'Top {top_k} recomandări pentru: *"{query}"*')
    st.divider()

    for rank, idx in enumerate(top_idx, 1):
        row   = df.iloc[idx]
        score = float(scores[idx])
        title = str(row.get('title', 'N/A'))
        year  = str(row.get('release_date', ''))[:4]
        genres = str(row.get('genres', '')).replace("'", '').strip("[]")
        overview = str(row.get('overview', ''))
        review   = str(row.get('review_summary', ''))

        with st.container():
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f'### {rank}. {title} {"(" + year + ")" if year and year != "nan" else ""}')
                if genres and genres not in ('', 'nan'):
                    st.caption(f' {genres}')
            with c2:
                st.metric('Scor', f'{score:.3f}')

            if overview and overview not in ('', 'nan'):
                st.write(overview[:400] + ('...' if len(overview) > 400 else ''))

            if review and review not in ('', 'nan'):
                with st.expander('Opinia criticilor'):
                    st.write(review[:500] + ('...' if len(review) > 500 else ''))

            st.divider()
