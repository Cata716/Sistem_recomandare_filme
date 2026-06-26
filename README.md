# Sistem de Recomandare Semantică de Filme

Sistem de recomandare bazat pe embeddings semantice fine-tuned, capabil să găsească filme relevante pornind de la o descriere în limbaj natural, fără a depinde de istoricul de interacțiuni al utilizatorilor.


## Conținut repository

Notebookurile urmăresc evoluția sistemului de la un baseline simplu până la modelul final, fiecare variantă adăugând sau modificând ceva față de precedenta.

**`notebook-compare-summaries.ipynb`** a fost punctul de plecare: am comparat șase tehnici de sumarizare (BART, DistilBART, PEGASUS-XSum, PEGASUS-CNN/DM, T5 și o metodă bazată pe extracția primelor două propoziții din descriere) pentru a vedea care produce rezumate mai utile pentru reprezentarea semantică a unui film.

**`bart_summarization+v1.ipynb`** este varianta baseline. Encodarea se face cu modelul all-MiniLM-L6-v2, fără fine-tuning, iar BART este folosit pentru a sumariza descrierile și recenziile. Scorul final combina similaritatea semantică cu ponderi bazate pe popularitate și scorul recenziilor.

**`notebook-varianta2.ipynb`** introduce primul fine-tuning, pe modelul all-mpnet-base-v2, folosind MultipleNegativesRankingLoss.

**`notebook-varianta3.ipynb`** extinde documentul semantic cu rezumatele recenziilor criticilor, pe lângă descrierea filmului.

**`notebook-varianta4.ipynb`** trece la BGE (bge-base-en-v1.5) și adaugă Hard Negative Mining cross-gen, în care negativele sunt filme din genuri diferite față de cel țintă.

**`notebook-varianta5.ipynb`** adaugă cuvintele cheie în documentul semantic și explorează Hard Negative Mining same-genre, în care negativele sunt selectate din filme din același gen cu ajutorul unui cross-encoder.

**`notebook-varianta6.ipynb`** este modelul final. Documentul semantic include rezumatul BART, opinia criticilor, genurile, cuvintele cheie și distribuția. Antrenarea combină negative din același gen și din genuri diferite (Mixed Hard Negative Mining). Rezultatul final este Hit@10 = 44.3% pe un pool de 40.197 de filme.

**`notebook-varianta6-cu-validare.ipynb`** reia antrenarea modelului final cu o împărțire formală 80/10/10 și early stopping, pentru a documenta mai riguros performanța pe seturi separate de validare și test.

**`notebook-varianta7.ipynb`** testează o alternativă în care în loc de rezumatul BART se folosesc primele două propoziții ale descrierii, extrase automat fără sumarizare.

**`app.py`** este aplicația Streamlit. Scrii o descriere în engleză, ea încarcă modelul și embeddings-urile pre-calculate și îți returnează filmele cele mai similare.


## Cum rulezi aplicația

Instalează dependențele:

```bash
pip install streamlit numpy pandas scikit-learn sentence-transformers
```
Aplicația are nevoie de trei fișiere care nu sunt incluse în repository. Descarcă-le de aici: https://drive.google.com/drive/folders/1Ra2K3VHVEWVdIt8jlY4lTyC09ioDpAmx

- `sbert_v6b.zip` - modelul fine-tuned (trebuie dezarhivat după descărcare)
- `movies_final_v6.csv` - datele filmelor
- `doc_embeddings_v6.npy` - embeddings pre-calculate

Pune-le în același director cu `app.py`.

Pornește aplicația:

```bash
streamlit run app.py
```

Aplicația rulează local la `http://localhost:8501`.


