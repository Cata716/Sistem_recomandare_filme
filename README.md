# Sistem de Recomandare Semantică de Filme

Sistem de recomandare bazat pe embeddings semantice fine-tuned, capabil să găsească filme relevante pornind de la o descriere în limbaj natural, fără a depinde de istoricul de interacțiuni al utilizatorilor.

---

## Conținut repository

**`notebook-compare-summaries.ipynb`** compară calitatea rezumatelor produse de patru modele diferite: BART, DistilBART, PEGASUS și T5. Scopul este alegerea celui mai potrivit model de sumarizare pentru construirea documentului semantic.

**`bart_summarization+v1.ipynb`** folosește modelul BART pentru a genera automat rezumate ale descrierilor și recenziilor filmelor. Constituie varianta baseline (V1) a sistemului: nu include fine-tuning, iar scorul final combină similaritatea semantică cu ponderi calculate pe baza popularității și a scorului de recenzii.

**`notebook-varianta2.ipynb`** aplică fine-tuning pe modelul all-mpnet-base-v2 folosind MultipleNegativesRankingLoss. Este primul experiment de antrenare a unui model de embeddings pe datele din proiect.

**`notebook-varianta3.ipynb`** extinde documentul semantic prin adăugarea rezumatelor de recenzii ale criticilor, pe lângă descrierea filmului. Modelul de bază rămâne all-mpnet-base-v2.

**`notebook-varianta4.ipynb`** face trecerea la modelul BGE (bge-base-en-v1.5) și introduce Hard Negative Mining cross-gen, în care negativele sunt filme din genuri diferite față de interogare.

**`notebook-varianta5.ipynb`** rafinează strategia de negative mining prin selecția negativelor din același gen, folosind un cross-encoder pentru a identifica filmele cel mai greu de distins de filmul țintă.

**`notebook-varianta6.ipynb`** reprezintă modelul final. Documentul semantic combină rezumatul BART, opinia criticilor, genurile, cuvintele cheie și distribuția filmului. Antrenarea folosește Mixed Hard Negative Mining, cu negative atât din același gen, cât și din genuri diferite. Atinge Hit@10 = 44.3% pe un pool de 40.197 de filme.

**`notebook-varianta6-cu-validare.ipynb`** reia antrenarea modelului final cu o împărțire formală 80/10/10 și early stopping, pentru a documenta performanța pe seturi separate de validare și test.

**`notebook-varianta7.ipynb`** testează o variantă alternativă în care în locul rezumatului BART se folosește overview-ul brut al filmului, trunchiat la lungimea maximă a modelului.

**`app.py`** este aplicația Streamlit care permite căutarea filmelor prin descriere în limbaj natural. Încarcă modelul fine-tuned și embeddings-urile pre-calculate, apoi returnează cele mai similare filme pe baza similarității cosinus.

---

## Rulare aplicație

### 1. Instalare dependențe

```bash
pip install -r requirements.txt
```

```
streamlit>=1.30.0
numpy>=1.24.0
pandas>=2.0.0
sentence-transformers>=2.7.0
scikit-learn>=1.3.0
```

### 2. Descărcare fișiere model

Din output-ul notebook-ului `notebook-varianta6` descarcă:
- `sbert_v6b/` — directorul cu modelul fine-tuned
- `movies_final_v6.csv` — datele filmelor
- `doc_embeddings_v6.npy` — embeddings pre-calculate

Plasează-le în același director cu `app.py`.

### 3. Pornire

```bash
streamlit run app.py
```

Aplicația rulează local la `http://localhost:8501`.

