# Laporan Proyek Machine Learning – Afif Hamzah

## Project Overview

Pada bagian ini, saya mengangkat permasalahan _“cold start”_ dan _“discoverability”_ pada platform streaming film: pengguna sering kesulitan menemukan film baru sesuai preferensi genre dan selera pribadi mereka. Sistem rekomendasi dapat membantu:

- Meningkatkan engagement dan retensi pengguna dengan menyajikan pilihan film yang lebih relevan.  
- Mengurangi waktu pencarian film sehingga pengalaman menonton menjadi lebih lancar.  

**Referensi**  
1. F. Ricci, L. Rokach, dan B. Shapira, _Recommender Systems Handbook_, Springer, 2015.  
2. L. Lops, M. de Gemmis, dan G. Semeraro, “Content-based Recommender Systems,” dalam _The Adaptive Web_, Springer, 2011, hlm. 73–105.  

---

## Business Understanding

### Problem Statements
- **PS1:** Pengguna baru tidak memiliki histori rating sehingga sulit mendapatkan rekomendasi yang relevan (_cold start user_).  
- **PS2:** Pengguna dewasa sulit menemukan film-film dengan genre tertentu karena katalog yang besar.  

### Goals
- **G1:** Menyediakan rekomendasi Top-N film yang relevan bagi setiap pengguna, termasuk pengguna baru.  
- **G2:** Meningkatkan personalization dengan memanfaatkan metadata genre dan pola rating pengguna.

### Solution Approach
1. **Content-Based Filtering (CBF)**  
   - Menghitung kesamaan (cosine similarity) antar film berdasarkan _bag-of-genres_ menggunakan TF-IDF.  
2. **Collaborative Filtering (CF)**  
   - Memanfaatkan matrix factorization (SVD) pada user–item rating matrix untuk prediksi rating dan ranking film.

---

## Data Understanding

**Sumber Data**  
- MovieLens 100K via KaggleHub:  
  `https://www.kaggle.com/prajitdatta/movielens-100k-dataset`

| File       | Deskripsi                                    |
|------------|----------------------------------------------|
| `u.data`   | `user_id`, `movie_id`, `rating` (1–5), `timestamp` |
| `u.item`   | `movie_id`, `movie_title`, `release_date`, `IMDb_URL`, 19 kolom genre binary |
| `u.genre`  | Mapping kode genre ke nama                   |
| `u.info`   | Ringkasan jumlah users, items, dan ratings   |

### Variabel/Fitur  
- **user_id** (int) – ID unik pengguna  
- **movie_id** (int) – ID unik film  
- **rating** (int 1–5) – Nilai rating film oleh user  
- **movie_title** (str) – Judul film  
- **release_date**, **IMDb_URL** (str) – Metadata tambahan  
- **genre_0…genre_18** (0/1) – Binary indicator untuk setiap genre  

### Exploratory Data Analysis (Ringkas)
- **Jumlah users**: 943  
- **Jumlah film**: 1 682  
- **Jumlah rating**: 100 000  
- **Sparsity** ≈ 93.7 %  
- **Distribusi rating**: mayoritas 3–5  
- **Top-10 film paling sering dirating**:
  1. *Star Wars* (1977)  
  2. *Contact* (1997)  
  3. *Fargo* (1996)  
  …  
- **Top-10 film terbaik (≥ 50 ratings)** berdasarkan rata-rata:  
  1. *Close Shave, A* (1995)  
  2. *Godfather, The* (1972)  
  …  
- **Genre terbanyak**: Drama, Comedy, Action  
- **Rata-rata rating per genre**: Film-Noir (3.8), Documentary (3.7), …  
- **Tren rilis film** per dekade: puncak di 1990-an  

---

## Data Preparation

1. **Drop kolom** `video_release_date` (100 % null).  
2. **Lowercase** judul film:
   ```python
   movies['movie_title'] = movies['movie_title'].str.lower()
3. **Gabung genre** dari 19 kolom binary menjadi satu string:
```python
genre_cols = [f'genre_{i}' for i in range(19)]
movies['genre'] = movies[genre_cols]\
    .apply(lambda r: ",".join([col.split("_")[1]
                               for col in genre_cols if r[col]==1]), axis=1)
```
4. **Split train/test** untuk CF (80 % train, 20 % test, random_state=42).
  Setiap langkah dijelaskan di notebook lengkap dengan alasan dan kode snippet.

# Modeling
1. Content-Based Filtering (CBF)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(token_pattern=r'[^,]+')
tfidf_matrix = tfidf.fit_transform(movies['genre'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['movie_title']).drop_duplicates()

def recommend_cbf(title, top_n=10):
    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]),
                        key=lambda x: x[1], reverse=True)[1:top_n+1]
    return movies[['movie_title','genre']].iloc[[i for i,_ in sim_scores]]

# Contoh:
recommend_cbf('toy story (1995)')
```

| movie\_title              | genre       |
| ------------------------- | ----------- |
| empire strikes back, the… | adventure,… |
| return of the jedi (1983) | adventure,… |
| independence day (id4)    | action,…    |
| mars attacks! (1996)      | comedy,…    |
| starship troopers (1997)  | action,…    |


2. Collaborative Filtering (CF) – SVD
```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id','movie_id','rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
rmse_score = rmse(predictions)
```
* RMSE (SVD): 0.9364
``` python
def recommend_cf(user_id, top_n=10):
    all_ids = ratings['movie_id'].unique()
    seen = ratings[ratings['user_id']==user_id]['movie_id']
    unseen = [m for m in all_ids if m not in seen]
    preds = [model.predict(user_id, m) for m in unseen]
    top = sorted(preds, key=lambda x: x.est, reverse=True)[:top_n]
    return movies[movies['movie_id'].isin([p.iid for p in top])][['movie_title','genre']]

# Contoh:
recommend_cf(user_id=10)
```

| movie\_title                   | genre     |
| ------------------------------ | --------- |
| mr. smith goes to washington … | drama     |
| henry v (1989)                 | drama,…   |
| titanic (1997)                 | drama,…   |
| schindler’s list (1993)        | drama,…   |
| close shave, a (1995)          | animation |

# Evaluation
Metrik yang Digunakan
RMSE (Root Mean Square Error)

![RMSE](https://raw.githubusercontent.com/AfifHamzah17/submission-BMLT-RS/main/RMSE.png)

Mengukur akurasi prediksi rating untuk CF.
| Pendekatan          | Metrik | Nilai  |
| ------------------- | ------ | ------ |
| Content-Based       | —      | —      |
| Collaborative (SVD) | RMSE   | 0.9364 |

Interpretasi:

* CF (SVD) berhasil memprediksi dengan kesalahan rata-rata < 1 poin rating.

* CBF tidak diukur RMSE karena berfokus pada ranking similarity, bukan prediksi nilai.
