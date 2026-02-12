Movie Quote Recommender App (Streamlit)
  
   Aplikasi web sederhana berbasis Streamlit untuk merekomendasikan kutipan film berdasarkan kemiripan semantik. Aplikasi ini membandingkan input teks pengguna dengan       kumpulan kutipan film menggunakan embedding NLP dan cosine similarity.

Project ini menggunakan beberapa model embedding:
1. BERT
2. SBERT
3. MPNet
   
Fitur Utama
1. Input teks bebas dari pengguna
2. Rekomendasi kutipan film paling relevan
3. Perbandingan hasil dari beberapa model embedding
4. Embedding sudah disimpan dalam file .npy sehingga tidak dihitung ulang setiap kali aplikasi dijalankan
   
Struktur Folder

├── app.py

├── movie_quotes.csv

├── bert_embeddings.npy

├── sbert_embeddings.npy

├── mpnet_embeddings.npy

└── requirements.txt

Dependensi & Library
Library utama yang digunakan dalam project ini:
1. streamlit – framework web app
2. torch – backend deep learning
3. transformers – model BERT
4. sentence-transformers – SBERT & MPNet
5. numpy – operasi numerik & loading embedding
6. pandas – pengolahan data CSV
7. scikit-learn – cosine similarity

  Semua dependensi sudah didefinisikan di requirements.txt

Langkah Instalasi
1. Clone repository

   git clone https://github.com/vincenttanaka/movie-quotes-recomender.git


3. Buat virtual environment (opsional tapi disarankan)
   
   python -m venv venv

   source venv/bin/activate   # Linux / Mac

   venv\Scripts\activate      # Windows

5. Install dependensi
   
   pip install -r requirements.txt
   
   Pastikan versi Python kamu kompatibel (disarankan Python 3.9+).

Cara Menjalankan Aplikasi

   Setelah instalasi selesai, jalankan perintah berikut:

   streamlit run app.py

   Aplikasi akan otomatis terbuka di browser

<img width="1349" height="914" alt="image" src="https://github.com/user-attachments/assets/6ee11afd-215e-480c-9c0a-2bfffc5d7506" />

