# MachineLearning
<h2>Prediksi Kelulusan Mahasiswa dengan Machine Learning</h2>
<p>
Proyek ini mengimplementasikan analisis data dan model <b>Machine Learning</b> untuk memprediksi 
kelulusan mahasiswa berdasarkan data akademik seperti <b>IPK</b>, <b>jumlah absensi</b>, 
dan <b>waktu belajar</b>. 
Melalui tahapan <i>data cleaning</i>, <i>exploratory data analysis (EDA)</i>, 
<i>feature engineering</i>, <i>model building</i>, <i>tuning</i>, dan <i>evaluasi</i>, 
proyek ini menghasilkan model <b>Random Forest</b> yang mampu memprediksi kelulusan secara akurat 
dan siap digunakan untuk proses <b>inference otomatis</b>.
</p>
<hr>
<h3>Tahapan Proyek</h3>
<ol>
  <li><b>Data Collection & Cleaning</b><br>
    Membaca dataset <code>kelulusan_mahasiswa.csv</code> menggunakan Pandas, 
    menghapus duplikasi, menangani nilai kosong, dan melakukan visualisasi awal 
    menggunakan <i>Seaborn</i> serta <i>Matplotlib</i> untuk mendeteksi outlier.
  </li>

  <li><b>Exploratory Data Analysis (EDA)</b><br>
    Melakukan analisis deskriptif dan visualisasi hubungan antar variabel 
    menggunakan histogram, scatter plot, dan heatmap korelasi 
    untuk memahami pola yang memengaruhi kelulusan mahasiswa.
  </li>

  <li><b>Feature Engineering</b><br>
    Menambahkan fitur baru seperti <code>Rasio_Absensi</code> dan <code>IPK_x_Study</code> 
    untuk meningkatkan performa model. 
    Dataset hasil olahan disimpan sebagai <code>processed_kelulusan.csv</code>.
  </li>

  <li><b>Dataset Splitting</b><br>
    Membagi dataset menjadi <b>train</b>, <b>validation</b>, dan <b>test set</b> 
    dengan rasio 70/15/15 menggunakan <code>train_test_split</code> 
    dengan <code>stratify</code> untuk menjaga distribusi label.
  </li>

  <li><b>Model Development</b><br>
    - <b>Baseline Model:</b> Logistic Regression dengan preprocessing pipeline 
      menggunakan <i>SimpleImputer</i> dan <i>StandardScaler</i>.<br>
    - <b>Model Alternatif:</b> Random Forest Classifier dengan parameter 
      <code>class_weight="balanced"</code> untuk menangani ketidakseimbangan kelas.
  </li>

  <li><b>Model Evaluation & Hyperparameter Tuning</b><br>
    Melakukan validasi silang (<i>StratifiedKFold</i>) dan tuning hyperparameter 
    menggunakan <i>GridSearchCV</i>. 
    Evaluasi dilakukan dengan metrik <b>F1-score (macro)</b>, 
    <b>classification report</b>, dan <b>ROC-AUC</b> curve.
  </li>

  <li><b>Feature Importance</b><br>
    Menampilkan peringkat kepentingan fitur (feature importance) dari model 
    <b>Random Forest</b> untuk interpretasi hasil.
  </li>

  <li><b>Model Deployment Preparation</b><br>
    Menyimpan model terbaik ke file <code>rf_model.pkl</code> menggunakan <i>joblib</i> 
    dan menambahkan contoh prediksi lokal (<i>inference</i>) dengan input fiktif.
  </li>
</ol>
<hr>
<li><b>Model Deployment Preparation</b><br>
    Menyimpan model terbaik ke file <code>rf_model.pkl</code> menggunakan <i>joblib</i> 
    dan menambahkan contoh prediksi lokal (<i>inference</i>) dengan input fiktif.
  </li>
</ol>
<hr>
<h3>Hasil Akhir</h3>

<p>
Model <b>Random Forest</b> memberikan performa terbaik dengan nilai 
<b>F1-score</b> dan <b>ROC-AUC</b> yang tinggi pada data validasi dan data uji. 
Pipeline ini dapat digunakan kembali untuk memprediksi kelulusan mahasiswa baru 
secara otomatis dengan hasil yang cepat dan konsisten.
</p>
