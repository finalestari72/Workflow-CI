# Workflow-CI — Water Potability

Repositori ini berisi **Kriteria 3 – Workflow CI** untuk Submission Machine Learning MSML Dicoding.
Model yang dilatih adalah **Random Forest Classifier** untuk prediksi kelayakan air minum.

---

## 📁 Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI (Advanced)
├── MLProject/
│   ├── modelling.py                        # Script training (MLflow entry point)
│   ├── conda.yaml                          # Environment dependencies (Python 3.12.7)
│   ├── MLProject                           # Konfigurasi MLflow Project
│   ├── water_potability_preprocessing.csv  # Dataset siap latih
│   └── DockerHub.txt                       # Tautan Docker Hub
└── README.md
```

---

## 🔐 GitHub Secrets yang Dibutuhkan

Tambahkan secrets di **GitHub → Settings → Secrets and variables → Actions**:

| Secret | Keterangan |
|---|---|
| `DAGSHUB_USERNAME` | Username DagsHub (misal: `finalestari2712`) |
| `DAGSHUB_TOKEN`    | Token dari dagshub.com → User Settings → Tokens |
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN`    | Access token dari hub.docker.com → Account Settings → Security |

---

## 🚀 Trigger CI

### Otomatis
Workflow berjalan setiap ada **push ke branch `main`/`master`** yang menyentuh file di `MLProject/`.

### Manual (workflow_dispatch)
Buka **Actions → CI — Water Potability MLflow Training → Run workflow**, lalu atur parameter:

| Parameter | Default | Keterangan |
|---|---|---|
| `n_estimators` | 200 | Jumlah pohon Random Forest |
| `max_depth` | 10 | Kedalaman maksimum (0 = None) |
| `min_samples_split` | 2 | Min sampel untuk split node |
| `min_samples_leaf` | 1 | Min sampel di leaf node |
| `max_features` | sqrt | Fitur per split |

---

## 📋 Alur CI — Advanced (4 pts)

```
Push ke main / Manual trigger
           │
           ▼
┌──────────────────────────────────────┐
│     Job: train-and-deploy            │
│                                      │
│  1. Run actions/checkout@v3          │
│  2. Set up Python 3.12.7             │
│  3. Check Env                        │  ← verifikasi environment
│  4. Install dependencies             │  ← mlflow, sklearn, pandas, dll
│  5. Run mlflow project               │  ← mlflow run MLProject
│  6. Get latest MLflow run_id         │  ← simpan ke GITHUB_ENV
│  7. Install Python dependencies      │  ← requests, dll
│  8. Upload to GitHub                 │  ← upload-artifact mlruns/
│  9. Commit mlruns ke repository      │  ← artefak tersimpan di repo
│ 10. Build Docker Model               │  ← mlflow models build-docker
│ 11. Log in to Docker Hub             │  ← docker/login-action
│ 12. Tag Docker Image                 │  ← tag :latest, :run-N, :sha
│ 13. Push Docker Image                │  ← push ke Docker Hub
└──────────────────────────────────────┘
```

---

## 📊 Penilaian

| Tingkat | Poin | Yang Dipenuhi |
|---|---|---|
| Basic   | 2 | Folder MLProject + workflow CI berjalan |
| Skilled | 3 | + Artefak disimpan ke GitHub (upload-artifact + commit) |
| **Advance** | **4** | **+ Docker image di-build & push ke Docker Hub** |

---

## 🐳 Docker Hub

Tautan: **https://hub.docker.com/r/finalestari72/workflow-ci**

```bash
# Pull image
docker pull finalestari72/water-potability-model:latest

# Jalankan server prediksi
docker run -p 5001:8080 finalestari72/water-potability-model:latest

# Test prediksi
curl http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["ph","Hardness","Solids","Chloramines","Sulfate",
                  "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"],
      "data": [[7.0, 200.0, 20000.0, 7.0, 300.0, 400.0, 14.0, 66.0, 4.0]]
    }
  }'
```

---

## 🔧 Jalankan Lokal

```bash
# Install dependencies
pip install mlflow==2.19.0 scikit-learn==1.5.2 pandas numpy matplotlib seaborn

# Jalankan MLflow Project secara lokal
mlflow run MLProject \
  -P n_estimators=200 \
  -P max_depth=10 \
  --env-manager=local

# Lihat hasil di MLflow UI
mlflow ui --port 5000
# Buka http://127.0.0.1:5000
```

---

*Submission Kelas Machine Learning Operations — MSML Dicoding*
*Penulis: Fina Lestari | 2026*
