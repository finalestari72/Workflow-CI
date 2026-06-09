# Workflow-CI — Water Potability

Repositori ini berisi workflow CI untuk **Kriteria 3 – Submission Machine Learning MSML**.

---

## 📁 Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI
├── MLProject/
│   ├── modelling.py                        # Script training (entry point)
│   ├── conda.yaml                          # Environment dependencies
│   ├── MLProject                           # Konfigurasi MLflow Project
│   ├── water_potability_preprocessing.csv  # Dataset siap latih
│   └── DockerHub.txt                       # Tautan Docker Hub
└── README.md
```

---

## 🔐 Setup GitHub Secrets

Sebelum workflow berjalan, tambahkan secrets berikut di **GitHub → Settings → Secrets → Actions**:

| Secret | Nilai |
|---|---|
| `DAGSHUB_USERNAME` | Username DagsHub kamu |
| `DAGSHUB_TOKEN` | Token dari dagshub.com/user/settings/tokens |
| `DOCKERHUB_USERNAME` | Username Docker Hub kamu |
| `DOCKERHUB_TOKEN` | Access token dari hub.docker.com/settings/security |

---

## 🚀 Cara Kerja CI

### Trigger otomatis
Workflow berjalan setiap kali ada **push ke branch main** yang mengubah file di folder `MLProject/`.

### Trigger manual
Buka tab **Actions → CI Water Potability → Run workflow**, lalu isi parameter:

| Parameter | Default | Keterangan |
|---|---|---|
| `n_estimators` | 200 | Jumlah pohon Random Forest |
| `max_depth` | 10 | Kedalaman maksimum (0 = None) |
| `min_samples_split` | 2 | Min sampel untuk split node |
| `min_samples_leaf` | 1 | Min sampel di leaf node |
| `max_features` | sqrt | Fitur per split |

---

## 📋 Alur CI (ci.yml)

```
Push / Manual trigger
        │
        ▼
┌─────────────────┐
│   Job: train    │
│                 │
│ 1. Checkout     │
│ 2. Setup Python │
│ 3. Install deps │
│ 4. mlflow run   │ ← MLProject entry point
│ 5. Upload       │ ← artefak ke GitHub Actions (Skilled)
│ 6. Commit       │ ← artefak ke repo (Skilled)
└────────┬────────┘
         │ needs: train
         ▼
┌─────────────────┐
│   Job: docker   │
│                 │
│ 1. Login Docker │
│ 2. Build image  │ ← mlflow models build-docker (Advance)
│ 3. Push image   │ ← ke Docker Hub (Advance)
└─────────────────┘
```

---

## 📊 Penilaian

| Tingkat | Poin | Yang Dipenuhi |
|---|---|---|
| Basic | 2 | Folder MLProject + workflow CI berjalan |
| Skilled | 3 | + Artefak disimpan ke GitHub (upload + commit) |
| Advance | 4 | + Docker image di-build & push ke Docker Hub |

---

## 🔧 Jalankan Lokal (opsional)

```bash
pip install mlflow dagshub scikit-learn pandas numpy matplotlib

# Jalankan MLflow Project langsung
mlflow run MLProject -P n_estimators=200 -P max_depth=10 --env-manager=local
```

---

*Submission Kelas Machine Learning — MSML Dicoding*
