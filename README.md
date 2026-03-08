# ESTIN RAG

Système **RAG (Retrieval-Augmented Generation)** pour interroger le **règlement intérieur de l’ESTIN** (École Supérieure en Sciences et Technologies de l’Informatique et du Numérique, Béjaïa, Algérie).

## Fonctionnalités

- **Chat web** : interface pour poser des questions en langage naturel
- **Réponses basées sur le règlement** : recherche vectorielle (Pinecone) + LLM (Groq)
- **Mémoire de conversation** : contexte conservé par fil de discussion
- **Sources citées** : affichage des articles utilisés pour la réponse
- **Rendu Markdown** : tableaux, titres et listes correctement affichés dans l’interface

## Architecture

```
Frontend (chat)  →  API FastAPI  →  Agent RAG (LangGraph)
                                        ↓
                        Outil de recherche  →  Pinecone (vecteurs)
                        LLM Groq
```

- **Frontend** : `frontend/` (HTML, JS, CSS) — chat et affichage des réponses
- **API** : `src/api/main.py` — routes `/api/v1/ask`, `/health`, service du frontend
- **RAG** : `src/rag/` — agent avec outil de recherche et mémoire
- **Vector store** : `src/vectorstore/` — Pinecone (embeddings HuggingFace)

Pour plus de détails, voir [DOCUMENTATION_PROJET.md](DOCUMENTATION_PROJET.md).

## Prérequis

- **Python 3.10+**
- Clés API : **Groq**, **HuggingFace**, **Pinecone**

## Installation et démarrage

### 1. Cloner et environnement

```bash
git clone <url-du-repo>
cd RI_ESTIN_RAG
python -m venv venv
```

**Windows (PowerShell)** :
```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS** :
```bash
source venv/bin/activate
```

### 2. Dépendances

```bash
pip install -r requirements.txt
```

### 3. Variables d’environnement

Créer un fichier **`.env`** à la racine du projet :

```env
GROQ_API_KEY=votre_cle_groq
HF_API_KEY=votre_cle_huggingface
PINECONE_API_KEY=votre_cle_pinecone

# Optionnel
PINECONE_INDEX_NAME=estin-regulations
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Données et index vectoriel

- Placer le PDF du règlement dans : **`data/documents/Reglement-interieur-ESTIN.pdf`**
- Construire l’index (à faire une fois, ou après modification du PDF) :

```bash
python scripts/build_index.py
```

Pour recréer l’index from scratch, mettre `reset=True` dans l’appel à `build_index()` dans le script puis relancer la commande ci‑dessus.

### 5. Lancer l’application

Depuis la racine du projet :

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Ou :

```bash
python -m src.api.main
```

Ouvrir dans le navigateur : **http://localhost:8000/**  
- Interface : chat  
- Santé : http://localhost:8000/health  
- Doc API : http://localhost:8000/docs  

## Structure du projet

```
RI_ESTIN_RAG/
├── frontend/           # Interface chat (HTML, JS, CSS)
├── src/
│   ├── api/            # FastAPI (routes, CORS, static)
│   ├── config/         # Settings (.env)
│   ├── data_processing/# Chargement PDF + chunking par article
│   ├── embeddings/     # HuggingFace (multilingual-e5-large)
│   ├── rag/            # Agent (outil recherche + mémoire)
│   └── vectorstore/    # Pinecone
├── data/documents/     # PDF du règlement
├── scripts/
│   └── build_index.py  # Construction de l’index Pinecone
├── .env                # Clés API (à créer, ne pas committer)
├── requirements.txt
└── DOCUMENTATION_PROJET.md
```


## Licence

MIT 
