Dataset:
Download from Kaggle: Brain MRI Dataset
Place inside /data folder with structure:
data/
  tumor/
  no_tumor/

  
mlops-project/
│
├── data/
│
├── pipeline/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│
├── inference/
│
├── deployment/
│   ├── api/
│   ├── docker/
│
├── monitoring/
│
├── results/
│
├── requirements.txt
├── README.md


1. Data Collection / Input
        ↓
2. Data Processing (src/)
        ↓
3. Model Training (src/)
        ↓
4. Evaluation (Results/confusion_matrix)
        ↓
5. Inference (inference/)
        ↓
6. API Deployment (api/)
        ↓
7. Monitoring (monitoring/)
        ↓
8. Containerization (docker/, docker-compose.yml)
