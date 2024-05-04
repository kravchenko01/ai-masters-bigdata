# MLflow

## MLProject

Содержит конфигурацию запуска `train.py`

## MLflow tracking server
Запуск сервера (в домашней директории `~`):
```bash
conda activate dsenv
mlflow server --port 6207 --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
```

mlflow ui доступен по адресу localhost:6207 после проброса порта 6207 при подключении к серверу: 
```bash
ssh -L 6207:localhost:6207 -A kravchenko01@158.160.81.38
```

## Train

Запуск тренировки в новом терминале:
```bash
conda activate dsenv
export MLFLOW_TRACKING_URI=http://localhost:6207
cd ai-masters-bigdata/projects/5mla/
mlflow run . -P train_path=/home/users/datasets/criteo/train1000.txt --env-manager=local
```

## Inference

Запуск инференса обученной модели:
```bash
mlflow models serve -p 7207 --env-manager=local -m mlruns/0/98c500413a4c44ac9c237debf57afe81/artifacts/model
```
