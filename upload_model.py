import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# 1. Carrega as variáveis do arquivo .env para a memória
load_dotenv()

# 2. Puxa o token com segurança
meu_token = os.getenv("HF_TOKEN")

if not meu_token:
    raise ValueError("Token do Hugging Face não encontrado! Verifique o arquivo .env.")

# 3. Faz o login
login(meu_token)

api = HfApi()

api.upload_folder(
    folder_path="app/model",
    repo_id="tiagoparibeiro/passos-magicos-model",
    repo_type="model"
)
print("Upload concluído com segurança!")