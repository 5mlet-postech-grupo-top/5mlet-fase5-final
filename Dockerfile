# 1. Imagem base leve e oficial do Python
FROM python:3.11-slim

# 2. Variáveis de ambiente para otimização do Python e MLOps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 3. Criação de um usuário não-root (Boas Práticas de Segurança)
RUN adduser --disabled-password --gecos "" appuser

# 4. Define o diretório de trabalho
WORKDIR /app

# 5. Copia os requisitos e instala (Aproveitando o cache do Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copia o código-fonte e os dados (Incluindo os artefatos do modelo)
COPY app ./app
COPY src ./src
COPY data ./data

# 7. Dá permissão ao usuário seguro para escrever no banco SQLite na pasta data
RUN chown -R appuser:appuser /app

# 8. Troca do usuário root para o usuário seguro
USER appuser

# 9. Expõe a porta do FastAPI
EXPOSE 8000

# 10. Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]