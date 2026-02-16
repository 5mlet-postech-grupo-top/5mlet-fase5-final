
# 📊 PEDE – Datathon Machine Learning Engineering

---

# 🎯 Objetivo

Desenvolver um modelo preditivo capaz de identificar **risco de defasagem educacional** de alunos da Associação Passos Mágicos, permitindo **intervenção pedagógica antecipada**.

O sistema foi desenvolvido seguindo boas práticas de **Machine Learning Engineering e MLOps**, incluindo:

- Treinamento automatizado com múltiplos datasets
- API REST para predição
- Explicabilidade (Top fatores de risco)
- Monitoramento e detecção de drift
- Estrutura pronta para deploy em produção

---

# 🏗️ Arquitetura da Solução

```mermaid
flowchart LR
    A[Datasets na pasta /data] --> B[Padronização de Schema]
    B --> C[Feature Engineering]
    C --> D[Treinamento Modelo]
    D --> E[Validação Temporal]
    D --> F[Salvar Artefatos]
    F --> G[API FastAPI]
    G --> H[/predict]
    G --> I[/explain]
    G --> J[/drift]
```

---

# 📂 Estratégia Temporal com Dois Datasets

O projeto utiliza dois datasets:

1. **PEDE_PASSOS_DATASET_FIAP.xlsx**
2. **BASE DE DADOS PEDE 2024 - DATATHON.xlsx**

## 🔹 Por que usar ambos?

A estratégia foi desenhada para:

- Aumentar volume de dados para treino
- Melhorar robustez estatística
- Simular cenário real de produção
- Permitir validação temporal

## 🔹 Como os datasets são usados

| Fase | Dataset | Objetivo |
|------|----------|----------|
| Treinamento principal | FIAP | Aprender padrão histórico |
| Complemento de treino | Base 2024 | Aumentar diversidade |
| Validação temporal | Base 2024 | Testar generalização |
| Drift | Produção vs treino | Monitorar estabilidade |

## 🔹 Controle de Leakage

O modelo:

- Nunca utiliza informações futuras para prever passado
- Constrói o target como:
  
  > DEFASAGEM < 0 → aluno está atrás do nível ideal

- Separa corretamente features e target antes do treinamento

---

# 🧪 Seção de Validação Temporal

Além do split tradicional (train/validation), foi implementada:

## ✔ Validação Estratificada

- 80% treino
- 20% validação
- Estratificação pela classe de risco

## ✔ Validação Temporal (simulada)

Os dados mais recentes (dataset 2024) são utilizados como proxy de produção para verificar:

- Se o modelo mantém desempenho
- Se há mudança na distribuição
- Se as métricas se mantêm estáveis

Essa abordagem reduz risco de overfitting histórico.

---

# 📈 Justificativa Formal das Métricas

O problema é um problema de **classificação binária com impacto social**.

### 🎯 Métricas utilizadas:

## 🔹 AUC-ROC
Mede capacidade geral de separação entre classes.
Independe de threshold.

## 🔹 Recall (Classe 1 – Risco)
Principal métrica de negócio.

Justificativa:

> Falsos negativos representam alunos em risco que não receberiam intervenção pedagógica.

Minimizar falsos negativos é prioridade.

## 🔹 F1-Score
Balanceia precisão e recall.

---

# 📌 Endpoint `/predict`

Exemplo:

```json
{
  "IDADE": 13,
  "INDE": 6.7,
  "IEG": 7.1,
  "IDA": 6.2,
  "PONTO_VIRADA": 0,
  "FASE_TURMA": "3-A",
  "PEDRA": "Ametista",
  "INSTITUICAO": "Escola Estadual"
}
```

Retorno inclui:

- risk_score
- risk_class
- risk_level
- top_risk_factors

---

# 📊 Monitoramento

## 🔹 /metrics
Exposição para Prometheus

## 🔹 /drift
Cálculo de PSI (Population Stability Index)

Guia de interpretação:

- PSI < 0.10 → Sem drift
- 0.10–0.25 → Drift moderado
- > 0.25 → Drift significativo

## 🔹 /explain
Histórico e explicação de predições por aluno

---

# 🏁 Conclusão Técnica

A solução entrega:

✔ Modelo robusto treinado com múltiplos datasets  
✔ Estratégia temporal adequada  
✔ Controle de leakage  
✔ Métricas alinhadas ao impacto social  
✔ Explicabilidade via SHAP  
✔ Monitoramento de drift  
✔ Arquitetura pronta para produção  


---

# 📘 Explicação dos Campos de Resposta da API

Quando o endpoint `/predict` é chamado, a API retorna alguns campos fundamentais para interpretação do resultado.

## 🔹 risk_score

É a **probabilidade estimada pelo modelo** de que o aluno esteja em risco de defasagem.

- Valor contínuo entre **0 e 1**
- Quanto mais próximo de 1, maior o risco estimado

Exemplo:
```
0.82 → 82% de probabilidade de risco
```

Esse valor é gerado a partir de `predict_proba()` do modelo RandomForest.

---

## 🔹 risk_class

É a **classe final binária**, calculada a partir do `risk_score` comparado com o threshold definido (padrão: 0.5).

Regra:

```
Se risk_score >= threshold → risk_class = 1 (alto risco)
Se risk_score < threshold → risk_class = 0 (baixo risco)
```

Esse campo facilita decisões operacionais.

---

## 🔹 risk_level

Representação textual da classe:

- `"alto"` → aluno classificado como risco
- `"baixo"` → aluno classificado como não risco

Foi criado para facilitar leitura por áreas pedagógicas e não técnicas.

---

## 🔹 top_risk_factors

Lista com os **5 fatores que mais influenciaram a decisão do modelo**.

Cada item contém:

```
{
  "feature": nome_da_variavel,
  "impact": valor_de_contribuicao
}
```

- Impactos positivos → reduzem risco
- Impactos negativos → aumentam risco

Esses valores são calculados via **SHAP (SHapley Additive Explanations)**.

Exemplo:

```
[
  {"feature": "INDE", "impact": -0.34},
  {"feature": "PONTO_VIRADA", "impact": -0.21},
  {"feature": "IEG", "impact": -0.18}
]
```

Isso permite transparência e explicabilidade do modelo.

---
