
# 📊 PEDE – Datathon Machine Learning Engineering

## 🎯 Objetivo

Desenvolver um modelo preditivo para identificar **risco de defasagem educacional** de alunos da Associação Passos Mágicos, permitindo intervenção pedagógica antecipada.

---

# 🚀 Como Executar

## 1️⃣ Coloque os datasets na pasta `data/`

- `PEDE_PASSOS_DATASET_FIAP.xlsx`
- `BASE DE DADOS PEDE 2024 - DATATHON.xlsx`

## 2️⃣ Treinar o modelo

```bash
python -m src.train
```

## 3️⃣ Subir a API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# 📌 Endpoint `/predict`

Exemplo de requisição:

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

---

# 📘 Explicação dos Parâmetros

## 🔹 IDADE
Idade do aluno no ano base.  
No dataset original: `IDADE_ALUNO_2020`.

Impacto:
- Usada para contextualizar maturidade.
- Utilizada no cálculo derivado `ANOS_PM_POR_IDADE`.

---

## 🔹 INDE
**Índice do Desenvolvimento Educacional.**

É a métrica geral do aluno baseada na ponderação de:
- IAN (Adequação ao nível)
- IDA (Aprendizagem)
- IEG (Engajamento)
- IAA (Autoavaliação)
- IPS (Psicossocial)
- IPP (Psicopedagógico)
- IPV (Ponto de Virada)

Escala aproximada: 0 a 10.

- < 6 → sinal de alerta
- 6–7.5 → médio
- > 8 → alto desempenho

---

## 🔹 IEG
**Indicador de Engajamento.**

Mede participação, envolvimento e comprometimento do aluno.

Baixo engajamento costuma aumentar risco de defasagem.

---

## 🔹 IDA
**Indicador de Aprendizagem.**

Representa desempenho acadêmico (notas).  
É um dos maiores preditores de risco.

---

## 🔹 PONTO_VIRADA
Campo booleano:

- 1 → atingiu o “Ponto de Virada”
- 0 → não atingiu

Se for 0, o risco tende a aumentar.

---

## 🔹 FASE_TURMA

Representa:

- Fase = nível de aprendizado
- Turma = grupo dentro da fase

Exemplo:
"3-A" → Fase 3, Turma A

É uma variável categórica utilizada via OneHotEncoding.

---

## 🔹 PEDRA

Classificação baseada no INDE:

- Quartzo → 2,405 a 5,506
- Ágata → 5,506 a 6,868
- Ametista → 6,868 a 8,230
- Topázio → 8,230 a 9,294

Ajuda o modelo a capturar faixas de desempenho.

---

## 🔹 INSTITUICAO

Instituição de ensino do aluno.

Captura contexto educacional e possíveis diferenças estruturais.

---

# 🧠 Como Explicar na Banca

> “O modelo utiliza indicadores pedagógicos estruturais (INDE, IDA, IEG), indicadores comportamentais (Ponto de Virada), e contexto educacional (Fase, Pedra, Instituição), permitindo capturar tanto desempenho acadêmico quanto engajamento e adequação ao nível.”

---

# 📈 Métricas Utilizadas

- AUC-ROC
- Recall (prioridade para evitar falsos negativos)
- F1-score

---

# 📊 Monitoramento

- `/metrics` → Prometheus
- `/drift` → PSI para detecção de mudança de distribuição
- `/explain` → histórico e fatores de risco do aluno

---

# 🏁 Conclusão

O sistema entrega:

✔ Treinamento automático com múltiplos datasets  
✔ API REST para predição  
✔ Explicabilidade (Top fatores de risco)  
✔ Monitoramento e detecção de drift  
✔ Estrutura pronta para deploy em produção  

