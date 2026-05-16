# Prompt Toolkit — CP02

**Checkpoint 02 · Prompt Engineering & Artificial Intelligence · FIAP**

Toolkit Python modular que aplica automaticamente as 4 técnicas de prompting (Zero-Shot, Few-Shot, Chain-of-Thought e Role Prompting) a tarefas de e-commerce, compara resultados e recomenda a melhor abordagem.

---

## Domínio

**E-commerce — Análise de Avaliações de Clientes Brasileiros**

| Tarefa | Tipo | Descrição |
|---|---|---|
| `classificacao_sentimento` | Classificação | POSITIVO / NEGATIVO / NEUTRO / MISTO |
| `extracao_dados` | Extração (NER) | Produto, preço, defeito, ação → JSON |
| `sumarizacao_review` | Sumarização | Resume reviews longas em 1 frase executiva |

---

## Estrutura

```
prompt-toolkit/
├── main.py                  # Ponto de entrada
├── requirements.txt
├── .env.example
├── src/
│   ├── llm_client.py        # Conexão com Ollama API (Aula 05)
│   ├── prompt_builder.py    # Anatomia dos prompts (Aula 05)
│   ├── techniques.py        # ZS, FS, CoT, Role (Aulas 06+07)
│   ├── tasks.py             # 3 tarefas do domínio (Aula 08)
│   ├── evaluator.py         # Tokens, acurácia, consistência
│   └── report.py            # CSV + 3 gráficos + recomendação
├── data/
│   ├── inputs.json          # 7 inputs reais por tarefa
│   └── examples.json        # Exemplos few-shot
├── prompts/
│   ├── system_prompts.json  # 3 personas detalhadas
│   └── templates.json       # Templates por tarefa
└── output/
    ├── resultados.csv
    └── graficos/
```

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repo>
cd prompt-toolkit

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env se necessário (OLLAMA_HOST, OLLAMA_MODEL)

# 5. Inicie o Ollama e baixe o modelo
ollama serve
ollama pull gpt-oss:120b
```

---

## Execução

```bash
python main.py
```

O toolkit executa automaticamente:
1. Verifica conexão com Ollama
2. Carrega inputs, exemplos e personas
3. Aplica as 4 técnicas × 3 tarefas × 7 inputs
4. Gera `output/resultados.csv`
5. Gera 3 gráficos em `output/graficos/`
6. Exibe recomendações automáticas no terminal

---

## Stack Técnica

| Componente | Versão | Função |
|---|---|---|
| Python | 3.10+ | Linguagem principal |
| Ollama | local | Servidor LLM gratuito |
| gpt-oss:120b | — | Modelo de linguagem |
| tiktoken | ≥0.7 | Contagem de tokens |
| pandas | ≥2.2 | Manipulação de dados |
| matplotlib | ≥3.8 | Geração de gráficos |
| python-dotenv | ≥1.0 | Variáveis de ambiente |
