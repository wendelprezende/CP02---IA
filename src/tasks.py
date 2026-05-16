"""
tasks.py — Definição das tarefas do domínio
Referência: Aula 08 — Prompts para Tarefas Específicas

Domínio: E-commerce (análise de avaliações de clientes brasileiros)
Problema: Automatizar a triagem e análise de reviews para reduzir
          o tempo de resposta do time de CX e gerar insights operacionais.

Cada tarefa é um dict seguindo a estrutura definida no Checkpoint 02:
- nome: identificador único
- tipo: classificacao | extracao | sumarizacao | geracao | traducao | code
- instrucao: o que o LLM deve fazer
- contexto: background do domínio
- formato_output: como a resposta deve ser estruturada
- exemplos_fewshot: lista de {'input', 'output'} para few-shot
- passos_cot: lista de passos para chain-of-thought
- persona: chave do prompts/system_prompts.json
"""


# ─── TAREFA 1: Classificação de Sentimento ────────────────────────────────────
# Tipo: classificacao
# Objetivo: Categorizar automaticamente o tom da avaliação para triagem de tickets
# Referência Aula 08: "Classificação: atribuir categorias a textos"

CLASSIFICACAO_SENTIMENTO = {
    "nome": "classificacao_sentimento",
    "tipo": "classificacao",
    "instrucao": (
        "Classifique a avaliação do cliente como POSITIVO, NEGATIVO, NEUTRO ou MISTO. "
        "POSITIVO: avaliação claramente satisfeita. "
        "NEGATIVO: avaliação claramente insatisfeita. "
        "NEUTRO: avaliação factual sem emoção clara. "
        "MISTO: avaliação com pontos positivos E negativos relevantes."
    ),
    "contexto": (
        "Você é um sistema de análise de sentimento para um e-commerce brasileiro. "
        "Avalie o tom emocional predominante, não apenas palavras isoladas."
    ),
    "formato_output": (
        "Responda APENAS com uma das palavras: POSITIVO, NEGATIVO, NEUTRO ou MISTO. "
        "Sem pontuação, sem explicação, sem espaços extras."
    ),
    "exemplos_fewshot": [
        {
            "input": "Produto incrível! Chegou antes do prazo e funciona perfeitamente.",
            "output": "POSITIVO",
        },
        {
            "input": "Horrível. Produto quebrou em 2 dias e o vendedor não responde.",
            "output": "NEGATIVO",
        },
        {
            "input": "A câmera é excelente, mas a bateria dura apenas 4 horas. Fica a dica.",
            "output": "MISTO",
        },
        {
            "input": "Produto recebido. Conforme descrito no anúncio.",
            "output": "NEUTRO",
        },
    ],
    "passos_cot": [
        "Leia a avaliação completa com atenção",
        "Liste mentalmente as expressões de satisfação (ex: 'adorei', 'perfeito', 'rápido')",
        "Liste mentalmente as expressões de insatisfação (ex: 'péssimo', 'defeituoso', 'atrasou')",
        "Avalie o peso emocional de cada grupo",
        "Se satisfação >> insatisfação: POSITIVO",
        "Se insatisfação >> satisfação: NEGATIVO",
        "Se satisfação ≈ insatisfação com relevância igual: MISTO",
        "Se nenhuma emoção clara: NEUTRO",
        "Responda SOMENTE com: POSITIVO, NEGATIVO, NEUTRO ou MISTO",
    ],
    "persona": "analista_cx",
}


# ─── TAREFA 2: Extração de Dados Estruturados ─────────────────────────────────
# Tipo: extracao (NER — Named Entity Recognition)
# Objetivo: Extrair produto, preço, defeito e ação desejada em JSON para o CRM
# Referência Aula 08: "Extração (NER): identificar e extrair entidades em JSON"

EXTRACAO_DADOS = {
    "nome": "extracao_dados",
    "tipo": "extracao",
    "instrucao": (
        "Extraia as seguintes informações da reclamação do cliente e retorne em JSON válido:\n"
        "- produto: nome exato do produto (string ou null)\n"
        "- preco: valor monetário no formato R$X.XXX,XX (string ou null)\n"
        "- defeito: descrição do problema relatado (string ou null)\n"
        "- acao_solicitada: o que o cliente quer (troca | reembolso | reparo | outro)"
    ),
    "contexto": (
        "Sistema de extração de dados para alimentar o CRM de pós-venda de um e-commerce. "
        "As informações extraídas são usadas para triagem automática de tickets de suporte."
    ),
    "formato_output": (
        "Retorne APENAS o JSON válido, sem texto antes ou depois. "
        "Exemplo: {\"produto\": \"...\", \"preco\": \"...\", \"defeito\": \"...\", \"acao_solicitada\": \"...\"}"
    ),
    "exemplos_fewshot": [
        {
            "input": "Meu iPhone 14 de R$5.299 caiu e a tela rachou. Quero conserto.",
            "output": '{"produto": "iPhone 14", "preco": "R$5.299,00", "defeito": "tela rachou após queda", "acao_solicitada": "reparo"}',
        },
        {
            "input": "Camiseta Nike desbotou na primeira lavagem. Paguei R$89. Quero meu dinheiro de volta.",
            "output": '{"produto": "Camiseta Nike", "preco": "R$89,00", "defeito": "desbotou na primeira lavagem", "acao_solicitada": "reembolso"}',
        },
        {
            "input": "A impressora HP não imprime nada. Custou R$399. Mandem outra.",
            "output": '{"produto": "impressora HP", "preco": "R$399,00", "defeito": "não imprime", "acao_solicitada": "troca"}',
        },
    ],
    "passos_cot": [
        "Localize o nome do produto: procure substantivos que indicam um bem de consumo",
        "Localize o valor monetário: procure padrões como R$, reais, mil reais",
        "Normalize o preço para o formato R$X.XXX,XX",
        "Identifique o defeito: qual problema específico o cliente descreve?",
        "Identifique a ação desejada: o cliente quer troca, reembolso, reparo ou outro?",
        "Monte o JSON com os 4 campos",
        "Use null para campos não mencionados",
        "Retorne SOMENTE o JSON, sem explicação",
    ],
    "persona": "especialista_logistica",
}


# ─── TAREFA 3: Sumarização de Reviews ─────────────────────────────────────────
# Tipo: sumarizacao
# Objetivo: Condensar reviews longas em 1 frase para dashboards executivos
# Referência Aula 08: "Sumarização: condensar textos mantendo informações-chave"

SUMARIZACAO_REVIEW = {
    "nome": "sumarizacao_review",
    "tipo": "sumarizacao",
    "instrucao": (
        "Resuma a avaliação do cliente em UMA única frase objetiva. "
        "A frase deve capturar os pontos principais (positivos e negativos) "
        "de forma equilibrada e informativa."
    ),
    "contexto": (
        "Sistema de sumarização para exibição em dashboards executivos de e-commerce. "
        "Os resumos são lidos por gerentes de produto e CX que precisam de insights rápidos."
    ),
    "formato_output": (
        "Uma frase de no máximo 25 palavras, em português brasileiro, tom neutro e informativo. "
        "Sem introduções como 'Resumo:' ou 'O cliente diz:'. Apenas a frase diretamente."
    ),
    "exemplos_fewshot": [
        {
            "input": "Livro incrível! Conteúdo transformador, capa dura resistente, entrega em 3 dias. Preço de R$45 muito justo. Recomendo fortemente para quem quer desenvolver disciplina e bons hábitos.",
            "output": "Livro aprovado: conteúdo transformador, boa qualidade física e entrega em 3 dias pelo preço justo de R$45.",
        },
        {
            "input": "Fone chegou com a orelha direita sem som. Não funcionou de jeito nenhum. A devolução foi rápida, reembolso em 2 dias. O produto é ruim mas o suporte é excelente.",
            "output": "Produto com defeito (sem som numa orelha), porém suporte eficiente com reembolso em 2 dias.",
        },
        {
            "input": "O liquidificador é potente e silencioso para o nível de potência. Copo de vidro facilita a limpeza. Porém o plástico da tampa parece frágil e o preço de R$350 está alto.",
            "output": "Liquidificador potente e silencioso com copo de vidro prático, mas tampa frágil e preço elevado para R$350.",
        },
    ],
    "passos_cot": [
        "Identifique o produto ou serviço que está sendo avaliado",
        "Liste os principais pontos POSITIVOS mencionados (máx 3)",
        "Liste os principais pontos NEGATIVOS mencionados (máx 3)",
        "Avalie quais pontos são mais relevantes para um gestor de negócio",
        "Selecione os 2-3 pontos mais importantes (pos e/ou neg)",
        "Construa uma frase que mencione produto + pontos selecionados",
        "Verifique se a frase tem no máximo 25 palavras",
        "Se necessário, reduza usando verbos mais curtos ou eliminando adjetivos",
        "Retorne SOMENTE a frase final, sem qualquer prefixo",
    ],
    "persona": "redator_ecommerce",
}


# ─── Registro de todas as tarefas ─────────────────────────────────────────────
# Usado pelo main.py para iterar sobre todas as tarefas automaticamente

TODAS_AS_TAREFAS = [
    CLASSIFICACAO_SENTIMENTO,
    EXTRACAO_DADOS,
    SUMARIZACAO_REVIEW,
]
