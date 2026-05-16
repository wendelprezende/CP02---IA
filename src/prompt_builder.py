"""
prompt_builder.py — Montagem de prompts por anatomia
Referência: Aula 05 — Os 4 componentes de um prompt eficaz

Princípios aplicados (Aula 05):
- Separar instrução de dados (usa delimitadores ---)
- Formato de output explícito e específico
- Linguagem positiva (o que fazer, não o que não fazer)
- Validação de componentes não vazios
"""


def montar_prompt(
    instrucao: str,
    contexto: str = "",
    input_dados: str = "",
    formato_output: str = "",
) -> str:
    """
    Monta um prompt estruturado com os 4 componentes da anatomia (Aula 05).

    Princípio fundamental: separar a instrução dos dados de entrada
    evita confusão do modelo e reduz riscos de prompt injection.

    Args:
        instrucao: O que o modelo deve fazer (OBRIGATÓRIO)
        contexto: Background, domínio e restrições (recomendado)
        input_dados: Dados sobre os quais a tarefa será executada
        formato_output: Como a resposta deve ser estruturada

    Returns:
        String com o prompt completo, pronto para envio ao LLM

    Raises:
        ValueError: Se a instrução estiver vazia
    """
    if not instrucao or not instrucao.strip():
        raise ValueError("A instrução não pode estar vazia. É o componente obrigatório do prompt.")

    partes = []

    # Contexto vem primeiro para "preparar" o modelo antes da instrução
    if contexto and contexto.strip():
        partes.append(f"[Contexto]\n{contexto.strip()}")

    # Instrução é o núcleo do prompt
    partes.append(f"[Instrução]\n{instrucao.strip()}")

    # Input separado por delimitadores para evitar conflito com a instrução
    if input_dados and input_dados.strip():
        partes.append(f"[Input]\n---\n{input_dados.strip()}\n---")

    # Formato de output orienta a estrutura da resposta
    if formato_output and formato_output.strip():
        partes.append(f"[Formato de Resposta]\n{formato_output.strip()}")

    return "\n\n".join(partes)


def adicionar_exemplos(prompt: str, exemplos: list[dict]) -> str:
    """
    Adiciona exemplos few-shot a um prompt existente (Aula 06).

    Os exemplos são inseridos após o contexto e instrução, mas antes do
    input real. Formato padrão: Input: "..." → Output: "..."

    Boas práticas (Aula 06):
    - 3-5 exemplos é o sweet spot
    - Exemplos devem cobrir todas as categorias possíveis
    - Formato consistente em todos os exemplos

    Args:
        prompt: Prompt base já montado com montar_prompt()
        exemplos: Lista de dicts com chaves 'input' e 'output'

    Returns:
        Prompt com exemplos inseridos antes do [Input] real
    """
    if not exemplos:
        return prompt

    linhas_exemplos = ["[Exemplos]"]
    for i, ex in enumerate(exemplos, 1):
        entrada = str(ex.get("input", "")).strip()
        saida = str(ex.get("output", "")).strip()
        if entrada and saida:
            linhas_exemplos.append(f'Exemplo {i}:\nInput: "{entrada}"\nOutput: "{saida}"')

    bloco_exemplos = "\n\n".join(linhas_exemplos)

    # Insere os exemplos antes do [Input] real (se existir)
    if "[Input]" in prompt:
        return prompt.replace("[Input]", f"{bloco_exemplos}\n\n[Input]")

    return f"{prompt}\n\n{bloco_exemplos}"


def adicionar_cot(prompt: str, passos: list[str]) -> str:
    """
    Adiciona instruções de raciocínio passo a passo (Chain-of-Thought) (Aula 06).

    Baseado em Wei et al. (2022): explicitar o processo de raciocínio
    melhora drasticamente a performance em tarefas que exigem análise.

    Args:
        prompt: Prompt base já montado
        passos: Lista de passos de raciocínio a seguir

    Returns:
        Prompt com bloco de CoT inserido antes do [Formato de Resposta]
    """
    if not passos:
        return prompt

    numerados = "\n".join(f"{i}. {passo.strip()}" for i, passo in enumerate(passos, 1))
    bloco_cot = f"[Raciocínio — Pense Passo a Passo]\n{numerados}"

    # Insere antes do [Formato de Resposta] para guiar antes de responder
    if "[Formato de Resposta]" in prompt:
        return prompt.replace("[Formato de Resposta]", f"{bloco_cot}\n\n[Formato de Resposta]")

    return f"{prompt}\n\n{bloco_cot}"
