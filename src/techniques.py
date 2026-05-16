"""
techniques.py — As 4 técnicas de prompting
Referência: Aulas 06 (ZS, FS, CoT) e 07 (Role Prompting)

Cada função recebe a definição da tarefa e o input, e retorna o prompt
montado pronto para envio ao LLMClient.

Técnicas implementadas:
1. Zero-Shot  — instrução direta, sem exemplos (Aula 06)
2. Few-Shot   — com 2-3 exemplos do data/examples.json (Aula 06)
3. Chain-of-Thought — raciocínio passo a passo (Aula 06)
4. Role Prompting — com persona via system prompt (Aula 07)
"""

from src.prompt_builder import montar_prompt, adicionar_exemplos, adicionar_cot


def zero_shot(tarefa: dict, input_texto: str) -> str:
    """
    Técnica Zero-Shot: instrução clara sem exemplos (Aula 06).

    Funciona bem quando:
    - A tarefa é comum (classificação, resumo, tradução)
    - A instrução é clara e específica
    - O formato de output é simples

    Regra de ouro (Aula 06): Sempre comece com zero-shot.
    Se não funcionar bem, adicione few-shot.

    Args:
        tarefa: Dicionário da tarefa (de tasks.py) com instrução e formato
        input_texto: O texto a ser processado

    Returns:
        Prompt completo como string
    """
    return montar_prompt(
        instrucao=tarefa["instrucao"],
        contexto=tarefa.get("contexto", ""),
        input_dados=input_texto,
        formato_output=tarefa["formato_output"],
    )


def few_shot(tarefa: dict, input_texto: str, exemplos: list[dict]) -> str:
    """
    Técnica Few-Shot: aprendizado in-context com exemplos (Aula 06).

    Baseado em Brown et al. (2020): modelos podem "aprender" um padrão
    apenas com exemplos no prompt, sem retreinamento.

    Boas práticas aplicadas:
    - 2-3 exemplos (sweet spot)
    - Exemplos cobrem categorias diversas
    - Formato Input → Output consistente

    Args:
        tarefa: Dicionário da tarefa com instrução e formato
        input_texto: O texto a ser processado
        exemplos: Lista de {'input': ..., 'output': ...}

    Returns:
        Prompt com exemplos inseridos antes do input real
    """
    prompt_base = montar_prompt(
        instrucao=tarefa["instrucao"],
        contexto=tarefa.get("contexto", ""),
        input_dados=input_texto,
        formato_output=tarefa["formato_output"],
    )
    return adicionar_exemplos(prompt_base, exemplos)


def chain_of_thought(tarefa: dict, input_texto: str, passos: list[str]) -> str:
    """
    Técnica Chain-of-Thought: raciocínio explícito passo a passo (Aula 06).

    Baseado em Wei et al. (2022): adicionar raciocínio intermediário melhora
    drasticamente a performance em tarefas de análise e classificação complexa.

    Dois modos possíveis:
    - Zero-shot CoT: "Pense passo a passo" (simples)
    - Few-shot CoT: passos explícitos por domínio (mais controle)
    Aqui usamos a versão com passos explícitos para máximo controle.

    Args:
        tarefa: Dicionário da tarefa com instrução e formato
        input_texto: O texto a ser processado
        passos: Lista de passos de raciocínio a seguir

    Returns:
        Prompt com bloco de CoT inserido
    """
    prompt_base = montar_prompt(
        instrucao=tarefa["instrucao"],
        contexto=tarefa.get("contexto", ""),
        input_dados=input_texto,
        formato_output=tarefa["formato_output"],
    )
    return adicionar_cot(prompt_base, passos)


def role_prompting(tarefa: dict, input_texto: str, persona: dict) -> tuple[str, str]:
    """
    Técnica Role Prompting: persona via system prompt (Aula 07).

    O system prompt define a persona que "controla" todo o comportamento
    do modelo: tom, expertise, limites e formato de resposta.

    Diferente das outras técnicas, retorna uma TUPLA (system, user)
    porque o sistema de chat da API separa esses dois roles.
    O LLMClient recebe system e user separadamente.

    Boas práticas (Aula 07):
    - Persona específica (especialidade, anos de experiência)
    - Tom definido explicitamente
    - Limites claros de atuação

    Args:
        tarefa: Dicionário da tarefa com instrução e formato
        input_texto: O texto a ser processado
        persona: Dicionário com 'system_prompt' e metadados da persona

    Returns:
        Tuple (system_prompt, user_prompt) para uso no LLMClient
    """
    system_prompt = persona.get("system_prompt", "Você é um assistente especializado.")

    user_prompt = montar_prompt(
        instrucao=tarefa["instrucao"],
        contexto="",  # Contexto já está na persona (system)
        input_dados=input_texto,
        formato_output=tarefa["formato_output"],
    )

    return system_prompt, user_prompt
