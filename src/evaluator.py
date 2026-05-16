"""
evaluator.py — Módulo de avaliação de qualidade e métricas
Referência: Checkpoint 02 — Critérios de avaliação

Métricas implementadas:
- contar_tokens: via tiktoken (contagem local, sem chamar LLM)
- medir_acuracia: match exato ou por keywords
- medir_consistencia: % de respostas iguais para mesmo prompt
- testar_temperatura: análise do impacto de temp=0.1, 0.5, 1.0
"""

import json
import re
import time
from typing import Optional

import tiktoken


def contar_tokens(texto: str, modelo: str = "gpt-4") -> int:
    """
    Conta tokens no texto usando tiktoken (contagem local, sem API).

    Tiktoken é a biblioteca oficial da OpenAI para contar tokens.
    Usamos o encoding do GPT-4 como proxy para modelos Ollama,
    pois a maioria usa tokenizers compatíveis com BPE.

    Fallback: estimativa baseada em palavras (≈ 0.75 tokens/palavra),
    usada quando tiktoken não consegue baixar os arquivos BPE.

    Args:
        texto: Texto a ser tokenizado
        modelo: Modelo de referência para o tokenizer (padrão: gpt-4)

    Returns:
        Número inteiro de tokens
    """
    try:
        enc = tiktoken.encoding_for_model(modelo)
        return len(enc.encode(texto))
    except KeyError:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(texto))
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: estimativa word-based (OpenAI: ~0.75 tokens por palavra em inglês,
    # ~1.3 tokens por palavra em português — usamos 1.0 como média conservadora)
    palavras = len(texto.split())
    chars = len(texto)
    # Heurística: max(palavras, chars/4) — mais próximo do real para texto misto
    return max(palavras, chars // 4)


def medir_acuracia(resposta: str, esperado: str | dict) -> float:
    """
    Mede a acurácia da resposta comparando com o valor esperado.

    Estratégia adaptativa por tipo de tarefa:
    - Para strings simples (classificação): match exato normalizado
    - Para dicts (extração JSON): % de campos corretos
    - Fallback: verifica se palavras-chave do esperado estão na resposta

    Args:
        resposta: Texto gerado pelo LLM
        esperado: Valor esperado (string para classificação, dict para extração)

    Returns:
        Score de 0.0 a 1.0 (1.0 = perfeito, 0.0 = errado)
    """
    resposta_normalizada = _normalizar(str(resposta))

    # ── Caso 1: esperado é um dicionário (extração JSON) ──
    if isinstance(esperado, dict):
        return _acuracia_json(resposta_normalizada, esperado)

    # ── Caso 2: esperado é string ──
    esperado_normalizado = _normalizar(str(esperado))

    # Match exato (ideal para classificação)
    if resposta_normalizada == esperado_normalizado:
        return 1.0

    # Match por keyword: o esperado está contido na resposta?
    if esperado_normalizado in resposta_normalizada:
        return 0.8

    # Match parcial: ao menos metade das palavras do esperado aparecem na resposta
    palavras_esperadas = set(esperado_normalizado.split())
    palavras_resposta = set(resposta_normalizada.split())
    if palavras_esperadas:
        overlap = len(palavras_esperadas & palavras_resposta) / len(palavras_esperadas)
        if overlap >= 0.5:
            return round(overlap * 0.6, 2)  # Penaliza match parcial

    return 0.0


def _acuracia_json(resposta_texto: str, esperado: dict) -> float:
    """
    Calcula acurácia para tarefas de extração JSON.
    Extrai JSON da resposta e compara campo a campo.
    """
    # Tenta extrair JSON da resposta (modelo pode adicionar texto extra)
    resposta_dict = _extrair_json(resposta_texto)

    if resposta_dict is None:
        return 0.0

    campos_certos = 0
    total_campos = len(esperado)

    for chave, valor_esperado in esperado.items():
        valor_resposta = resposta_dict.get(chave, "")

        # Normaliza e compara
        ve = _normalizar(str(valor_esperado))
        vr = _normalizar(str(valor_resposta))

        if ve == vr:
            campos_certos += 1
        elif ve in vr or vr in ve:
            campos_certos += 0.5  # Match parcial

    return round(campos_certos / total_campos, 2) if total_campos > 0 else 0.0


def _extrair_json(texto: str) -> Optional[dict]:
    """Extrai o primeiro JSON válido de um texto."""
    # Primeiro, tenta parsear o texto completo
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    # Busca por blocos JSON com regex
    matches = re.findall(r"\{[^{}]+\}", texto, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def _normalizar(texto: str) -> str:
    """Normaliza texto para comparação: lowercase, sem espaços extras."""
    return texto.lower().strip()


def medir_consistencia(respostas: list[str]) -> float:
    """
    Mede consistência: percentual de respostas idênticas para o mesmo prompt.

    Consistência é crucial em produção — um prompt que às vezes diz POSITIVO
    e às vezes NEGATIVO para o mesmo texto não pode ser usado.

    Algoritmo: encontra a resposta mais frequente e calcula sua proporção.

    Args:
        respostas: Lista de respostas geradas pelo LLM para o mesmo prompt

    Returns:
        Float de 0.0 a 1.0 (1.0 = todas as respostas iguais)
    """
    if not respostas:
        return 0.0

    if len(respostas) == 1:
        return 1.0

    # Normaliza todas as respostas
    normalizadas = [_normalizar(r) for r in respostas]

    # Conta a frequência de cada resposta
    contagem: dict[str, int] = {}
    for r in normalizadas:
        contagem[r] = contagem.get(r, 0) + 1

    # Consistência = frequência da resposta mais comum / total
    max_freq = max(contagem.values())
    return round(max_freq / len(respostas), 2)


def testar_temperatura(
    llm_client,
    prompt: str,
    system: Optional[str] = None,
    temperaturas: list[float] = None,
    repeticoes: int = 3,
) -> list[dict]:
    """
    Testa o impacto da temperatura na consistência e qualidade das respostas.

    Temperatura baixa (0.1): respostas mais determinísticas e consistentes
    Temperatura média (0.5): equilíbrio entre consistência e naturalidade
    Temperatura alta (1.0): respostas mais criativas e variadas

    Args:
        llm_client: Instância do LLMClient
        prompt: Prompt a ser testado
        system: System prompt opcional
        temperaturas: Lista de temperaturas a testar (padrão: [0.1, 0.5, 1.0])
        repeticoes: Quantas vezes rodar cada temperatura

    Returns:
        Lista de dicts com resultados por temperatura
    """
    if temperaturas is None:
        temperaturas = [0.1, 0.5, 1.0]

    resultados = []

    for temp in temperaturas:
        respostas = []
        tokens_lista = []
        tempos_lista = []

        for _ in range(repeticoes):
            try:
                resultado = llm_client.chat(
                    prompt=prompt,
                    system=system,
                    temperature=temp,
                    max_tokens=256,
                )
                respostas.append(resultado["resposta"])
                tokens_lista.append(resultado["tokens_resposta"])
                tempos_lista.append(resultado["tempo_ms"])
                time.sleep(0.5)  # Pequena pausa para não sobrecarregar a API
            except RuntimeError as e:
                print(f"    ⚠  Erro na temperatura {temp}: {e}")
                break

        if respostas:
            consistencia = medir_consistencia(respostas)
            resultados.append({
                "temperatura": temp,
                "respostas": respostas,
                "consistencia": consistencia,
                "tokens_medio": round(sum(tokens_lista) / len(tokens_lista), 1) if tokens_lista else 0,
                "tempo_medio_ms": round(sum(tempos_lista) / len(tempos_lista)) if tempos_lista else 0,
            })

    return resultados
