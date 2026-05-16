"""
main.py — Ponto de entrada do Prompt Toolkit
Checkpoint 02 — Prompt Engineering & Artificial Intelligence — FIAP

Domínio: E-commerce (análise de avaliações de clientes brasileiros)
Stack:   Ollama API + tiktoken + pandas + matplotlib

Fluxo de execução:
1. Verifica conexão com Ollama
2. Carrega inputs, exemplos e personas dos arquivos JSON
3. Para cada tarefa (classificação, extração, sumarização):
   a. Para cada técnica (ZS, FS, CoT, Role):
      - Monta o prompt
      - Envia ao LLM
      - Mede tokens, tempo e acurácia
4. Gera relatório: CSV + 3 gráficos + recomendações
5. Executa análise de temperatura no melhor prompt identificado

Uso:
    python main.py
"""

import json
import sys
import time
from pathlib import Path

from src.llm_client import LLMClient
from src.prompt_builder import montar_prompt
from src.techniques import zero_shot, few_shot, chain_of_thought, role_prompting
from src.tasks import TODAS_AS_TAREFAS
from src import evaluator
from src import report


# ─── Constantes de configuração ───────────────────────────────────────────────

DATA_DIR = Path("data")
PROMPTS_DIR = Path("prompts")

TEMPERATURA_PADRAO = 0.0      # Determinístico para reprodutibilidade
MAX_TOKENS_PADRAO = 256
REPETICOES_CONSISTENCIA = 3   # Vezes que cada prompt é executado para medir consistência


def carregar_json(caminho: Path) -> dict | list:
    """Carrega um arquivo JSON com tratamento de erros."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def executar_tecnica(
    nome_tecnica: str,
    tarefa: dict,
    input_texto: str,
    exemplos: list,
    persona: dict,
    llm: LLMClient,
) -> dict:
    """
    Executa uma técnica de prompting e retorna as métricas da execução.

    Args:
        nome_tecnica: 'zero_shot' | 'few_shot' | 'chain_of_thought' | 'role_prompting'
        tarefa: Definição da tarefa (de tasks.py)
        input_texto: Texto de entrada a ser processado
        exemplos: Exemplos few-shot carregados de data/examples.json
        persona: Persona carregada de prompts/system_prompts.json
        llm: Instância do LLMClient

    Returns:
        Dict com prompt, resposta e métricas
    """
    system_prompt = None
    user_prompt = ""

    # ── Monta o prompt conforme a técnica ──
    if nome_tecnica == "zero_shot":
        user_prompt = zero_shot(tarefa, input_texto)

    elif nome_tecnica == "few_shot":
        exemplos_tarefa = exemplos.get(tarefa["nome"], [])[:3]  # Máx 3 exemplos
        user_prompt = few_shot(tarefa, input_texto, exemplos_tarefa)

    elif nome_tecnica == "chain_of_thought":
        user_prompt = chain_of_thought(tarefa, input_texto, tarefa["passos_cot"])

    elif nome_tecnica == "role_prompting":
        chave_persona = tarefa.get("persona", "analista_cx")
        system_prompt, user_prompt = role_prompting(tarefa, input_texto, persona)

    # ── Conta tokens do prompt (antes de enviar) ──
    tokens_prompt_local = evaluator.contar_tokens(user_prompt)
    if system_prompt:
        tokens_prompt_local += evaluator.contar_tokens(system_prompt)

    # ── Envia ao LLM ──
    resultado_llm = llm.chat(
        prompt=user_prompt,
        system=system_prompt,
        temperature=TEMPERATURA_PADRAO,
        max_tokens=MAX_TOKENS_PADRAO,
    )

    resposta = resultado_llm["resposta"]

    # ── Usa tokens da Ollama se disponíveis, senão usa tiktoken ──
    tokens_prompt_final = resultado_llm.get("tokens_prompt") or tokens_prompt_local
    tokens_resposta = resultado_llm.get("tokens_resposta") or evaluator.contar_tokens(resposta)
    tokens_total = tokens_prompt_final + tokens_resposta

    return {
        "prompt": user_prompt,
        "system": system_prompt,
        "resposta": resposta,
        "tokens_prompt": tokens_prompt_final,
        "tokens_resposta": tokens_resposta,
        "tokens_total": tokens_total,
        "tempo_ms": resultado_llm["tempo_ms"],
    }


def medir_consistencia_tecnica(
    nome_tecnica: str,
    tarefa: dict,
    input_texto: str,
    exemplos: list,
    persona: dict,
    llm: LLMClient,
) -> float:
    """
    Executa a mesma técnica N vezes no mesmo input para medir consistência.
    """
    respostas = []
    for _ in range(REPETICOES_CONSISTENCIA):
        try:
            resultado = executar_tecnica(
                nome_tecnica, tarefa, input_texto, exemplos, persona, llm
            )
            respostas.append(resultado["resposta"])
            time.sleep(0.3)
        except RuntimeError:
            break
    return evaluator.medir_consistencia(respostas)


def main():
    print("\n" + "═" * 60)
    print("  PROMPT TOOLKIT — Checkpoint 02 — FIAP")
    print("  Domínio: E-commerce | Análise de Reviews de Clientes")
    print("═" * 60 + "\n")

    # ── PASSO 1: Verifica conexão com Ollama ──
    print("🔌 Verificando conexão com Ollama...")
    llm = LLMClient()
    if not llm.verificar_conexao():
        print("\n❌ Não foi possível conectar ao Ollama. Encerrando.")
        print("   → Certifique-se que o Ollama está rodando: ollama serve")
        print(f"   → E que o modelo está disponível: ollama pull {llm.model}")
        sys.exit(1)

    # ── PASSO 2: Carrega dados ──
    print("\n📂 Carregando dados...")
    try:
        inputs_json = carregar_json(DATA_DIR / "inputs.json")
        exemplos_json = carregar_json(DATA_DIR / "examples.json")
        system_prompts = carregar_json(PROMPTS_DIR / "system_prompts.json")
        print(f"  ✓  {len(inputs_json)} conjuntos de inputs carregados")
        print(f"  ✓  Exemplos few-shot carregados")
        print(f"  ✓  {len(system_prompts)} personas carregadas")
    except FileNotFoundError as e:
        print(f"  ✗  Arquivo não encontrado: {e}")
        sys.exit(1)

    # ── PASSO 3: Loop principal ──
    TECNICAS = ["zero_shot", "few_shot", "chain_of_thought", "role_prompting"]
    LABELS = {
        "zero_shot": "Zero-Shot",
        "few_shot": "Few-Shot",
        "chain_of_thought": "Chain-of-Thought",
        "role_prompting": "Role Prompting",
    }

    todos_resultados = []

    for tarefa in TODAS_AS_TAREFAS:
        nome_tarefa = tarefa["nome"]
        inputs_tarefa = inputs_json.get(nome_tarefa, [])

        print(f"\n{'─'*60}")
        print(f"  📋 TAREFA: {nome_tarefa.upper()} ({tarefa['tipo']})")
        print(f"  {len(inputs_tarefa)} inputs × {len(TECNICAS)} técnicas")
        print(f"{'─'*60}")

        # Persona da tarefa
        chave_persona = tarefa.get("persona", "analista_cx")
        persona = system_prompts.get(chave_persona, {})

        for i, item in enumerate(inputs_tarefa, 1):
            input_texto = item["input"]
            esperado = item["esperado"]

            print(f"\n  Input {i}/{len(inputs_tarefa)}: \"{input_texto[:60]}...\"" if len(input_texto) > 60 else f"\n  Input {i}/{len(inputs_tarefa)}: \"{input_texto}\"")

            for tecnica in TECNICAS:
                print(f"    [{LABELS[tecnica]}] ", end="", flush=True)

                try:
                    resultado = executar_tecnica(
                        tecnica, tarefa, input_texto,
                        exemplos_json, persona, llm
                    )

                    # Mede acurácia
                    acuracia = evaluator.medir_acuracia(resultado["resposta"], esperado)

                    # Mede consistência (executa mais vezes no input 1 para economizar tempo)
                    if i == 1:
                        consistencia = medir_consistencia_tecnica(
                            tecnica, tarefa, input_texto, exemplos_json, persona, llm
                        )
                    else:
                        consistencia = None  # Será preenchida por interpolação no relatório

                    print(
                        f"✓ | Resposta: \"{resultado['resposta'][:30]}\" | "
                        f"Acurácia: {acuracia:.0%} | "
                        f"Tokens: {resultado['tokens_total']} | "
                        f"Tempo: {resultado['tempo_ms']}ms"
                    )

                    todos_resultados.append({
                        "tarefa": nome_tarefa,
                        "tecnica": tecnica,
                        "input_idx": i,
                        "input": input_texto,
                        "esperado": str(esperado),
                        "resposta": resultado["resposta"],
                        "acuracia": acuracia,
                        "tokens_prompt": resultado["tokens_prompt"],
                        "tokens_resposta": resultado["tokens_resposta"],
                        "tokens_total": resultado["tokens_total"],
                        "tempo_ms": resultado["tempo_ms"],
                        "consistencia": consistencia if consistencia is not None else 0.0,
                    })

                except RuntimeError as e:
                    print(f"✗ | Erro: {str(e)[:80]}")
                    todos_resultados.append({
                        "tarefa": nome_tarefa,
                        "tecnica": tecnica,
                        "input_idx": i,
                        "input": input_texto,
                        "esperado": str(esperado),
                        "resposta": "ERRO",
                        "acuracia": 0.0,
                        "tokens_prompt": 0,
                        "tokens_resposta": 0,
                        "tokens_total": 0,
                        "tempo_ms": 0,
                        "consistencia": 0.0,
                    })

    # ── PASSO 4: Gera relatório ──
    print("\n\n📊 Gerando relatório...")
    df = report.gerar_tabela(todos_resultados)
    report.grafico_acuracia(df)
    report.grafico_custo(df)

    # ── PASSO 5: Teste de temperatura ──
    print("\n🌡  Executando teste de temperatura...")

    # Usa a primeira tarefa e o primeiro input como referência
    tarefa_ref = TODAS_AS_TAREFAS[0]
    input_ref = inputs_json.get(tarefa_ref["nome"], [{}])[0].get("input", "")
    prompt_ref = zero_shot(tarefa_ref, input_ref)

    resultados_temperatura = evaluator.testar_temperatura(
        llm_client=llm,
        prompt=prompt_ref,
        temperaturas=[0.1, 0.5, 1.0],
        repeticoes=3,
    )
    report.grafico_temperatura(resultados_temperatura)

    # ── PASSO 6: Recomendações ──
    recomendacoes = report.recomendar(df)

    # ── Resumo final ──
    print("\n✅ EXECUÇÃO CONCLUÍDA")
    print(f"   Total de execuções: {len(todos_resultados)}")
    print(f"   Resultados: output/resultados.csv")
    print(f"   Gráficos:   output/graficos/")
    print()


if __name__ == "__main__":
    main()
