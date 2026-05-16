"""
report.py — Módulo de geração de relatórios e visualizações
Referência: Checkpoint 02 — Análise e visualização (peso 20%)

Entregáveis:
- CSV com todos os resultados (output/resultados.csv)
- Gráfico de acurácia por técnica (output/graficos/acuracia.png)
- Gráfico de custo em tokens por técnica (output/graficos/custo_tokens.png)
- Gráfico de consistência por temperatura (output/graficos/temperatura.png)
- Recomendação automática da melhor técnica por tarefa (console)
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Usa backend não-interativo para salvar sem exibir janela
matplotlib.use("Agg")

# Configuração visual padronizada
CORES_TECNICAS = {
    "zero_shot": "#4C72B0",
    "few_shot": "#55A868",
    "chain_of_thought": "#C44E52",
    "role_prompting": "#8172B2",
}
LABELS_TECNICAS = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain-of-Thought",
    "role_prompting": "Role Prompting",
}

OUTPUT_DIR = Path("output")
GRAFICOS_DIR = OUTPUT_DIR / "graficos"


def _garantir_diretorios():
    """Cria os diretórios de output se não existirem."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    GRAFICOS_DIR.mkdir(exist_ok=True)


def gerar_tabela(resultados: list[dict]) -> pd.DataFrame:
    """
    Gera o DataFrame consolidado de resultados e salva como CSV.

    Cada linha representa uma execução: tarefa × técnica × input.

    Args:
        resultados: Lista de dicts com métricas de cada execução

    Returns:
        DataFrame com todos os resultados
    """
    _garantir_diretorios()

    df = pd.DataFrame(resultados)

    # Ordenação para facilitar leitura
    if not df.empty:
        colunas_ordem = [
            "tarefa", "tecnica", "input_idx",
            "acuracia", "tokens_prompt", "tokens_resposta", "tokens_total",
            "tempo_ms", "consistencia", "resposta",
        ]
        colunas_existentes = [c for c in colunas_ordem if c in df.columns]
        colunas_extra = [c for c in df.columns if c not in colunas_existentes]
        df = df[colunas_existentes + colunas_extra]

    csv_path = OUTPUT_DIR / "resultados.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  ✓  Tabela salva em: {csv_path}")

    return df


def grafico_acuracia(df: pd.DataFrame):
    """
    Gráfico de barras agrupadas: acurácia média por técnica e tarefa.

    Mostra qual técnica obteve melhor acurácia em cada tipo de tarefa.
    """
    _garantir_diretorios()

    if df.empty or "acuracia" not in df.columns:
        print("  ⚠  Sem dados de acurácia para gerar gráfico.")
        return

    # Agrega: média de acurácia por tarefa × técnica
    resumo = (
        df.groupby(["tarefa", "tecnica"])["acuracia"]
        .mean()
        .reset_index()
        .rename(columns={"acuracia": "acuracia_media"})
    )

    tarefas = resumo["tarefa"].unique()
    tecnicas = resumo["tecnica"].unique()
    n_tarefas = len(tarefas)
    n_tecnicas = len(tecnicas)

    fig, ax = plt.subplots(figsize=(10, 6))

    largura = 0.8 / n_tecnicas
    for i, tecnica in enumerate(tecnicas):
        dados = resumo[resumo["tecnica"] == tecnica]
        posicoes = []
        valores = []
        for j, tarefa in enumerate(tarefas):
            linha = dados[dados["tarefa"] == tarefa]
            posicoes.append(j + i * largura - (n_tecnicas - 1) * largura / 2)
            valores.append(float(linha["acuracia_media"].iloc[0]) if not linha.empty else 0.0)

        cor = CORES_TECNICAS.get(tecnica, "#888888")
        label = LABELS_TECNICAS.get(tecnica, tecnica)
        barras = ax.bar(posicoes, valores, width=largura * 0.9, color=cor, label=label, alpha=0.85)

        # Rótulos de valor em cada barra
        for barra, val in zip(barras, valores):
            if val > 0:
                ax.text(
                    barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 0.01,
                    f"{val:.0%}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    # Formatação
    ax.set_xlabel("Tarefa", fontsize=12)
    ax.set_ylabel("Acurácia Média", fontsize=12)
    ax.set_title("Acurácia por Técnica e Tarefa", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(range(n_tarefas))
    ax.set_xticklabels([t.replace("_", "\n") for t in tarefas], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    caminho = GRAFICOS_DIR / "acuracia.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Gráfico de acurácia salvo em: {caminho}")


def grafico_custo(df: pd.DataFrame):
    """
    Gráfico de barras: tokens médios consumidos por técnica.

    Tokens = custo direto em produção (billing e latência).
    Permite identificar técnicas mais econômicas.
    """
    _garantir_diretorios()

    if df.empty or "tokens_total" not in df.columns:
        print("  ⚠  Sem dados de tokens para gerar gráfico de custo.")
        return

    resumo = (
        df.groupby("tecnica")["tokens_total"]
        .mean()
        .reset_index()
        .rename(columns={"tokens_total": "tokens_medio"})
        .sort_values("tokens_medio", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    cores = [CORES_TECNICAS.get(t, "#888888") for t in resumo["tecnica"]]
    labels = [LABELS_TECNICAS.get(t, t) for t in resumo["tecnica"]]

    barras = ax.barh(labels, resumo["tokens_medio"], color=cores, alpha=0.85, height=0.5)

    # Rótulos de valor
    for barra, val in zip(barras, resumo["tokens_medio"]):
        ax.text(
            barra.get_width() + 2,
            barra.get_y() + barra.get_height() / 2,
            f"{val:.0f} tokens",
            va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Tokens Médios por Execução", fontsize=12)
    ax.set_title("Custo em Tokens por Técnica", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, resumo["tokens_medio"].max() * 1.2)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Nota explicativa
    ax.text(
        0.99, 0.02,
        "⚡ Menos tokens = menor custo e latência",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#555555", style="italic",
    )

    plt.tight_layout()
    caminho = GRAFICOS_DIR / "custo_tokens.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Gráfico de custo salvo em: {caminho}")


def grafico_temperatura(resultados_temperatura: list[dict]):
    """
    Gráfico de linha: consistência por temperatura (0.1, 0.5, 1.0).

    Mostra o trade-off entre determinismo (baixa temp) e criatividade (alta temp).

    Args:
        resultados_temperatura: Lista retornada por evaluator.testar_temperatura()
    """
    _garantir_diretorios()

    if not resultados_temperatura:
        print("  ⚠  Sem dados de temperatura para gerar gráfico.")
        return

    temps = [r["temperatura"] for r in resultados_temperatura]
    consistencias = [r["consistencia"] for r in resultados_temperatura]
    tokens = [r["tokens_medio"] for r in resultados_temperatura]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Eixo 1: Consistência (linha azul)
    cor_consist = "#4C72B0"
    ax1.plot(temps, consistencias, marker="o", color=cor_consist, linewidth=2.5,
             markersize=9, label="Consistência", zorder=5)
    ax1.fill_between(temps, consistencias, alpha=0.1, color=cor_consist)
    ax1.set_xlabel("Temperatura", fontsize=12)
    ax1.set_ylabel("Consistência", fontsize=12, color=cor_consist)
    ax1.tick_params(axis="y", labelcolor=cor_consist)
    ax1.set_ylim(0, 1.1)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

    # Rótulos de consistência
    for temp, cons in zip(temps, consistencias):
        ax1.annotate(f"{cons:.0%}", (temp, cons), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=cor_consist, fontweight="bold")

    # Eixo 2: Tokens médios (barra laranja)
    ax2 = ax1.twinx()
    cor_tokens = "#DD8452"
    ax2.bar(temps, tokens, width=0.08, alpha=0.4, color=cor_tokens, label="Tokens médios", zorder=2)
    ax2.set_ylabel("Tokens Médios", fontsize=12, color=cor_tokens)
    ax2.tick_params(axis="y", labelcolor=cor_tokens)

    # Título e formatação
    ax1.set_title("Impacto da Temperatura na Consistência e Custo", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xticks(temps)
    ax1.set_xticklabels([f"temp={t}" for t in temps], fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines[["top"]].set_visible(False)

    # Legenda combinada
    patch1 = mpatches.Patch(color=cor_consist, label="Consistência")
    patch2 = mpatches.Patch(color=cor_tokens, alpha=0.5, label="Tokens médios")
    ax1.legend(handles=[patch1, patch2], loc="lower left", fontsize=9)

    plt.tight_layout()
    caminho = GRAFICOS_DIR / "temperatura.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Gráfico de temperatura salvo em: {caminho}")


def recomendar(df: pd.DataFrame) -> dict:
    """
    Gera recomendação automática da melhor técnica por tarefa.

    Critério de ranking (pesos):
    - Acurácia: 50% (resultado correto é o mais importante)
    - Consistência: 30% (previsibilidade em produção)
    - Eficiência de tokens: 20% (menor custo é melhor)

    Args:
        df: DataFrame com os resultados de todas as execuções

    Returns:
        Dict {tarefa: {tecnica_recomendada, justificativa, metricas}}
    """
    recomendacoes = {}

    if df.empty:
        return recomendacoes

    tarefas = df["tarefa"].unique()

    print("\n" + "═" * 60)
    print("  RECOMENDAÇÕES AUTOMÁTICAS POR TAREFA")
    print("═" * 60)

    for tarefa in tarefas:
        df_tarefa = df[df["tarefa"] == tarefa]

        resumo = df_tarefa.groupby("tecnica").agg(
            acuracia_media=("acuracia", "mean"),
            tokens_medio=("tokens_total", "mean"),
            consistencia_media=("consistencia", "mean"),
        ).reset_index()

        if resumo.empty:
            continue

        # Normaliza tokens (inverte: menos tokens = melhor)
        max_tokens = resumo["tokens_medio"].max()
        if max_tokens > 0:
            resumo["eficiencia"] = 1 - (resumo["tokens_medio"] / max_tokens)
        else:
            resumo["eficiencia"] = 1.0

        # Score ponderado
        resumo["score"] = (
            resumo["acuracia_media"] * 0.50
            + resumo["consistencia_media"] * 0.30
            + resumo["eficiencia"] * 0.20
        )

        melhor = resumo.loc[resumo["score"].idxmax()]
        tecnica = melhor["tecnica"]
        label = LABELS_TECNICAS.get(tecnica, tecnica)

        justificativa = (
            f"Acurácia: {melhor['acuracia_media']:.0%} | "
            f"Consistência: {melhor['consistencia_media']:.0%} | "
            f"Tokens: {melhor['tokens_medio']:.0f} | "
            f"Score: {melhor['score']:.2f}"
        )

        recomendacoes[tarefa] = {
            "tecnica_recomendada": tecnica,
            "label": label,
            "justificativa": justificativa,
            "metricas": melhor.to_dict(),
        }

        print(f"\n  📌 Tarefa: {tarefa.upper()}")
        print(f"     ✅ Melhor técnica: {label}")
        print(f"     📊 {justificativa}")

        # Tabela comparativa de todas as técnicas para essa tarefa
        print("\n     Comparativo completo:")
        for _, linha in resumo.sort_values("score", ascending=False).iterrows():
            marcador = "⭐" if linha["tecnica"] == tecnica else "  "
            lbl = LABELS_TECNICAS.get(linha["tecnica"], linha["tecnica"])
            print(
                f"     {marcador} {lbl:<20} | "
                f"Acurácia: {linha['acuracia_media']:.0%} | "
                f"Tokens: {linha['tokens_medio']:.0f} | "
                f"Score: {linha['score']:.2f}"
            )

    print("\n" + "═" * 60)

    return recomendacoes
