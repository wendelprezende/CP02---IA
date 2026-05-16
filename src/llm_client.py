"""
llm_client.py — Módulo de conexão com a Ollama API
Referência: Aula 05 — Prompt Engineering: Conceito, Anatomia e Boas Práticas

Responsabilidades:
- Encapsula toda comunicação com a Ollama REST API
- Suporta system prompts (para role prompting, Aula 07)
- Trata erros de rede com retry automático
- Retorna métricas de tokens e tempo para o evaluator
"""

import os
import time
import requests
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Cliente para a Ollama API local.

    A Ollama expõe uma API REST compatível com o formato de chat da OpenAI.
    Endpoint principal: POST http://localhost:11434/api/chat

    Parâmetros configuráveis via variáveis de ambiente (.env):
    - OLLAMA_HOST: URL base da API (padrão: http://localhost:11434)
    - OLLAMA_MODEL: Modelo a ser utilizado (padrão: gpt-oss:120b)
    - OLLAMA_TIMEOUT: Timeout em segundos (padrão: 120)
    - OLLAMA_MAX_RETRIES: Tentativas em caso de falha (padrão: 3)
    """

    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self.endpoint = f"{self.host}/api/chat"

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> dict:
        """
        Envia um prompt para o modelo e retorna a resposta com métricas.

        Args:
            prompt: Mensagem do usuário (role: user)
            system: System prompt opcional (role: system) — usado no role prompting
            temperature: Aleatoriedade da resposta (0.0 = determinístico, 2.0 = criativo)
            max_tokens: Limite de tokens na resposta gerada

        Returns:
            dict com chaves:
              - resposta (str): Texto gerado pelo modelo
              - tokens_prompt (int): Tokens consumidos no prompt
              - tokens_resposta (int): Tokens gerados na resposta
              - tempo_ms (int): Tempo total de chamada em milissegundos
              - modelo (str): Nome do modelo usado

        Raises:
            RuntimeError: Após esgotar todas as tentativas de retry
        """
        mensagens = []

        # Adiciona system prompt se fornecido (Aula 07 — Role Prompting)
        if system:
            mensagens.append({"role": "system", "content": system})

        mensagens.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": mensagens,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        ultimo_erro = None
        for tentativa in range(1, self.max_retries + 1):
            try:
                inicio = time.time()
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                tempo_ms = int((time.time() - inicio) * 1000)

                response.raise_for_status()
                dados = response.json()

                # Extrai texto da resposta
                resposta_texto = dados.get("message", {}).get("content", "").strip()

                # Extrai métricas de tokens da resposta da Ollama
                tokens_prompt = dados.get("prompt_eval_count", 0)
                tokens_resposta = dados.get("eval_count", 0)

                return {
                    "resposta": resposta_texto,
                    "tokens_prompt": tokens_prompt,
                    "tokens_resposta": tokens_resposta,
                    "tempo_ms": tempo_ms,
                    "modelo": self.model,
                }

            except requests.exceptions.ConnectionError:
                ultimo_erro = (
                    f"[Tentativa {tentativa}/{self.max_retries}] "
                    "Ollama não está rodando. Execute: ollama serve"
                )
                print(f"  ⚠  {ultimo_erro}")

            except requests.exceptions.Timeout:
                ultimo_erro = (
                    f"[Tentativa {tentativa}/{self.max_retries}] "
                    f"Timeout após {self.timeout}s. "
                    "Tente aumentar OLLAMA_TIMEOUT no .env"
                )
                print(f"  ⚠  {ultimo_erro}")

            except requests.exceptions.HTTPError as e:
                ultimo_erro = f"Erro HTTP {e.response.status_code}: {e.response.text[:200]}"
                print(f"  ✗  {ultimo_erro}")
                break  # Erros HTTP não se resolvem com retry

            except Exception as e:
                ultimo_erro = f"Erro inesperado: {type(e).__name__}: {e}"
                print(f"  ✗  {ultimo_erro}")
                break

            # Aguarda antes de tentar novamente (backoff exponencial leve)
            if tentativa < self.max_retries:
                time.sleep(2 ** tentativa)

        raise RuntimeError(f"Falha ao comunicar com Ollama após {self.max_retries} tentativas. Último erro: {ultimo_erro}")

    def verificar_conexao(self) -> bool:
        """
        Verifica se o servidor Ollama está acessível e o modelo disponível.

        Returns:
            True se conectado e modelo disponível, False caso contrário
        """
        try:
            # Verifica se o servidor está no ar
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()

            modelos = response.json().get("models", [])
            nomes = [m.get("name", "") for m in modelos]

            if self.model in nomes:
                print(f"  ✓  Ollama conectado | Modelo '{self.model}' disponível")
                return True
            else:
                print(
                    f"  ⚠  Ollama conectado, mas modelo '{self.model}' não encontrado.\n"
                    f"     Modelos disponíveis: {nomes}\n"
                    f"     Execute: ollama pull {self.model}"
                )
                return False

        except Exception as e:
            print(f"  ✗  Ollama inacessível em {self.host}: {e}")
            return False
