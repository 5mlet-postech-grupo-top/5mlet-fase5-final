#!/usr/bin/env python
"""
Gerador de tráfego para simular chamadas aleatórias aos endpoints da API.
Útil para popular o Prometheus com dados de monitoramento.

Uso:
    python scripts/generate_traffic.py [--num-requests 200] [--delay 0.1]
"""

import argparse
import random
import time

try:
    import requests
except ImportError:
    print("Erro: 'requests' não está instalado.")
    print("Instale com: pip install requests")
    exit(1)


def generate_example_payload():
    """Gera um payload de exemplo para /predict."""
    return {
        "IDADE": random.randint(10, 18),
        "INDE": random.random() * 10,
        "IEG": random.random() * 10,
        "IDA": random.random() * 10,
        "PONTO_VIRADA": random.randint(0, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gerador de tráfego para a API PEDE."
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=200,
        help="Número de requisições a gerar (padrão: 200)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Intervalo entre requisições em segundos (padrão: 0.1)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8000",
        help="Host da API (padrão: http://localhost:8000)",
    )

    args = parser.parse_args()

    base_url = args.host
    endpoints = ["/health", "/predict", "/drift"]
    examples = [generate_example_payload() for _ in range(5)]

    print(f"Iniciando gerador de tráfego...")
    print(f"  Host: {base_url}")
    print(f"  Requisições: {args.num_requests}")
    print(f"  Intervalo: {args.delay}s")
    print()

    success_count = 0
    error_count = 0

    for i in range(args.num_requests):
        ep = random.choice(endpoints)
        url = base_url + ep

        try:
            if ep == "/predict":
                payload = random.choice(examples)
                response = requests.post(url, json=payload, timeout=5)
            else:
                response = requests.get(url, timeout=5)

            if response.status_code == 200:
                success_count += 1
                status_str = "✓"
            else:
                error_count += 1
                status_str = f"✗ {response.status_code}"

            print(f"[{i+1:3d}/{args.num_requests}] {status_str} {ep}")

        except Exception as e:
            error_count += 1
            print(f"[{i+1:3d}/{args.num_requests}] ✗ Erro: {str(e)}")

        time.sleep(args.delay)

    print()
    print(f"Conclusão:")
    print(f"  Sucessos: {success_count}")
    print(f"  Erros: {error_count}")
    print()
    print(f"Acesse o Prometheus em http://localhost:9090 ou consulte /metrics")


if __name__ == "__main__":
    main()
