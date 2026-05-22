"""
Extração de preços das principais ações da bolsa americana (últimos 60 dias)
Dependência: pip install yfinance

Saída:
    TICKER | DATA | PRECO_FECHAMENTO
    (uma linha por registro)
"""

import yfinance as yf
import datetime

TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta
    "TSLA",   # Tesla
    "AMD",    # AMD
    "BRK-B",  # Berkshire Hathaway
    "JPM",    # JPMorgan
    "V",      # Visa
    "JNJ",    # Johnson & Johnson
    "WMT",    # Walmart
    "NFLX",   # Netflix
    "ORCL",   # Oracle
]

END_DATE   = datetime.date.today()
START_DATE = END_DATE - datetime.timedelta(days=100)


COMPANY_NAMES = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corp.",
    "NVDA":  "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms",
    "TSLA":  "Tesla Inc.",
    "AMD":   "Advanced Micro Devices",
    "BRK-B": "Berkshire Hathaway",
    "JPM":   "JPMorgan Chase",
    "V":     "Visa Inc.",
    "JNJ":   "Johnson & Johnson",
    "WMT":   "Walmart Inc.",
    "NFLX":  "Netflix Inc.",
    "ORCL":  "Oracle Corp.",
}


def fetch_prices(tickers: list[str], start: datetime.date, end: datetime.date) -> dict[str, list[str]]:
    """Retorna dict {ticker: [valores de fechamento por data]}."""
    resultado = {}

    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(start=start, end=end)

            if data.empty:
                print(f"[aviso] sem dados para {ticker}")
                continue

            valores = []
            for date, row in data.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                close    = round(float(row["Close"]), 2)
                # valores.append(f"{date_str}: ${close}")
                valores.append(f"{close}")

            resultado[ticker] = valores

        except Exception as e:
            print(f"[erro] {ticker}: {e}")

    return resultado


def main():
    print(f"Buscando dados de {START_DATE} a {END_DATE}...\n")

    dados = fetch_prices(TICKERS, START_DATE, END_DATE)

    total = 0
    for ticker, valores in dados.items():
        nome = COMPANY_NAMES.get(ticker, ticker)
        print(f"{nome} ({ticker})")
        for v in valores:
            print(f" {v}")
        print()
        total += len(valores)

    print(f"Total de registros: {total}")


if __name__ == "__main__":
    main()