import sqlite3
import pandas as pd
from pathlib import Path

VARIANT = 12
DB_FILE = f"lab8_variant{VARIANT:02d}.db"
CSV_A = f"variant{VARIANT:02d}_A.csv"  # факти (ops)
CSV_B = f"variant{VARIANT:02d}_B.csv"  # довідник (accounts)

TABLE_A = "ops"
TABLE_B = "accounts"

# SQL створення таблиць
CREATE_B = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id INTEGER PRIMARY KEY,
    owner TEXT,
    currency TEXT
);
"""

CREATE_A = """
CREATE TABLE IF NOT EXISTS ops (
    op_id INTEGER PRIMARY KEY,
    account_id INTEGER,
    amount REAL,
    type TEXT,
    FOREIGN KEY(account_id) REFERENCES accounts(account_id)
);
"""

# 5 SQL-запитів
QUERIES = {
    "large_ops": "SELECT * FROM ops WHERE amount > 4000;",
    "withdrawals": "SELECT * FROM ops WHERE type = 'Withdrawal';",
    "join_accounts": """
        SELECT o.op_id, a.owner, a.currency, o.amount, o.type
        FROM ops o
        JOIN accounts a ON o.account_id = a.account_id;
    """,
    "summary_by_currency_type": """
        SELECT a.currency, o.type, SUM(o.amount) as total_amount, COUNT(*) as count_ops
        FROM ops o
        JOIN accounts a ON o.account_id = a.account_id
        GROUP BY a.currency, o.type;
    """,
    "top_ops": "SELECT * FROM ops ORDER BY amount DESC LIMIT 10;",
}

# Які запити зберігати в CSV через Pandas
EXPORT_QUERIES = ["join_accounts", "summary_by_currency_type"]


def run_sql(conn, sql, params=None):
    cur = conn.cursor()
    cur.execute(sql, params or ())
    conn.commit()


def import_csv_to_table(conn, csv_file, table_name):
    df = pd.read_csv(csv_file)
    df.to_sql(table_name, conn, if_exists="append", index=False)


def table_count(conn, table_name):
    return pd.read_sql_query(f"SELECT COUNT(*) AS cnt FROM {table_name};", conn)[
        "cnt"
    ].iloc[0]


def main():
    # 1) створення БД та таблиць
    if Path(DB_FILE).exists():
        Path(DB_FILE).unlink()

    conn = sqlite3.connect(DB_FILE)
    run_sql(conn, CREATE_B)
    run_sql(conn, CREATE_A)

    # 2) імпорт CSV у таблиці
    import_csv_to_table(conn, CSV_B, TABLE_B)
    import_csv_to_table(conn, CSV_A, TABLE_A)

    cntA = table_count(conn, TABLE_A)
    cntB = table_count(conn, TABLE_B)

    # 3) виконання запитів + збереження деяких у csv
    lines = []
    lines.append(f"Лабораторна робота №8 — Звіт (Варіант {VARIANT})")
    lines.append(f"База даних: {DB_FILE}")
    lines.append(f"Таблиця {TABLE_A}: {cntA} рядків")
    lines.append(f"Таблиця {TABLE_B}: {cntB} рядків")
    lines.append("")

    for name, sql in QUERIES.items():
        df_res = pd.read_sql_query(sql, conn)
        lines.append(f"== {name} ==")
        lines.append(
            f"SQL: {sql.strip()[:150]}{'...' if len(sql.strip()) > 150 else ''}"
        )
        lines.append(f"Rows: {len(df_res)}")
        lines.append(df_res.head(5).to_string(index=False))
        lines.append("")

        if name in EXPORT_QUERIES:
            out_csv = f"{name}_result.csv"
            df_res.to_csv(out_csv, index=False)

    # 4) підсумковий звіт
    lines.append("== Висновки ==")
    lines.append(
        "1. Була створена база даних SQLite з двома таблицями: 'accounts' та 'ops'."
    )
    lines.append("2. Дані були успішно завантажені з CSV файлів.")
    lines.append(
        "3. Виконано 5 SQL запитів, включаючи фільтрацію, об'єднання та групування."
    )
    lines.append(
        "4. Результати аналізу (сумарні суми по валютах та типах операцій) експортовані у CSV."
    )

    Path(f"report_variant{VARIANT:02d}.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    conn.close()

    print(f"Saved: {DB_FILE}, report_variant{VARIANT:02d}.txt")
    for q in EXPORT_QUERIES:
        print(f"CSV: {q}_result.csv")


if __name__ == "__main__":
    main()
