import csv
import sys
from pathlib import Path
from typing import Dict, List


class DataType:
    def __init__(self, values: Dict[str, str]):
        self.values = values

    def __str__(self) -> str:
        res = ""
        for key, value in self.values.items():
            res += f"{key}: {value}\n"
        return res

    def __getitem__(self, key):
        return self.values[key]


class Table:
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data
        if not self.data:
            self.shape = (0, 0)
            self.dtype = DataType({})
            return
        self.shape = (len(self.data[0]), len(self.data))
        self.dtype = DataType(
            {
                col: self.guess_column_type([row[col] for row in self.data])
                for col in self.data[0]
            }
        )

    def rows(self):
        return self.shape[0]

    def columns(self):
        return self.shape[1]

    def guess_column_type(self, values):
        """
        Guess a column type from a list of string values.
        Returns: "int", "float", or "string"
        Handles None/empty values by skipping them
        """
        saw_float = False
        saw_int = False

        for raw in values:
            if raw is None:
                continue  # ignore None values
            s = str(raw).strip()
            if s == "":
                continue  # ignore empty values

            # Try int first (no decimal point)
            try:
                int(s)
                saw_int = True
                continue
            except ValueError:
                pass

            # Try float
            try:
                float(s)
                saw_float = True
                continue
            except ValueError:
                return "string"

        if saw_float:
            return "float"
        if saw_int:
            return "int"
        return "string"

    def __str__(self) -> str:
        return self.cols()

    def col_names(self):
        if not self.data:
            return []
        return list(self.data[0].keys())

    def cols(self, columns=None, rows=5):
        if not self.data:
            return ""
        if not columns:
            columns = list(self.data[0].keys())
        # Use the keys from the first row as column headers (preserves order)
        headers = list(self.data[0].keys())
        illegal_columns = [column for column in columns if column not in headers]
        if illegal_columns:
            print(f"Warning: Columns {', '.join(illegal_columns)} not found in table")
        # Actual headers that will reach the table
        headers = [header for header in headers if header in columns]
        if len(headers) == 0:
            return ""

        if rows < 0:
            rows = len(self.data)
        trimmed = self.data[:rows]

        # Compute the maximum width for each column (header vs any value in that column)
        longest_column_widths = []
        for header in headers:
            max_width = len(header)
            for row in trimmed:
                val = row.get(header, "")
                max_width = max(max_width, len(str(val)))
            longest_column_widths.append(max_width)
        # Build header line
        res = ""
        for header, width in zip(headers, longest_column_widths):
            res += f"{header:<{width}} "
        res = res.rstrip() + "\n"
        # Build each data row
        for row in trimmed:
            for header, width in zip(headers, longest_column_widths):
                val = row.get(header, "")
                res += f"{val:<{width}} "
            res = res.rstrip() + "\n"
        return res

    def filter(self, condition, column: str):
        if not self.data:
            return Table([])
        return Table(
            [
                row
                for row in self.data
                if row.get(column, "")
                and row.get(column, "").strip()
                and condition(row[column])
            ]
        )

    def __valid_numeric_column__(self, column: str) -> List[float]:
        if not self.data:
            raise ValueError("Table is empty")
        if self.dtype[column] == "string":
            raise ValueError(f"Column '{column}' is not numeric")
        values = self._numeric_values(column)
        if not values:
            raise ValueError(f"Column '{column}' has no numeric values")
        return values

    def _numeric_values(self, column: str) -> List[float]:
        values = []
        for row in self.data:
            raw = row.get(column, "")
            if raw is None:
                continue
            s = str(raw).strip()
            if s == "":
                continue
            try:
                values.append(float(s))
            except ValueError as exc:
                raise ValueError(
                    f"Column '{column}' has non-numeric value: {raw}"
                ) from exc
        return values

    def max(self, column: str) -> float:
        return max(self.__valid_numeric_column__(column))

    def min(self, column: str) -> float:
        return min(self.__valid_numeric_column__(column))

    def count(self, column: str) -> int:
        if not self.data:
            raise ValueError("Table is empty")

        values = []
        for row in self.data:
            raw = row.get(column, "")
            if raw is None:
                continue
            s = str(raw).strip()
            if s == "":
                continue
            values.append(s)
        return len(values)

    def mean(self, column: str) -> float:
        return sum(self.__valid_numeric_column__(column)) / self.count(column)

    def median(self, column: str) -> float:
        values = self.__valid_numeric_column__(column)
        values.sort()
        n = len(values)
        if n % 2 == 1:
            return values[n // 2]
        else:
            return (values[n // 2 - 1] + values[n // 2]) / 2

    def popular(self, column: str, top=5):
        counter = {}
        for row in self.data:
            value = row.get(column, "")
            if value is None:
                continue
            value = str(value).strip()
            if value == "":
                continue
            counter[value] = counter.get(value, 0) + 1
        return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top]


def parse_csv(file_path) -> Table:
    with open(file_path, "r") as file:
        lines = file.readlines()
        index_name_tuple = []
        for field in lines[0].strip().split(","):
            index_name_tuple.append((len(index_name_tuple), field))

        raw = [line.strip().split(",") for line in lines[1:]]
        data = [{name: row[index] for index, name in index_name_tuple} for row in raw]
    return Table(data)


if __name__ == "__main__":
    data = parse_csv(sys.argv[1])
    print("======== Original data characteristics ========")
    print(data.shape)
    for col_name in data.col_names():
        filled_cells = data.count(col_name)
        print(
            f"Column '{col_name}' type '{data.dtype[col_name]}' (empty cells: {data.columns() - filled_cells})"
        )
    print()
    print(data.cols(columns=["subject", "points"]))
    print()
    print(data.popular("subject"))

    # Column for statistics "points"
    print("Statistics for 'points' column:")
    max_points = data.max("points")
    print("Maximum:", max_points)
    print("Minimum:", data.min("points"))
    print("Average:", data.mean("points"))
    print("Median:", data.median("points"))
    print("Valid values:", data.count("points"))

    print("======== Filtered output: ========")

    filtered_data = data.filter(lambda x: float(x) >= max_points * 0.7, "points")
    print(filtered_data.shape)
    print(filtered_data.cols(columns=["subject", "points"]))

    print(filtered_data.min("points"))

    # Report builder (Variant 12)
    VARIANT = 12
    rows = data.data
    filtered_rows = filtered_data.data
    cols = data.col_names()

    miss = {col: 0 for col in cols}
    for row in rows:
        for col in cols:
            value = row.get(col)
            if value is None or str(value).strip() == "":
                miss[col] += 1

    try:
        stats = {
            "count": str(filtered_data.count("points")),
            "min": str(filtered_data.min("points")),
            "max": str(filtered_data.max("points")),
            "mean": str(filtered_data.mean("points")),
            "median": str(filtered_data.median("points")),
        }
    except ValueError:
        stats = {
            "count": "0",
            "min": "None",
            "max": "None",
            "mean": "None",
            "median": "None",
        }
    top = filtered_data.popular("subject", top=5)

    out_csv = Path(__file__).parent / f"result_variant{VARIANT:02d}.csv"
    out_txt = Path(__file__).parent / f"report_variant{VARIANT:02d}.txt"

    with open(out_csv, "w", newline="", encoding="utf-8") as file:
        if filtered_rows:
            writer = csv.DictWriter(file, fieldnames=list(filtered_rows[0].keys()))
            writer.writeheader()
            writer.writerows(filtered_rows)

    report_lines = []
    report_lines.append(f"Лабораторна робота No1 — Звіт (Варіант {VARIANT})")
    report_lines.append(f"Вхідний файл: {sys.argv[1]}")
    report_lines.append(f"Кількість записів (всього): {len(rows)}")
    report_lines.append(f"Кількість записів (після фільтра): {len(filtered_rows)}")
    report_lines.append("")
    report_lines.append("Колонки:")
    report_lines.append(", ".join(cols))
    report_lines.append("")
    report_lines.append("Пропуски по колонках:")
    for c, m in miss.items():
        report_lines.append(f"- {c}: {m}")
    report_lines.append("")
    report_lines.append("Статистика числової колонки 'points' (після фільтра):")
    report_lines.append(f"- count: {stats['count']}")
    report_lines.append(f"- min: {stats['min']}")
    report_lines.append(f"- max: {stats['max']}")
    report_lines.append(f"- mean: {stats['mean']}")
    report_lines.append(f"- median: {stats['median']}")
    report_lines.append("")
    report_lines.append(
        "Топ-5 значень категоріальної колонки 'subject' (після фільтра):"
    )
    for i, (val, cnt) in enumerate(top, 1):
        report_lines.append(f"{i}) {val} — {cnt}")

    with open(out_txt, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print("Готово")
    print("Збережено:", out_csv, "та", out_txt)
