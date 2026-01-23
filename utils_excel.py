from io import BytesIO
import pandas as pd

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]

        # auto width nhẹ
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col[:2000]:
                if cell.value is None:
                    continue
                max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 45)

    return output.getvalue()
