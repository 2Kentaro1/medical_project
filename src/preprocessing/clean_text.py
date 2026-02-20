import re
import html

def clean_text(text: str) -> str:
    if text is None or str(text).strip() == "":
        return "[NO_TEXT]"

    # HTMLデコード
    text = html.unescape(text)

    # 文字化けの代表的なものだけ除去
    text = text.replace("�", " ")

    # 小文字化（英語部分に有効）
    text = text.lower()

    # 英数字・日本語・一部記号を残す
    # 診断精度研究で重要な記号（%, <, >, -, .）は残す
    text = re.sub(r"[^0-9a-zA-Zぁ-んァ-ン一-龥%<>=\-\.\s]", " ", text)

    # 余分なスペース削除
    text = re.sub(r"\s+", " ", text).strip()

    return text
