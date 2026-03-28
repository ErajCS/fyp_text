import os

def fix_emojis(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace common emojis with ASCII
    rep = {
        "\u2705": "OK",
        "\U0001f680": "INFO", 
        "\u274c": "ERROR",
        "\u26a0\ufe0f": "WARNING",
        "\ud83d\udd0e": "SEARCH",
        "\ud83c\udf10": "LANG"
    }
    
    for emoji, text in rep.items():
        content = content.replace(emoji, text)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed emojis in {file_path}")

fix_emojis('c:/Users/smwaj/fyp_text/agri_ui/rag_demo.py')
fix_emojis('c:/Users/smwaj/fyp_text/agri_ui/app.py')
