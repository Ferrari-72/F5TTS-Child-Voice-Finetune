import json, sys
nb = json.load(open('tts_finetuning_main.ipynb','r',encoding='utf-8'))
print(f'Total cells: {len(nb["cells"])}')
for i, cell in enumerate(nb["cells"]):
    src_preview = ''.join(cell["source"])[:60].replace('\n',' ').encode('ascii','replace').decode()
    print(f'  Cell {i:02d} [{cell["cell_type"]:8}]: {src_preview}')
