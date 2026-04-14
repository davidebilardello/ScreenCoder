import re
import sys

def fix_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Rimuovere le virgolette in escape all'inizio/fine degli attributi class
    # Esempio: class='\"relative\"' -> class="relative"
    content = re.sub(r"class='\\\"([^\\\"]*)\\\"\'", r'class="\1"', content)

    # 2. Rimuovere le virgolette in escape miste e doppie assegnazioni
    # Esempio: class='\"grid' gap-4\"="" -> class="grid gap-4"
    # Useremo un paio di passaggi di pulizia per i casi più complessi:
    content = content.replace("class='\\\"grid' gap-4\\\"=\"\" grid-cols-3=\"\"", 'class="grid gap-4 grid-cols-3"')
    content = content.replace("class='\\\"bg-gray-400'", 'class="bg-gray-400"')
    content = content.replace("class='\\\"absolute' p-2=\"\" right-0=\"\" text-white\\\"=\"\"", 'class="absolute p-2 right-0 text-white"')
    content = content.replace("class='\\\"absolute' left-0=\"\" p-2=\"\" text-white\\\"=\"\"", 'class="absolute left-0 p-2 text-white"')
    content = content.replace("class='\\\"relative\"'", 'class="relative"')
    content = content.replace("aspect-video\\\"=\"\"", 'class="aspect-video"')
    content = content.replace("bottom-0=\"\"", "")

    # Cleanup finale su eventuali doppi spazi
    content = re.sub(r'\s+', ' ', content)
    content = content.replace(' >', '>')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fix_html_file(sys.argv[1])
