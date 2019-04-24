from datamuse import datamuse
import time
api = datamuse.Datamuse()
try:
    words = open("words_alpha.txt", encoding='utf-8')
    rhymes = open("rhymes.txt", 'a', encoding='utf-8')
    progress = open("progress.txt", 'r', encoding='utf-8')

    line_no = int(progress.readline())
    progress.close()
    line_begin = line_no
    print(line_no)
    for i, line in enumerate(words):
        if (i == line_no):
            if (line_no < line_begin + 75000):
                line = line.rstrip()
                try:
                    api_rhymes = api.words(rel_rhy=line, max=5)
                    [rhymes.write(line+';'+rh["word"]+'\n')
                        for rh in api_rhymes]
                    print(line)
                except:
                    print("json decode on " + line)
                
                if len(api_rhymes) < 5:
                    try:
                        near_rhymes = api.words(rel_nry=line, max=5)
                        [rhymes.write(line+';'+rh["word"]+'\n')
                            for rh in near_rhymes]
                        print(line)
                    except:
                        print("json decode on " + line)
                rhymes.flush()
                line_no = line_no+1
            else:
                break
finally:
    progress = open("progress.txt", 'w', encoding='utf-8')
    progress.write(str(line_no))
    words.close()
    rhymes.close()
    progress.close()
