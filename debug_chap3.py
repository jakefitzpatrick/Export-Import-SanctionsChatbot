from PyPDF2 import PdfReader
reader = PdfReader("finalCopy_2026HTSRev4.pdf")
full = "\n".join([p.extract_text() or "" for p in reader.pages]).lower()
pos = full.find('chapter 3')
print('pos=', pos)
if pos != -1:
    print(full[pos:pos+400])
else:
    print("No 'chapter 3' found")
