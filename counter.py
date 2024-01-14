file = open('eng_nl_texts_train.txt', 'r', encoding='utf-8')

en_count = 0
nl_count = 0

for line in file:
   if line.startswith('en'):
       en_count += 1
   elif line.startswith('nl'):
       nl_count += 1

print("Number of lines starting with 'en':", en_count)
print("Number of lines starting with 'nl':", nl_count)

file.close()
