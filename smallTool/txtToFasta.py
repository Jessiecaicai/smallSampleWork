
## txt2fasta
with open('50.txt') as txt:
    txtData = txt.readlines()
    #print(txtData)
fastaFile = open('50.fasta','w')

countInt = 0
for sequence in txtData:
    countInt = countInt+1
    countString = str(countInt)
    fastaFile.write('>'+countString)
    fastaFile.write('\n'+sequence)
    #print(sequence)

fastaFile.close()
