
import pandas as pd

df = pd.read_csv('/users/psen/preprocessed.csv', delimiter=',')
print(df.size)

with open('/users/psen/preprocessed_v1.csv',"w")  as f1:
	f1.write("id,text,class\n")
	for index in df.index:
		x=df["text"][index]
		x=x.replace('"',"")
		class_x=df["class"][index]
		st = x.split(" ")
		if len(st) >=10:
			f1.write(str(index))
			f1.write(',"')
			f1.write(x)
			f1.write('",')
			f1.write(str(class_x))
			f1.write("\n")
f1.close()
			
df = pd.read_csv('/users/psen/preprocessed_v1.csv', delimiter=',')
print(df.size)
