import pandas as pd
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
               names = ["idade","classe_de_trabalho","fnlwgt","educação", "número_de_escolaridade", "estado_civil", "Ocupação", 
                        "relacionamento", "raça", "sexo", "ganho_de_capital", "perda_de_capital", "horas_por_semana", "país_natal", ">50K_ou_<=50K."])
df1 = df.select_dtypes(include=numerics)
df.head()
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nDemosntração do Banco de dados.\n')
print(df)
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nDescrição do bando de dados com: Média, Desvio Padrão, Maior Valor, Menos Valor.\n')
print(df.describe())
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nA médiana do banco de dados.\n')
print(df1.median())
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nA moda do banco de dados.\n')
print(df1.mode())
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nA aplicação do query.\n')
#print(df.query('idade > 28 & (estado_civil != "Divorced" | relacionamento == "Not-in-family"'))
print(df.query('classe_de_trabalho=="Private  "'))