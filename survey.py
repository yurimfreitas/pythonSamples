import pandas as pd

# Read the CSV file
df = pd.read_csv("C:\\Users\\yfreitas\Documents\\pythonSamples\\Survey.csv")

questao_media = input("Digite a questao que deseja verificar a media: ")

def media_questao_escolhida(questao_escolhida):
   df_filtrado = df[df['questao'] == questao_escolhida]
   return df_filtrado['nota'].sum()/len(df_filtrado)

def participant_max_overall():
   df_media = df.groupby('pessoa')['nota'].mean().reset_index().rename(columns={'nota': 'media_nota'})
   return df_media[df_media['media_nota'] == df_media['media_nota'].max()]

def percentual_questao(nota_questao):
   df_nota = df[df['nota'] == nota_questao]
   return (len(df_nota) * 100) / len(df) 


print("A media da questao escolhida 'e: ", round(media_questao_escolhida(questao_media),2))

print("A pessoa com maior nota nas questoes: \n", round(participant_max_overall(),2))

print("O percentual de resposta com nota 4: \n", round(percentual_questao(4),2))

print("O percentual de resposta com nota 5: \n", round(percentual_questao(5),2))