import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Nome da aba:
st.set_page_config(
    page_title="House Prices - CATI Jr.",
    page_icon="🏠",
    layout="wide"
)

# início da página
st.title("House Prices")
st.caption("Projeto Trainee de Dados • CATI Jr.")

st.markdown("""
## Previsão de Preços de Imóveis

Este projeto utiliza técnicas de **Machine Learning** para estimar o valor de venda de imóveis do dataset **House Prices**.

Durante o desenvolvimento foram realizadas as seguintes etapas:

- Análise Exploratória dos Dados (EDA)
- Limpeza de dados
- Engenharia de Features
- Comparação entre modelos supervisionados de Machine Learning
- Aprendizado Não Supervisionado (K-Means)

O **Random Forest** apresentou o melhor desempenho e foi utilizado neste simulador.
""")

st.divider()

st.subheader("Variáveis mais importantes: ")

st.markdown("""
1. **OverallQual** – Qualidade geral do imóvel

2. **AreaInterna** – Área interna total

3. **LotArea** – Área do terreno

4. **TotalBanheiros - Número total de banheiros**

5. **GarageCars - Número de automóveis que cabem na garagem**

6. **LotFrontage - Largura da frente da casa**

7. **IdadeCasa**

8. **GarageArea - Área da garagem**

9. **IdadeReforma - Tempo desde a última reforma**
""")

st.subheader("Perfis de residências encontradas pelo K-Means")

st.markdown("""
Foram identificadas três perfis principais de imóveis:
            
**Cluster 0**
- Imóveis de alto padrão.
- Maior qualidade de construção.
- Maior área interna.
- Mais banheiros.
- Garagens maiores.
- Casas mais novas e recentemente reformadas.

**Cluster 1**
- Imóveis de padrão intermediário.
- Características medianas.

**Cluster 2**
- Imóveis mais simples.
- Menor área.
- Construções mais antigas e com reformas mais antigas.
- Menos banheiros.
- Lotes menores.
- Menos espaço na garagem para automóveis.
""")

# funções importantes do modelo
def tratamento_nulos(df):

    df = df.fillna({
        'PoolQC': 'None',
        'MiscFeature': 'None',
        'Alley': 'None',
        'Fence': 'None',
        'FireplaceQu': 'None',
        'GarageType': 'None',
        'GarageFinish': 'None',
        'GarageQual': 'None',
        'GarageCond': 'None',
        'BsmtExposure': 'None',
        'BsmtFinType2': 'None',
        'BsmtQual': 'None',
        'BsmtFinType1': 'None',
        'BsmtCond': 'None'
    })

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(
            df['Electrical'].mode()[0]
        )

    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['LotFrontage'] = df['LotFrontage'].fillna(
        df['LotFrontage'].median()
    )

    return df

def ajustes_finais(df):
    df["IdadeCasa"] = df['YrSold'] - df['YearBuilt'] # Ano de venda - Ano de construção.
    # Criei essa variável para verificar se a idade da casa influencia nos preços.
    # Casas mais novas são mais caras?

    df["AreaInterna"] = (df['1stFlrSF'] +
    df['2ndFlrSF'] +
    df['TotalBsmtSF']) # Soma das áreas dos andares + porão.
    # Optei por criar essa variável para saber a influência
    # da área interna total da casa em relação aos preços.
    # Casas maiores são mais caras? ou tem outras variáveis melhores?

    df['AreaExterna'] = (df["WoodDeckSF"] + # Soma das áreas externas
    df["OpenPorchSF"] +
    df["EnclosedPorch"] +
    df["3SsnPorch"] +
    df["ScreenPorch"])
    # A área externa da casa influencia nos preços?

    df["IdadeReforma"] = df["YrSold"] - df["YearRemodAdd"] # Tempo desde a última reforma.
    # Optei por criar essa para verificar se casas recentemente reformadas são mais caras.

    df["TotalBanheiros"] = (
    df["FullBath"] +
    0.5 * df["HalfBath"] +
    df["BsmtFullBath"] +
    0.5 * df["BsmtHalfBath"]
    )
    # Casas com mais banheiros são mais caras?

    df = df.drop('Id', axis=1) # exclusão de coluna Id, que não vai ser relevante para a análise.

    return df

@st.cache_resource  # Carrega o modelo apenas uma vez para evitar
# treinar a Random Forest sempre que a página atualizar.
def carregar_e_treinar():

    treino = pd.read_csv("train.csv")

    # Tratamento dos valores ausentes
    treino = tratamento_nulos(treino)

    # Engenharia de features
    treino = ajustes_finais(treino)

    features = [
        "OverallQual",
        "AreaInterna",
        "LotArea",
        "TotalBanheiros",
        "GarageCars",
        "LotFrontage",
        "IdadeCasa",
        "GarageArea",
        "IdadeReforma"
    ]

    treino = treino.dropna(subset=features + ["SalePrice"])

    X = treino[features]
    y = treino["SalePrice"]

    modelo = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X, y)

    return modelo

modelo = carregar_e_treinar()


# Simulação dos preços
st.header("🏡 Realize uma simulação")

st.write("""
Preencha as características do imóvel abaixo para que o modelo
Random Forest estime o preço de venda.
""")

col1, col2 = st.columns(2)

with col1:

    qualidade = st.slider(
        "Qualidade Geral",
        1,
        10,
        5
    )

    area = st.number_input(
        "Área Interna (ft²)",
        min_value=300,
        value=1500,
        step=50
    )

    terreno = st.number_input(
        "Área do Terreno (ft²)",
        min_value=1300,
        value=9000,
        step=100
    )

    banheiros = st.slider(
        "Total de Banheiros",
        1,
        6,
        2
    )

with col2:

    garagem = st.slider(
        "Vagas na Garagem",
        0,
        5,
        2
    )

    frente = st.number_input(
        "Largura da Frente do Terreno",
        min_value=21,
        value=70
    )

    area_garagem = st.number_input(
        "Área da Garagem",
        min_value=0,
        value=500
    )

    idade = st.number_input(
        "Idade da Casa",
        min_value=0,
        value=20
    )

    idade_reforma = st.number_input(
        "Idade da Reforma",
        min_value=0,
        value=5
    )

st.divider()

if st.button(
    "💰 Prever preço",
    use_container_width=True,
    type="primary"
):

    entrada = pd.DataFrame({

        "OverallQual":[qualidade],
        "AreaInterna":[area],
        "LotArea":[terreno],
        "TotalBanheiros":[banheiros],
        "GarageCars":[garagem],
        "LotFrontage":[frente],
        "IdadeCasa":[idade],
        "GarageArea":[area_garagem],
        "IdadeReforma":[idade_reforma]

    })

    preco = modelo.predict(entrada)[0]

    st.success(f"""
## 🏠 Preço estimado

### **US$ {preco:,.2f}**
""")
    
st.divider()

st.caption("""
Projeto desenvolvido por **Nathália Cristina Cadeu**

Projeto Trainee de Dados • CATI Jr.
""")
