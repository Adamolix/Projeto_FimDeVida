import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from supabase import create_client, Client

def formatar_telefone(numero):
    numeros = "".join(filter(str.isdigit, numero))

    if len(numeros) == 11:
        return f"({numeros[:2]}) {numeros[2:7]}-{numeros[7:]}"
    elif len(numeros) == 10:
        return f"({numeros[:2]}) {numeros[2:6]}-{numeros[6:]}"

    return numero


st.set_page_config(page_title="Questionario FimDeVida", layout="centered")
st.title("Demonstração do uso de Machine Learning para classificação de veículos quanto à probabilidade de fim de vida")
st.caption("Protótipo acadêmico — modelo treinado com dados simulados.")
st.info("Este sistema demonstra uma possível triagem para classifiação de veículos quanto a probabilidade de fim de vida.")

@st.cache_resource
def conectar_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SECRET_KEY"])

def salvar_no_supabase(registro: dict):
    return conectar_supabase().table("veiculos").insert(registro).execute()

def carregar_registros():
    resposta = conectar_supabase().table("veiculos").select("*").order("created_at", desc=True).execute()
    return resposta.data

@st.cache_resource
def treinar_modelo():
    rng = np.random.default_rng(42)
    n = 5000
    idade = rng.integers(1, 26, n)
    quilometragem = np.clip(rng.normal(12000 * idade, 40000, n), 5000, 420000).astype(int)
    manutencoes_ano = np.clip(rng.poisson(1 + idade / 8), 0, 10)
    custo_manutencao = np.clip(rng.normal(800 + idade * 180 + manutencoes_ano * 450, 1800, n), 0, 20000).astype(int)
    acidente_grave = rng.binomial(1, np.clip(0.03 + idade * 0.008, 0, 0.35))
    falhas_frequentes = rng.binomial(1, np.clip(0.04 + idade * 0.02, 0, 0.65))
    veiculo_parado = rng.binomial(1, np.clip(0.01 + idade * 0.006 + falhas_frequentes * 0.10, 0, 0.45))
    valor_estimado = np.clip(90000 - idade * 3300 + rng.normal(0, 9000, n), 3000, 120000).astype(int)
    relacao_reparo_valor = custo_manutencao / np.maximum(valor_estimado, 1)
    score_oculto = (
        idade * 0.10 + (quilometragem / 100000) * 0.8 + manutencoes_ano * 0.25
        + falhas_frequentes * 1.8 + acidente_grave * 1.4 + veiculo_parado * 2.5
        + relacao_reparo_valor * 3.0 + rng.normal(0, 0.8, n)
    )
    fim_de_vida = (score_oculto > 5.5).astype(int)
    dados = pd.DataFrame({
        "idade": idade,
        "quilometragem": quilometragem,
        "manutencoes_ano": manutencoes_ano,
        "custo_manutencao_12m": custo_manutencao,
        "acidente_grave": acidente_grave,
        "falhas_frequentes": falhas_frequentes,
        "veiculo_parado": veiculo_parado,
        "valor_estimado": valor_estimado,
        "fim_de_vida": fim_de_vida,
    })
    X = dados.drop(columns=["fim_de_vida"])
    y = dados["fim_de_vida"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    modelo = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    modelo.fit(X_treino, y_treino)
    acuracia = accuracy_score(y_teste, modelo.predict(X_teste))
    return modelo, acuracia

def classificar(probabilidade):
    if probabilidade < 0.30:
        return "Baixo"
    if probabilidade < 0.60:
        return "Moderado"
    if probabilidade < 0.80:
        return "Alto"
    return "Muito alto"

def calcular_pontos(idade, quilometragem, manutencoes_ano, custo_manutencao, valor_estimado, acidente_grave, falhas_frequentes, veiculo_parado):
    pontos = 0
    if idade >= 15:
        pontos += 15
    elif idade >= 10:
        pontos += 8
    if quilometragem >= 200000:
        pontos += 20
    elif quilometragem >= 150000:
        pontos += 10
    if manutencoes_ano >= 5:
        pontos += 15
    elif manutencoes_ano >= 3:
        pontos += 8
    if valor_estimado > 0:
        relacao = custo_manutencao / valor_estimado
        if relacao >= 0.50:
            pontos += 25
        elif relacao >= 0.25:
            pontos += 12
    if acidente_grave:
        pontos += 10
    if falhas_frequentes:
        pontos += 15
    if veiculo_parado:
        pontos += 25
    return min(pontos, 100)

modelo_ml, acuracia_demo = treinar_modelo()

st.subheader("Questionário")
with st.form("formulario_veiculo", clear_on_submit=False):
    st.markdown("### Dados do proprietário")
    col1, col2 = st.columns(2)
    with col1:
        nome = st.text_input("Nome do proprietário *")
        telefone = st.text_input(
            "Telefone *",
            placeholder="(41) 99999-9999"
        )
    with col2:
        email = st.text_input("E-mail *")
        cidade = st.text_input("Cidade / Estado")

    st.markdown("### Dados do veículo")
    col1, col2 = st.columns(2)
    ano_atual = datetime.now().year
    with col1:
        modelo_veiculo = st.text_input("Modelo do veículo *", placeholder="Ex.: Sandero")
        ano_veiculo = st.number_input("Ano do veículo", min_value=1980, max_value=ano_atual, value=min(2012, ano_atual), step=1)
        quilometragem = st.number_input("Quilometragem atual (km)", min_value=0, max_value=1000000, value=150000, step=1000)
    with col2:
        manutencoes_ano = st.number_input("Manutenções / reparos nos últimos 12 meses", min_value=0, max_value=30, value=2, step=1)
        custo_manutencao = st.number_input("Custo de manutenção nos últimos 12 meses (R$)", min_value=0.0, max_value=200000.0, value=2500.0, step=100.0)
        valor_estimado = st.number_input("Valor aproximado atual do veículo (R$)", min_value=1.0, max_value=1000000.0, value=25000.0, step=500.0)

    st.markdown("### Condição atual")
    acidente_grave = st.checkbox("O veículo já sofreu acidente grave ou dano estrutural?")
    falhas_frequentes = st.checkbox("O veículo apresenta falhas mecânicas ou elétricas frequentes?")
    veiculo_parado = st.checkbox("O veículo está parado ou sem condições normais de circulação?")
    data_ultima_manutencao = st.date_input("Data da última manutenção")
    causa_ultima_manutencao = st.text_area("Causa da última manutenção", placeholder="Ex.: troca da embreagem, problema de arrefecimento...")
    consentimento = st.checkbox("Concordo que os dados informados sejam registrados neste protótipo acadêmico.")
    enviar = st.form_submit_button("Analisar e registrar veículo", type="primary", use_container_width=True)

if enviar:
    if not all([nome.strip(), telefone.strip(), email.strip(), modelo_veiculo.strip()]):
        st.error("Preencha nome, telefone, e-mail e modelo do veículo.")
    elif not consentimento:
        st.error("É necessário marcar o consentimento para registrar os dados.")
    elif len("".join(filter(str.isdigit, telefone))) not in [10, 11]:
        st.error("Digite um telefone válido com DDD, contendo 10 ou 11 dígitos.")
    else:
        idade = max(0, ano_atual - int(ano_veiculo))
        entrada = pd.DataFrame([{
            "idade": idade,
            "quilometragem": int(quilometragem),
            "manutencoes_ano": int(manutencoes_ano),
            "custo_manutencao_12m": float(custo_manutencao),
            "acidente_grave": int(acidente_grave),
            "falhas_frequentes": int(falhas_frequentes),
            "veiculo_parado": int(veiculo_parado),
            "valor_estimado": float(valor_estimado),
        }])
        probabilidade = float(modelo_ml.predict_proba(entrada)[0][1])
        risco = classificar(probabilidade)
        pontos = calcular_pontos(idade, quilometragem, manutencoes_ano, custo_manutencao, valor_estimado, acidente_grave, falhas_frequentes, veiculo_parado)
        registro = {
            "nome": nome.strip(),
            "telefone": formatar_telefone(telefone),
            "email": email.strip(),
            "cidade": cidade.strip(),
            "modelo": modelo_veiculo.strip(),
            "ano": int(ano_veiculo),
            "idade": int(idade),
            "quilometragem": int(quilometragem),
            "manutencoes_ano": int(manutencoes_ano),
            "custo_manutencao": float(custo_manutencao),
            "valor_estimado": float(valor_estimado),
            "acidente_grave": bool(acidente_grave),
            "falhas_frequentes": bool(falhas_frequentes),
            "veiculo_parado": bool(veiculo_parado),
            "data_ultima_manutencao": data_ultima_manutencao.isoformat(),
            "causa_ultima_manutencao": causa_ultima_manutencao.strip(),
            "pontos": int(pontos),
            "probabilidade_ia": round(probabilidade * 100, 2),
            "classificacao": risco,
        }
        try:
            salvar_no_supabase(registro)
            st.success("Veículo analisado e registrado com sucesso.")
            st.subheader("Resultado da triagem")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilidade estimada pela IA", f"{probabilidade * 100:.1f}%")
            with col2:
                st.metric("Sistema de pontuação", f"{pontos}/100")
            st.write(f"**Classificação do modelo:** {risco}")
            if risco in ("Alto", "Muito alto"):
                st.warning("Recomendação do protótipo: encaminhar o veículo para uma avaliação técnica presencial.")
            elif risco == "Moderado":
                st.info("Recomendação do protótipo: acompanhar o veículo e considerar uma avaliação preventiva.")
            else:
                st.info("Recomendação do protótipo: nenhuma ação imediata indicada pela triagem.")
            comparacao = pd.DataFrame({"Método": ["Sistema de pontos", "Machine Learning"], "Resultado": [pontos, round(probabilidade * 100, 1)]})
            st.subheader("Comparação dos métodos")
            st.bar_chart(comparacao.set_index("Método"))
            st.caption(f"Acurácia obtida apenas no conjunto de teste SIMULADO: {acuracia_demo * 100:.1f}%. Não representa desempenho real.")
        except Exception as erro:
            st.error("Não foi possível salvar o cadastro no Supabase.")
            with st.expander("Detalhes técnicos"):
                st.code(str(erro))

st.divider()
st.subheader("🔒 Área administrativa")
senha_admin = st.text_input("Senha de administrador", type="password", key="senha_admin")
if senha_admin:
    if senha_admin == st.secrets["ADMIN_PASSWORD"]:
        try:
            registros = carregar_registros()
            df = pd.DataFrame(registros)
            st.success("Acesso liberado.")
            if df.empty:
                st.info("Ainda não existem veículos cadastrados.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Veículos cadastrados", len(df))
                with col2:
                    prioritarios = df[df["classificacao"].isin(["Alto", "Muito alto"])]
                    st.metric("Prioritários", len(prioritarios))
                with col3:
                    st.metric("Probabilidade média", f"{df['probabilidade_ia'].mean():.1f}%")
                filtro_risco = st.selectbox("Filtrar por classificação", ["Todos", "Baixo", "Moderado", "Alto", "Muito alto"])
                df_filtrado = df.copy()
                if filtro_risco != "Todos":
                    df_filtrado = df_filtrado[df_filtrado["classificacao"] == filtro_risco]
                colunas_tabela = ["created_at", "nome", "telefone", "email", "cidade", "modelo", "ano", "quilometragem", "pontos", "probabilidade_ia", "classificacao"]
                colunas_existentes = [c for c in colunas_tabela if c in df_filtrado.columns]
                st.dataframe(df_filtrado[colunas_existentes], use_container_width=True, hide_index=True)
                csv = df_filtrado.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Baixar cadastros em CSV", data=csv, file_name="veiculos_cadastrados.csv", mime="text/csv", use_container_width=True)
        except Exception as erro:
            st.error("Não foi possível carregar os dados do Supabase.")
            with st.expander("Detalhes técnicos"):
                st.code(str(erro))
    else:
        st.error("Senha incorreta.")

st.divider()
with st.expander("Como este protótipo funciona?"):
    st.write("""
    1. O proprietário preenche o questionário.
    2. A Random Forest analisa os dados informados.
    3. Um sistema de pontuação simples roda em paralelo.
    4. O resultado da IA e do sistema de pontos é apresentado.
    5. O cadastro é salvo no banco de dados Supabase.
    6. A área administrativa permite visualizar e exportar os registros.

    O modelo atual foi treinado com dados simulados. Para aplicação industrial,
    ele deve ser retreinado e validado com dados históricos reais.
    """)

st.warning("Protótipo acadêmico. Não utilizar para decisões reais sobre descarte, segurança, manutenção ou condição mecânica de veículos.")
