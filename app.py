import json
from dotenv import dotenv_values
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import plotly.graph_objects as go

from openai import OpenAI
from qdrant_client import QdrantClient

env = dotenv_values("model/.env")

if "QDRANT_URL" in st.secrets:
    env["QDRANT_URL"] = st.secrets["QDRANT_URL"]
if "QDRANT_API_KEY" in st.secrets:
    env["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
if "OPENAI_API_KEY" in st.secrets:
    env["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


QDRANT_COLLECTION_NAME = "find_friends_clusters"

EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_DIM = 3072

MODEL_NAME = "welcome_survey_clustering_pipeline_v2"

DATA = "welcome_survey_simple_v2.csv"

CLUSTER_NAMES_AND_DESCRIPTIONS = "welcome_survey_cluster_names_and_description_v2.json"

def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep = ";")
    df_with_cluster = predict_model(model, data = all_df)
    return df_with_cluster

def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r") as f:
        return json.loads(f.read())
#
# OPENAI
#

def get_openai_client():
    return OpenAI(api_key = env["OPENAI_API_KEY"])

def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input = [text],
        model = EMBEDDING_MODEL,
        dimensions = EMBEDDING_DIM
    )

    return result.data[0].embedding

#
# DATABASE
#

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=env["QDRANT_URL"], 
        api_key=env["QDRANT_API_KEY"],
    )

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Podpięty do bazdy danych")
    else:
        st.toast(""""
        Nie jesteś podpięty do bazdy danych !!
        Dzwoń do admina 
        """)
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        st.stop()

def search_cluster_in_db(query):
    qdrant_client = get_qdrant_client()
    clusters = qdrant_client.search(
            collection_name = QDRANT_COLLECTION_NAME,
            query_vector = get_embedding(query),
            limit = 2
        )
    result = []
    for cluster in clusters:
            result.append({
                "cluster": cluster.payload["cluster_id"],
                "text": cluster.payload["description"],
                "score": cluster.score
            })
    return result


#
# MAIN
#

# openai API protection
if not ss.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        ss["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Podaj swój klucz API OpenAI aby móc korzystac z tej aplikacji")
        ss["openai_api_key"] = st.text_input("Klucz API", type = "password")
        if ss["openai_api_key"]:
            st.rerun()

if not ss.get("openai_api_key"):
    st.stop()

# Session state init
if "search_mode" in ss:
    ss["search_mode"] = ""

# HEADER
st.markdown("## Znajdz kolegę, koleżankę 👫")
st.markdown("---")

assure_db_collection_exists()

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pommożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    tab1, tab2 = st.tabs(["wybierz", "opisz"])
    with tab1:
        age = st.selectbox(
            "Wiek",
            [
                "<18",
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                ">=65"
            ],
        )
        edu_level = st.selectbox(
            "Wykształcenie",
            [
                "Podstawowe",
                "Średnie",
                "Wyższe"   
            ]
        )
        fav_animals = st.selectbox(
            "Ulubione zwierzęta",
            [
                "Brak ulubionych",
                "Psy",
                "Koty",
                "Psy i koty"
            ]
        )
        fav_place = st.selectbox(
            "Ulubione miejsce",
            [
                "Nad wodą",
                "W lesie",
                "W górach",
                "Inne"
            ]
        )
        gender = st.radio("Płeć", ["Mężczyzna", "Kobieta"])
        ss["search_mode"] = "search"

    with tab2:
        description_from_user = st.text_area(
            "Opisz siebie",
            placeholder = """

            podpowiedzi:

            - jakie lubisz zwierzęta
            - gdzie lubisz jeździć na wakacje
            - czy jesteś kobieta czy meżczyzna
            - młody czy doświadczony
                    (możesz podać cyfrę)
            - gdzie chodziłeś do szkoły
            
            """,
            height = 400, key = "description")
        if st.button("Wyśllij"):
            ss["search_mode"] = "description"


    person_df = pd.DataFrame([
        {
            "age": age,
            "edu_level": edu_level,
            "fav_animals": fav_animals,
            "fav_place": fav_place,
            "gender": gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_add_descriptions = get_cluster_names_and_descriptions()

if ss["search_mode"] == "description":
    result_of_searching = search_cluster_in_db(description_from_user)
    predicted_cluster_id = result_of_searching[0]["cluster"]

elif ss["search_mode"] == "search":
    predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]

# predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_add_descriptions[predicted_cluster_id]

# Header
st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data["description"])

# info section

col1, col2, col3 = st.columns(3)
with col1:
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
    st.metric("Liczba Twoich znajomych", len(same_cluster_df))
with col2:
    st.metric("Liczba ankietowanych", len(all_df))

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x = "age")
fig.update_layout(
    title = "Rozkład wieku w grupie",
    xaxis_title = "Wiek",
    yaxis_title = "Liczba osób"
)
st.plotly_chart(fig)

fig = go.Figure(data= [go.Pie(
    values = [value for value in same_cluster_df["age"].value_counts() if value != 0],
    labels = same_cluster_df["age"].value_counts().loc[lambda x: x != 0].index.tolist(),
    hole = .3
)])
fig.update_layout(
    title = "Wiek ptocentowo"
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x = "edu_level")
fig.update_layout(
    title = "Rozkład wykształcenia w grupie",
    xaxis_title = "Wykształcenie",
    yaxis_title = "Liczba osób"
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x = "fav_animals")
fig.update_layout(
    title = "Rozkład wulubionych zwierząt w grupie",
    xaxis_title = "Ulubione zwierzęta",
    yaxis_title = "Liczba osób"
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x = "fav_place")
fig.update_layout(
    title = "Rozkład wulubionych miejsc w grupie",
    xaxis_title = "Ulubione miejsce",
    yaxis_title = "Liczba osób"
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x = "gender")
fig.update_layout(
    title = "Rozkład półci w grupie",
    xaxis_title = "Płeć",
    yaxis_title = "Liczba osób"
)
st.plotly_chart(fig)

if ss["search_mode"] == "description":
    st.markdown("---")
    st.markdown("## Statystyki")

    st.markdown("#### Najbliższy klaster")

    kol1, kol2 = st.columns(2)
    with kol1:
        st.metric("Klaster", result_of_searching[0]['cluster'])

    with kol2:
        st.metric("Ocena", result_of_searching[0]['score'])
    
    st.markdown("#### Następnym najbliższy klaster")

    kol1, kol2 = st.columns(2)
    with kol1:
        st.metric("Kalater", result_of_searching[1]['cluster'])

    with kol2:
        st.metric("Ocena", result_of_searching[1]['score'])

