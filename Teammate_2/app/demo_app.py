"""Streamlit demo for the pulsar mini-project.

Takes the eight HTRU2 measurements for a single candidate and returns

  1. whether it is a pulsar, from the supervised GDA model, and
  2. if it is, which of the three unsupervised groups it belongs to,
     from both k-means and the Self-Organizing Map.

Launch from inside the Teammate_2 directory:

    streamlit run app/demo_app.py
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# The app lives one level below the project root, so paths are resolved
# relative to Teammate_2 rather than to the working directory streamlit
# happens to be launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)

# Both pickled models are custom classes, so their modules have to be
# importable before pickle.load can rebuild them.
for path in (PROJECT_DIR, os.path.join(REPO_DIR, 'Teammate_1', 'src')):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.preprocess import FEATURE_LABELS, FEATURE_NAMES, load_data

ARTIFACT_PATH = os.path.join(PROJECT_DIR, 'outputs', 'models',
                             'unsupervised_artifacts.pkl')
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'HTRU_2.csv')

# Short descriptions of the three k-means clusters, written from the
# cluster profile table produced by main_unsupervised.py.
CLUSTER_NOTES = {
    0: 'Mid-range integrated profile with a moderately dispersed DM-SNR '
       'curve. The largest and most typical group.',
    1: 'Low integrated-profile mean with strong profile skew and the '
       'highest DM-SNR mean. Bright, sharply peaked candidates.',
    2: 'High integrated-profile mean with a very weak DM-SNR curve and '
       'extreme DM-SNR skew. The smallest and most distinctive group.',
}


@st.cache_resource
def load_artifacts():
    """Loads the fitted scaler, k-means, SOM, PCA and GDA objects."""
    with open(ARTIFACT_PATH, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_reference_data():
    """Loads the raw dataset so the app can show sensible input ranges."""
    return load_data(DATA_PATH)


def predict_candidate(artifacts, values):
    """Runs one candidate through the supervised and unsupervised models."""
    x_raw = np.array(values, dtype=np.float64).reshape(1, -1)

    pulsar_prob = float(artifacts['gda'].predict_proba(x_raw)[0])
    is_pulsar = pulsar_prob >= 0.5

    # The clustering models were fitted on standardised pulsar data, so
    # the candidate has to go through the same scaler.
    x_scaled = artifacts['scaler'].transform(x_raw)

    kmeans_cluster = int(artifacts['kmeans'].predict(x_scaled)[0])
    bmu = int(artifacts['som'].predict(x_scaled)[0])
    som_cluster = int(artifacts['som_node_groups'][bmu])
    bmu_row, bmu_col = divmod(bmu, artifacts['som'].cols)
    x_pca = artifacts['pca'].transform(x_scaled)[0]

    return {
        'pulsar_probability': pulsar_prob,
        'is_pulsar': is_pulsar,
        'kmeans_cluster': kmeans_cluster,
        'som_cluster': som_cluster,
        'som_node': (bmu_row, bmu_col),
        'pca_coords': x_pca,
    }


def plot_position(artifacts, df, x_pca):
    """Plots the candidate against the clustered pulsar population."""
    pulsars = df[df['target'] == 1][FEATURE_NAMES]
    scaled = artifacts['scaler'].transform(pulsars.values)
    coords = artifacts['pca'].transform(scaled)
    labels = artifacts['kmeans'].predict(scaled)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', s=12,
               alpha=0.45)
    ax.scatter(x_pca[0], x_pca[1], color='red', s=180, marker='*',
               edgecolor='black', linewidth=0.8, label='This candidate',
               zorder=5)
    var = artifacts['pca'].explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var[0] * 100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var[1] * 100:.1f}% variance)')
    ax.set_title('Candidate position among known pulsars')
    ax.legend(loc='best')
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title='Pulsar Candidate Explorer',
                       layout='wide')
    st.title('Pulsar Candidate Explorer')
    st.caption('UE24CS352A Machine Learning mini-project, HTRU2 dataset. '
               'Enter the eight measurements for a candidate signal to get '
               'a pulsar prediction and a cluster assignment.')

    if not os.path.exists(ARTIFACT_PATH):
        st.error('Fitted models not found. Run `python main_unsupervised.py` '
                 'from the Teammate_2 directory first.')
        return

    artifacts = load_artifacts()
    df = load_reference_data()
    pulsars = df[df['target'] == 1]

    st.sidebar.header('Candidate measurements')

    preset = st.sidebar.selectbox(
        'Start from',
        ['Median pulsar', 'Median non-pulsar', 'Random pulsar',
         'Random non-pulsar'],
    )

    if preset == 'Median pulsar':
        defaults = pulsars[FEATURE_NAMES].median().values
    elif preset == 'Median non-pulsar':
        defaults = df[df['target'] == 0][FEATURE_NAMES].median().values
    elif preset == 'Random pulsar':
        defaults = pulsars[FEATURE_NAMES].sample(1).values[0]
    else:
        defaults = df[df['target'] == 0][FEATURE_NAMES].sample(1).values[0]

    values = []
    for name, default in zip(FEATURE_NAMES, defaults):
        column = df[name]
        values.append(st.sidebar.number_input(
            FEATURE_LABELS[name],
            min_value=float(column.min()),
            max_value=float(column.max()),
            value=float(default),
            step=0.01,
            format='%.4f',
        ))

    result = predict_candidate(artifacts, values)

    left, right = st.columns([1, 1])

    with left:
        st.subheader('Supervised prediction')
        if result['is_pulsar']:
            st.success(f'Classified as a **pulsar** '
                       f'(P = {result["pulsar_probability"]:.4f})')
        else:
            st.warning(f'Classified as **not a pulsar** '
                       f'(P = {result["pulsar_probability"]:.4f})')
        st.progress(min(max(result['pulsar_probability'], 0.0), 1.0))
        st.caption('Gaussian Discriminant Analysis from the supervised half '
                   'of the project, fitted on all 17,898 candidates.')

        st.subheader('Cluster assignment')
        if not result['is_pulsar']:
            st.info('The clustering models were trained on confirmed pulsars '
                    'only, so the group below is shown for reference and is '
                    'not meaningful for a non-pulsar candidate.')

        col_a, col_b = st.columns(2)
        col_a.metric('K-means cluster', result['kmeans_cluster'])
        col_b.metric('SOM cluster', result['som_cluster'])
        st.caption(f'Best Matching Unit on the 5x5 grid: '
                   f'row {result["som_node"][0]}, '
                   f'column {result["som_node"][1]}')

        if result['kmeans_cluster'] == result['som_cluster']:
            st.success('Both methods agree on the group.')
        else:
            st.warning('The two methods disagree, which usually means the '
                       'candidate sits near a cluster boundary.')

        st.markdown(f'**Cluster {result["kmeans_cluster"]}:** '
                    f'{CLUSTER_NOTES[result["kmeans_cluster"]]}')

    with right:
        st.subheader('Where it sits')
        st.pyplot(plot_position(artifacts, df, result['pca_coords']))

    with st.expander('Cluster profiles (mean of each raw feature)'):
        st.dataframe(artifacts['cluster_profile'].round(3))

    with st.expander('Candidate input as entered'):
        st.dataframe(pd.DataFrame([values], columns=FEATURE_NAMES).T
                     .rename(columns={0: 'value'}))


if __name__ == '__main__':
    main()
