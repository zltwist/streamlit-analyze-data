import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

from analysis import full_analysis, dynamic_segment_comparison, stress_summary

st.set_page_config(page_title='Analyzer Dashboard', layout='wide', initial_sidebar_state='expanded')

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT.parent / 'dataset'

if Path('style.css').exists():
    with open('style.css', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(18,20,35,0.95), rgba(28,48,66,0.9));
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: 0.35rem 0 0 0; opacity: 0.88; }
    .card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: white;
        border: 1px solid rgba(20,20,40,0.08);
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        height: 100%;
    }
    .card h4 { margin: 0 0 0.35rem 0; }
    .muted { color: #5b6472; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Analyzer Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


def list_dataset_files():
    if not DATASET_DIR.exists():
        return []
    return sorted([p for p in DATASET_DIR.iterdir() if p.is_file()])


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in ['.csv', '.txt']:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, encoding='latin1')
    if suffix in ['.xls', '.xlsx']:
        return pd.read_excel(path)
    return pd.read_csv(path)


def fmt_pct(value):
    try:
        return f'{value:.2%}'
    except Exception:
        return '-'


def section_header(title: str, subtitle: str = ''):
    st.markdown(f"<div class='card'><h4>{title}</h4><div class='muted'>{subtitle}</div></div>", unsafe_allow_html=True)


st.sidebar.header('Dataset')
dataset_files = list_dataset_files()
df = None
selected_file = None

if dataset_files:
    choices = ['None'] + [p.name for p in dataset_files]
    selected_file = st.sidebar.selectbox('Pilih file dari folder dataset/', choices)
    if selected_file != 'None':
        try:
            df = read_dataset(DATASET_DIR / selected_file)
        except Exception as e:
            st.sidebar.error(f'Gagal membaca file: {e}')
else:
    st.sidebar.info('Folder dataset/ tidak ditemukan atau kosong.')

uploaded = st.sidebar.file_uploader('Atau upload CSV/XLSX', type=['csv', 'xls', 'xlsx'])
if uploaded is not None and df is None:
    try:
        if uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        selected_file = uploaded.name
    except Exception as e:
        st.sidebar.error(f'Gagal membaca file upload: {e}')

st.sidebar.header('Parameter Analisis')
gain_yes = st.sidebar.number_input('Gain jika deposit', value=100.0, step=5.0)
loss_no = st.sidebar.number_input('Loss jika tidak deposit', value=-20.0, step=5.0)
n_sim = st.sidebar.number_input('Jumlah simulasi Monte Carlo', min_value=1000, max_value=100000, value=10000, step=1000)
compare_dimension = st.sidebar.selectbox('Dimensi perbandingan', ['job', 'education', 'housing'])
top_n_compare = st.sidebar.slider('Top N segment yang dibandingkan', min_value=3, max_value=10, value=5)
prob_shift = st.sidebar.slider('Shock probabilitas deposit', min_value=-50, max_value=50, value=0, step=5, help='Menggeser P(deposit=yes) dalam skenario stres') / 100.0
gain_shift = st.sidebar.slider('Shock gain', min_value=-50, max_value=50, value=0, step=5, help='Persentase perubahan gain pada skenario stres') / 100.0
loss_shift = st.sidebar.slider('Shock loss', min_value=-50, max_value=50, value=0, step=5, help='Persentase perubahan loss pada skenario stres') / 100.0
run_analysis = st.sidebar.button('Jalankan Analisis')

if df is not None:
    st.markdown('### Preview dataset')
    st.write(f'File aktif: {selected_file}')
    st.dataframe(df.head(10), use_container_width=True, height=240)
else:
    st.info('Pilih dataset dari sidebar untuk memulai analisis.')
    st.stop()

required_cols = ['deposit', 'job', 'education', 'housing', 'balance']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.warning('Beberapa kolom data tidak ditemukan: ' + ', '.join(missing_cols))

if not run_analysis:
    st.info('Pilih dataset lalu klik **Jalankan Analisis** untuk melihat hasil.')
    st.stop()

try:
    result = full_analysis(df, gain_yes=gain_yes, loss_no=loss_no, n_simulations=int(n_sim))
except Exception as e:
    st.error(f'Analisis gagal: {e}')
    st.stop()

comparison_df = dynamic_segment_comparison(df, dimension=compare_dimension, top_n=int(top_n_compare), gain_yes=gain_yes, loss_no=loss_no)
stress = stress_summary(result['p_yes'], gain_yes, loss_no, probability_shift_pct=prob_shift, gain_shift_pct=gain_shift, loss_shift_pct=loss_shift)

# Top summary cards
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"<div class='card'><h4>P(deposit = yes)</h4><div style='font-size:1.7rem;font-weight:700'>{result['p_yes']:.2%}</div><div class='muted'>Probabilitas marginal utama</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='card'><h4>EV per nasabah</h4><div style='font-size:1.7rem;font-weight:700'>{result['ev_all']:.2f}</div><div class='muted'>Tanpa segmentasi</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='card'><h4>Total EV all</h4><div style='font-size:1.7rem;font-weight:700'>{result['total_ev_all']:,.0f}</div><div class='muted'>Offer ke semua nasabah</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='card'><h4>Total EV targeting</h4><div style='font-size:1.7rem;font-weight:700'>{result['total_ev_targeting']:,.0f}</div><div class='muted'>EV > 0</div></div>", unsafe_allow_html=True)

st.markdown('---')
st.subheader('Kontrol Perbandingan')
st.markdown("<div class='card'><h4>Skenario stres</h4><p class='muted'>Slider di sidebar langsung memperbarui probabilitas, gain, dan loss.</p></div>", unsafe_allow_html=True)
stress_df = pd.DataFrame([
    {'Scenario': 'Base', 'P(yes)': result['p_yes'], 'Gain': gain_yes, 'Loss': loss_no, 'EV': stress['base_ev']},
    {'Scenario': 'Stress', 'P(yes)': stress['stressed_p_yes'], 'Gain': stress['stressed_gain'], 'Loss': stress['stressed_loss'], 'EV': stress['stressed_ev']},
])
st.dataframe(stress_df.style.format({'P(yes)': '{:.2%}', 'Gain': '{:.2f}', 'Loss': '{:.2f}', 'EV': '{:.2f}'}), use_container_width=True)

st.markdown(f"<div class='card'><h4>Top {top_n_compare} {compare_dimension} comparison</h4><p class='muted'>Perbandingan segment mengikuti slider top-N.</p></div>", unsafe_allow_html=True)
if not comparison_df.empty:
    chart = alt.Chart(comparison_df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('EV:Q', title='Expected Value'),
        y=alt.Y(f'{compare_dimension}:N', sort='-x', title=compare_dimension),
        color=alt.condition(alt.datum.EV > 0, alt.value('#1f9d55'), alt.value('#c0392b')),
        tooltip=[compare_dimension, 'P(yes)', 'EV', 'Count']
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(comparison_df.style.format({'P(yes)': '{:.2%}', 'EV': '{:.2f}'}), use_container_width=True)
else:
    st.info('Dimensi yang dipilih tidak tersedia untuk perbandingan.')

st.markdown('---')
main_tabs = st.tabs(['1. Decision Under Risk', '2. Probabilistic Modeling', '3. Simulation & Sensitivity', '4. Download / Data'])

# TAB 1
with main_tabs[0]:
    st.subheader('Decision Under Risk')

    a1, a2, a3 = st.columns(3)
    a1.metric('P(deposit=yes)', fmt_pct(result['p_yes']))
    a2.metric('P(deposit=no)', fmt_pct(result['p_no']))
    a3.metric('EV tanpa segmentasi', f"{result['ev_all']:.2f}")

    sub_tabs = st.tabs(['Summary', 'Job', 'Education', 'Housing', 'Balance', 'Recommendation'])

    with sub_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'><h4>Inti keputusan</h4><p class='muted'>Jika EV > 0 maka menawarkan produk menguntungkan.</p></div>", unsafe_allow_html=True)
            st.write(f"**EV per nasabah**: {result['ev_all']:.2f}")
            st.write(f"**Keputusan**: {'MENAWARKAN' if result['ev_all'] > 0 else 'TIDAK MENAWARKAN'}")
        with c2:
            base_df = pd.DataFrame([
                {'Metric': 'P(deposit=yes)', 'Value': result['p_yes']},
                {'Metric': 'P(deposit=no)', 'Value': result['p_no']},
                {'Metric': 'EV all', 'Value': result['ev_all']},
                {'Metric': 'Total EV all', 'Value': result['total_ev_all']},
                {'Metric': 'Total EV targeting', 'Value': result['total_ev_targeting']},
            ])
            st.dataframe(base_df, use_container_width=True)

        st.markdown('#### Perbandingan segmentasi')
        if not comparison_df.empty:
            st.dataframe(comparison_df.style.format({'P(yes)': '{:.2%}', 'EV': '{:.2f}'}), use_container_width=True)

    with sub_tabs[1]:
        if not result['df_job'].empty:
            left, right = st.columns([1.2, 1])
            with left:
                chart = alt.Chart(result['df_job']).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X('EV:Q', title='Expected Value'),
                    y=alt.Y('Job:N', sort='-x', title='Job'),
                    color=alt.condition(alt.datum.EV > 0, alt.value('#1f9d55'), alt.value('#c0392b')),
                    tooltip=['Job', 'P(yes)', 'EV', 'Count']
                ).properties(height=420)
                st.altair_chart(chart, use_container_width=True)
            with right:
                st.dataframe(result['df_job'].style.format({'P(yes)': '{:.2%}', 'EV': '{:.2f}'}), use_container_width=True, height=420)
        else:
            st.info('Kolom job tidak ditemukan pada dataset ini.')

    with sub_tabs[2]:
        if not result['df_edu'].empty:
            left, right = st.columns([1.2, 1])
            with left:
                chart = alt.Chart(result['df_edu']).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X('EV:Q', title='Expected Value'),
                    y=alt.Y('Education:N', sort='-x', title='Education'),
                    color=alt.condition(alt.datum.EV > 0, alt.value('#1f9d55'), alt.value('#c0392b')),
                    tooltip=['Education', 'P(yes)', 'EV', 'Count']
                ).properties(height=420)
                st.altair_chart(chart, use_container_width=True)
            with right:
                st.dataframe(result['df_edu'].style.format({'P(yes)': '{:.2%}', 'EV': '{:.2f}'}), use_container_width=True, height=420)
        else:
            st.info('Kolom education tidak ditemukan pada dataset ini.')

    with sub_tabs[3]:
        if not result['df_housing'].empty:
            chart = alt.Chart(result['df_housing']).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('EV:Q', title='Expected Value'),
                y=alt.Y('Housing:N', sort='-x', title='Housing'),
                color=alt.condition(alt.datum.EV > 0, alt.value('#1f9d55'), alt.value('#c0392b')),
                tooltip=['Housing', 'P(yes)', 'EV', 'Count']
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(result['df_housing'].style.format({'P(yes)': '{:.2%}', 'EV': '{:.2f}'}), use_container_width=True)
        else:
            st.info('Kolom housing tidak ditemukan pada dataset ini.')

    with sub_tabs[4]:
        if not result['df_balance'].empty:
            chart = alt.Chart(result['df_balance']).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('EV:Q', title='Expected Value'),
                y=alt.Y('Balance:N', sort='-x', title='Balance Category'),
                color=alt.condition(alt.datum.EV > 0, alt.value('#1f9d55'), alt.value('#c0392b')),
                tooltip=['Balance', 'P(yes)', 'EV', 'Count']
            ).properties(height=340)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(result['df_balance'].style.format({'P(yes)': '{:.2%}', 'EV': '{:.2f}'}), use_container_width=True)
        else:
            st.info('Kolom balance tidak ditemukan pada dataset ini.')

    with sub_tabs[5]:
        df_aug = result['df_augmented']
        col_a, col_b = st.columns(2)
        with col_a:
            if 'recommend_combined' in df_aug.columns:
                rec = df_aug['recommend_combined'].value_counts().rename_axis('recommend').reset_index(name='count')
                rec['recommend'] = rec['recommend'].map({True: 'Target (EV > 0)', False: 'Hindari (EV ≤ 0)'})
                pie = alt.Chart(rec).mark_arc(innerRadius=65).encode(
                    theta='count:Q',
                    color='recommend:N',
                    tooltip=['recommend', 'count']
                ).properties(height=320)
                st.altair_chart(pie, use_container_width=True)
            else:
                st.info('Rekomendasi combined tidak tersedia.')
        with col_b:
            if 'recommend_combined' in df_aug.columns:
                st.dataframe(df_aug[['job', 'education', 'balance', 'ev_combined', 'recommend_combined']].head(25) if {'job', 'education', 'balance', 'ev_combined', 'recommend_combined'}.issubset(df_aug.columns) else df_aug.head(25), use_container_width=True, height=320)
            st.markdown(f"**Target nasabah**: {int(df_aug['recommend_combined'].sum()) if 'recommend_combined' in df_aug.columns else 0}")
            st.markdown(f"**Hindari**: {int((~df_aug['recommend_combined']).sum()) if 'recommend_combined' in df_aug.columns else 0}")

# TAB 2
with main_tabs[1]:
    st.subheader('Probabilistic Modeling')

    prob_tabs = st.tabs(['Marginal & Conditional', 'Joint Probability', 'Continuous Distribution', 'Discrete Distribution', 'Risk Model & DSS'])

    with prob_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'><h4>Probabilitas marginal</h4><p class='muted'>P(yes) dan P(no) untuk target deposit.</p></div>", unsafe_allow_html=True)
            base = pd.DataFrame({'Class': ['Deposit Yes', 'Deposit No'], 'Probability': [result['p_yes'], result['p_no']]})
            bar = alt.Chart(base).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('Class:N', title='Class'),
                y=alt.Y('Probability:Q', title='Probability'),
                color=alt.Color('Class:N', legend=None),
                tooltip=['Class', alt.Tooltip('Probability:Q', format='.2%')]
            ).properties(height=300)
            st.altair_chart(bar, use_container_width=True)
        with c2:
            st.dataframe(base.style.format({'Probability': '{:.2%}'}), use_container_width=True, height=300)

        st.markdown('#### Conditional probability tables')
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('**By Job**')
            st.dataframe(result['job_prob'].style.format({'P(yes|cond)': '{:.2%}'}), use_container_width=True, height=260)
        with cc2:
            st.markdown('**By Education**')
            st.dataframe(result['education_prob'].style.format({'P(yes|cond)': '{:.2%}'}), use_container_width=True, height=260)
        cc3, cc4 = st.columns(2)
        with cc3:
            st.markdown('**By Housing**')
            st.dataframe(result['housing_prob'].style.format({'P(yes|cond)': '{:.2%}'}), use_container_width=True, height=260)
        with cc4:
            st.markdown('**By Loan**')
            st.dataframe(result['loan_prob'].style.format({'P(yes|cond)': '{:.2%}'}), use_container_width=True, height=260)

    with prob_tabs[1]:
        j1, j2 = st.columns(2)
        if not result['joint_job_deposit'].empty:
            heat_job = result['joint_job_deposit'].reset_index().melt(id_vars='job', var_name='deposit', value_name='probability')
            chart1 = alt.Chart(heat_job).mark_rect().encode(
                x='deposit:N',
                y='job:N',
                color=alt.Color('probability:Q', scale=alt.Scale(scheme='yelloworangered')),
                tooltip=['job', 'deposit', alt.Tooltip('probability:Q', format='.2%')]
            ).properties(height=320)
            j1.altair_chart(chart1, use_container_width=True)
        if not result['joint_edu_deposit'].empty:
            heat_edu = result['joint_edu_deposit'].reset_index().melt(id_vars='education', var_name='deposit', value_name='probability')
            chart2 = alt.Chart(heat_edu).mark_rect().encode(
                x='deposit:N',
                y='education:N',
                color=alt.Color('probability:Q', scale=alt.Scale(scheme='yelloworangered')),
                tooltip=['education', 'deposit', alt.Tooltip('probability:Q', format='.2%')]
            ).properties(height=320)
            j2.altair_chart(chart2, use_container_width=True)
        st.markdown('**Joint probability tables**')
        kk1, kk2 = st.columns(2)
        with kk1:
            if not result['joint_job_deposit'].empty:
                st.dataframe(result['joint_job_deposit'].style.format('{:.2%}'), use_container_width=True)
        with kk2:
            if not result['joint_edu_deposit'].empty:
                st.dataframe(result['joint_edu_deposit'].style.format('{:.2%}'), use_container_width=True)

    with prob_tabs[2]:
        if not result['continuous_stats'].empty:
            st.dataframe(result['continuous_stats'].style.format({'Mean': '{:.2f}', 'Median': '{:.2f}', 'Std': '{:.2f}', 'Skewness': '{:.3f}', 'Kurtosis': '{:.3f}', 'Min': '{:.2f}', 'Max': '{:.2f}'}), use_container_width=True)
        else:
            st.info('Tidak ada variabel kontinu yang bisa dianalisis dari dataset ini.')

        # Visualize continuous distributions when available
        cont_cols = [c for c in ['age', 'balance', 'duration'] if c in df.columns]
        if cont_cols:
            vis_tabs = st.tabs(cont_cols)
            for tab, col in zip(vis_tabs, cont_cols):
                with tab:
                    series = pd.to_numeric(df[col], errors='coerce').dropna()
                    if series.empty:
                        st.info(f'Kolom {col} tidak berisi data numerik yang cukup.')
                        continue
                    tmp = pd.DataFrame({col: series})
                    hist = alt.Chart(tmp).transform_density(
                        col,
                        as_=[col, 'density']
                    ).mark_area(opacity=0.55).encode(
                        x=alt.X(f'{col}:Q', title=col),
                        y='density:Q'
                    ).properties(height=260)
                    box = alt.Chart(tmp).mark_boxplot(size=45).encode(y=alt.Y(f'{col}:Q', title=col)).properties(height=120)
                    st.altair_chart(hist, use_container_width=True)
                    st.altair_chart(box, use_container_width=True)

    with prob_tabs[3]:
        if result['discrete_tables']:
            disc_tabs = st.tabs(list(result['discrete_tables'].keys()))
            for tab, (var, table) in zip(disc_tabs, result['discrete_tables'].items()):
                with tab:
                    chart = alt.Chart(table).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X('Probability:Q', title='Probability'),
                        y=alt.Y(f'{var}:N', sort='-x', title=var),
                        color=alt.Color(f'{var}:N', legend=None),
                        tooltip=[var, alt.Tooltip('Probability:Q', format='.2%'), 'Count']
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
                    st.dataframe(table.style.format({'Probability': '{:.2%}'}), use_container_width=True)
        else:
            st.info('Tidak ada variabel diskrit yang cocok di dataset ini.')

    with prob_tabs[4]:
        if not result['risk_model'].empty:
            r1, r2 = st.columns([1.2, 1])
            with r1:
                chart = alt.Chart(result['risk_model']).mark_circle(size=90, opacity=0.75).encode(
                    x=alt.X('avg_balance:Q', title='Average Balance'),
                    y=alt.Y('p_deposit_yes:Q', title='P(deposit=yes)'),
                    color=alt.Color('risk_score:Q', scale=alt.Scale(scheme='redyellowgreen')),
                    tooltip=['job', 'education', alt.Tooltip('p_deposit_yes:Q', format='.2%'), 'risk_level']
                ).properties(height=360)
                st.altair_chart(chart, use_container_width=True)
            with r2:
                st.dataframe(result['risk_model'].style.format({'p_deposit_yes': '{:.2%}', 'avg_age': '{:.2f}', 'avg_balance': '{:.2f}', 'risk_score': '{:.2%}'}), use_container_width=True, height=360)
        else:
            st.info('Risk model tidak bisa dibuat karena kolom job/education/balance tidak lengkap.')

        st.markdown('#### Prediksi untuk nasabah contoh')
        st.dataframe(result['test_predictions'].style.format({'P(deposit=yes)': '{:.2%}'}), use_container_width=True)

        st.markdown('#### DSS output untuk dataset aktif')
        dss_cols = [c for c in ['age', 'job', 'education', 'balance', 'p_deposit_pred', 'risk_category', 'recommendation'] if c in result['df_prob'].columns]
        st.dataframe(result['df_prob'][dss_cols].head(20), use_container_width=True)

# TAB 3
with main_tabs[2]:
    st.subheader('Simulation & Sensitivity')

    s1, s2 = st.columns(2)
    s1.metric('Mean deposit rate simulasi', f"{result['sim_deposit_rate'].mean():.2%}")
    s2.metric('Mean EV simulasi', f"{result['sim_ev'].mean():.2f}")

    sim_tabs = st.tabs(['Monte Carlo', 'Sensitivity Matrix', 'Probability Sensitivity', 'Job Sensitivity', 'Tornado', 'Scenarios', 'Robustness'])

    with sim_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            hist = alt.Chart(pd.DataFrame({'rate': result['sim_deposit_rate']})).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X('rate:Q', bin=alt.Bin(maxbins=45), title='Deposit Rate'),
                y='count()'
            ).properties(title='Monte Carlo Deposit Rate', height=300)
            st.altair_chart(hist, use_container_width=True)
        with c2:
            hist_ev = alt.Chart(pd.DataFrame({'ev': result['sim_ev']})).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X('ev:Q', bin=alt.Bin(maxbins=45), title='EV'),
                y='count()'
            ).properties(title='Monte Carlo EV', height=300)
            st.altair_chart(hist_ev, use_container_width=True)
        st.write(f"EV deterministik: **{result['ev_deterministic']:.2f}**")
        st.write(f"Break-even P(yes): **{result['be_threshold']:.2%}**")

        st.markdown('#### Skenario stres berbasis slider')
        st.dataframe(stress_df.style.format({'P(yes)': '{:.2%}', 'Gain': '{:.2f}', 'Loss': '{:.2f}', 'EV': '{:.2f}'}), use_container_width=True)

    with sim_tabs[1]:
        heat_df = result['sensitivity_df'].reset_index().melt(id_vars='index', var_name='Loss', value_name='EV')
        heat_df = heat_df.rename(columns={'index': 'Gain'})
        heat = alt.Chart(heat_df).mark_rect().encode(
            x=alt.X('Loss:O', title='Loss'),
            y=alt.Y('Gain:O', title='Gain'),
            color=alt.Color('EV:Q', scale=alt.Scale(scheme='redyellowgreen')),
            tooltip=['Gain', 'Loss', alt.Tooltip('EV:Q', format='.2f')]
        ).properties(height=380)
        st.altair_chart(heat, use_container_width=True)
        st.dataframe(result['sensitivity_df'].style.format('{:.2f}'), use_container_width=True)

    with sim_tabs[2]:
        prob_df = pd.DataFrame({'Probability': result['p_range'], 'EV': result['ev_by_prob']})
        line = alt.Chart(prob_df).mark_line(point=True).encode(
            x=alt.X('Probability:Q', title='Probability of deposit'),
            y=alt.Y('EV:Q', title='Expected Value'),
            tooltip=['Probability', alt.Tooltip('EV:Q', format='.2f')]
        ).properties(height=320)
        st.altair_chart(line, use_container_width=True)
        st.dataframe(prob_df.style.format({'Probability': '{:.2%}', 'EV': '{:.2f}'}), use_container_width=True)

    with sim_tabs[3]:
        if not result['job_probs'].empty and not result['job_sensitivity_df'].empty:
            job_loss_cols = [c for c in result['job_sensitivity_df'].columns if c.startswith('EV_Loss_')]
            long = result['job_sensitivity_df'].melt(id_vars=['Job', 'P(yes)'], value_vars=job_loss_cols, var_name='Scenario', value_name='EV')
            long['Scenario'] = long['Scenario'].str.replace('EV_Loss_', 'Loss=', regex=False)
            chart = alt.Chart(long).mark_bar().encode(
                x=alt.X('EV:Q', title='EV'),
                y=alt.Y('Job:N', sort='-x'),
                color=alt.Color('Scenario:N'),
                tooltip=['Job', 'Scenario', alt.Tooltip('EV:Q', format='.2f')]
            ).properties(height=420)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(result['job_sensitivity_df'].style.format({'P(yes)': '{:.2%}'}), use_container_width=True)
        else:
            st.info('Tidak ada analisis sensitivitas per job karena kolom job tidak lengkap.')

    with sim_tabs[4]:
        tornado = result['tornado_df'].copy()
        tornado['EV_start'] = tornado['EV_start'].astype(float)
        tornado['EV_end'] = tornado['EV_end'].astype(float)
        tornado_long = pd.concat([
            tornado[['Parameter', 'EV_start']].rename(columns={'EV_start': 'EV'}).assign(Bound='Start'),
            tornado[['Parameter', 'EV_end']].rename(columns={'EV_end': 'EV'}).assign(Bound='End')
        ])
        bar = alt.Chart(tornado_long).mark_bar().encode(
            x=alt.X('EV:Q', title='EV'),
            y=alt.Y('Parameter:N', sort='-x'),
            color='Bound:N',
            tooltip=['Parameter', 'Bound', alt.Tooltip('EV:Q', format='.2f')]
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)
        st.dataframe(tornado.style.format({'EV_start': '{:.2f}', 'EV_end': '{:.2f}', 'Range': '{:.2f}'}), use_container_width=True)

    with sim_tabs[5]:
        scenario = result['scenario_df']
        chart = alt.Chart(scenario).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X('Scenario:N', sort='-y', title='Scenario'),
            y=alt.Y('EV:Q', title='Expected Value'),
            color=alt.condition(alt.datum.EV > 0, alt.value('#1f9d55'), alt.value('#c0392b')),
            tooltip=['Scenario', alt.Tooltip('EV:Q', format='.2f'), alt.Tooltip('P(yes):Q', format='.2%'), 'Gain', 'Loss']
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(scenario.style.format({'EV': '{:.2f}', 'P(yes)': '{:.2%}'}), use_container_width=True)

    with sim_tabs[6]:
        robust = result['robustness_df']
        heat = robust.pivot_table(index='Gain', columns='Loss', values='EV')
        heat_long = heat.reset_index().melt(id_vars='Gain', var_name='Loss', value_name='EV')
        chart = alt.Chart(heat_long).mark_rect().encode(
            x=alt.X('Loss:O', title='Loss'),
            y=alt.Y('Gain:O', title='Gain'),
            color=alt.Color('EV:Q', scale=alt.Scale(scheme='redyellowgreen')),
            tooltip=['Gain', 'Loss', alt.Tooltip('EV:Q', format='.2f')]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(robust.style.format({'P(yes)': '{:.2%}', 'EV': '{:.2f}'}), use_container_width=True)

# TAB 4
with main_tabs[3]:
    st.subheader('Download / Data Output')

    d1, d2, d3 = st.columns(3)
    d1.download_button('Download Decision Under Risk (CSV)', data=result['df_augmented'].to_csv(index=False).encode('utf-8'), file_name='decision_under_risk.csv', mime='text/csv')
    d2.download_button('Download Probabilistic Modeling (CSV)', data=result['df_prob'].to_csv(index=False).encode('utf-8'), file_name='probabilistic_modeling.csv', mime='text/csv')
    d3.download_button('Download Simulation & Sensitivity (CSV)', data=result['scenario_df'].to_csv(index=False).encode('utf-8'), file_name='simulation_sensitivity.csv', mime='text/csv')

    st.markdown('#### Data preview')
    preview_tabs = st.tabs(['Decision', 'Probability', 'Risk Model', 'Scenarios'])
    with preview_tabs[0]:
        st.dataframe(result['df_augmented'].head(20), use_container_width=True)
    with preview_tabs[1]:
        st.dataframe(result['df_prob'].head(20), use_container_width=True)
    with preview_tabs[2]:
        st.dataframe(result['risk_model'].head(20), use_container_width=True)
    with preview_tabs[3]:
        st.dataframe(result['scenario_df'], use_container_width=True)
