import numpy as np
import pandas as pd
from typing import Any, Dict, List


def calculate_ev(p_yes: float, gain_yes: float, loss_no: float) -> float:
    return (p_yes * gain_yes) + ((1 - p_yes) * loss_no)


def _ensure_deposit(df: pd.DataFrame) -> pd.DataFrame:
    if 'deposit' not in df.columns:
        raise KeyError("Dataset does not contain 'deposit' column")
    out = df.copy()
    out['deposit'] = out['deposit'].astype(str).str.lower().str.strip()
    return out


def _mean_yes(series: pd.Series) -> float:
    return (series.astype(str).str.lower().str.strip() == 'yes').mean()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def risk_category(p: float) -> str:
    if pd.isna(p):
        return 'Unknown'
    if p < 0.3:
        return 'Sangat Tinggi (P<30%)'
    if p < 0.45:
        return 'Tinggi (30-45%)'
    if p < 0.55:
        return 'Sedang (45-55%)'
    if p < 0.7:
        return 'Rendah (55-70%)'
    return 'Sangat Rendah (≥70%)'


def prediction_category(p: float) -> str:
    if p > 0.6:
        return 'PRIORITAS'
    if p > 0.45:
        return 'TARGET'
    return 'KURANG PRIORITAS'


def decision_under_risk(df: pd.DataFrame, gain_yes: float = 100, loss_no: float = -20) -> Dict[str, Any]:
    df = _ensure_deposit(df)
    res: Dict[str, Any] = {}

    p_yes = _mean_yes(df['deposit'])
    p_no = 1 - p_yes
    ev_all = calculate_ev(p_yes, gain_yes, loss_no)

    res['p_yes'] = p_yes
    res['p_no'] = p_no
    res['ev_all'] = ev_all

    # Segment tables
    def _segment_table(col: str, label: str) -> pd.DataFrame:
        if col not in df.columns:
            return pd.DataFrame(columns=[label, 'P(yes)', 'EV', 'Count'])
        rows: List[Dict[str, Any]] = []
        for value in df[col].dropna().unique():
            mask = df[col] == value
            p = _mean_yes(df.loc[mask, 'deposit'])
            rows.append({label: value, 'P(yes)': p, 'EV': calculate_ev(p, gain_yes, loss_no), 'Count': int(mask.sum())})
        if not rows:
            return pd.DataFrame(columns=[label, 'P(yes)', 'EV', 'Count'])
        return pd.DataFrame(rows).sort_values('EV', ascending=False).reset_index(drop=True)

    df_job = _segment_table('job', 'Job')
    df_edu = _segment_table('education', 'Education')
    df_housing = _segment_table('housing', 'Housing')

    if 'balance' in df.columns:
        balance_series = _safe_numeric(df['balance'])
        df = df.copy()
        df['balance'] = balance_series
        df['balance_category'] = pd.cut(
            df['balance'],
            bins=[-np.inf, 0, 500, 1000, 2000, 5000, np.inf],
            labels=['negatif', '0-500', '500-1000', '1000-2000', '2000-5000', '>5000']
        )
        rows = []
        for value in df['balance_category'].dropna().unique():
            mask = df['balance_category'] == value
            p = _mean_yes(df.loc[mask, 'deposit'])
            rows.append({'Balance': str(value), 'P(yes)': p, 'EV': calculate_ev(p, gain_yes, loss_no), 'Count': int(mask.sum())})
        df_balance = pd.DataFrame(rows).sort_values('EV', ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=['Balance', 'P(yes)', 'EV', 'Count'])
    else:
        df_balance = pd.DataFrame(columns=['Balance', 'P(yes)', 'EV', 'Count'])

    # Augmented df
    out = df.copy()
    if 'job' in out.columns:
        job_prob_map = out.groupby('job')['deposit'].apply(_mean_yes).to_dict()
        out['p_yes_by_job'] = out['job'].map(job_prob_map)
        out['ev_by_job'] = calculate_ev(out['p_yes_by_job'], gain_yes, loss_no)
        out['recommend_by_job'] = out['ev_by_job'] > 0
    else:
        out['p_yes_by_job'] = np.nan
        out['ev_by_job'] = np.nan
        out['recommend_by_job'] = False

    if 'education' in out.columns:
        edu_prob_map = out.groupby('education')['deposit'].apply(_mean_yes).to_dict()
        out['p_yes_by_edu'] = out['education'].map(edu_prob_map)
        out['ev_by_edu'] = calculate_ev(out['p_yes_by_edu'], gain_yes, loss_no)
        out['recommend_by_edu'] = out['ev_by_edu'] > 0
    else:
        out['p_yes_by_edu'] = np.nan
        out['ev_by_edu'] = np.nan
        out['recommend_by_edu'] = False

    if {'job', 'education'}.issubset(out.columns):
        out['p_yes_combined'] = out.groupby(['job', 'education'])['deposit'].transform(_mean_yes)
        out['ev_combined'] = calculate_ev(out['p_yes_combined'], gain_yes, loss_no)
        out['recommend_combined'] = out['ev_combined'] > 0
    else:
        out['p_yes_combined'] = np.nan
        out['ev_combined'] = np.nan
        out['recommend_combined'] = False

    total_ev_all = ev_all * len(out)
    total_ev_targeting = out.loc[out['recommend_combined'], 'ev_combined'].sum() if 'recommend_combined' in out.columns else 0

    res.update({
        'df_job': df_job,
        'df_edu': df_edu,
        'df_housing': df_housing,
        'df_balance': df_balance,
        'df_augmented': out,
        'total_ev_all': total_ev_all,
        'total_ev_targeting': total_ev_targeting,
        'recommend_counts': out['recommend_combined'].value_counts() if 'recommend_combined' in out.columns else pd.Series(dtype=int),
    })
    return res


def probabilistic_modeling(df: pd.DataFrame) -> Dict[str, Any]:
    df = _ensure_deposit(df)
    res: Dict[str, Any] = {}

    p_yes = _mean_yes(df['deposit'])
    p_no = 1 - p_yes
    res['p_yes'] = p_yes
    res['p_no'] = p_no

    def cond_table(col: str, label: str) -> pd.DataFrame:
        if col not in df.columns:
            return pd.DataFrame(columns=[label, 'P(yes|cond)', 'Count'])
        rows = []
        for value in df[col].dropna().unique():
            mask = df[col] == value
            rows.append({label: value, 'P(yes|cond)': _mean_yes(df.loc[mask, 'deposit']), 'Count': int(mask.sum())})
        return pd.DataFrame(rows).sort_values('P(yes|cond)', ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=[label, 'P(yes|cond)', 'Count'])

    res['job_prob'] = cond_table('job', 'Job')
    res['education_prob'] = cond_table('education', 'Education')
    res['housing_prob'] = cond_table('housing', 'Housing')
    res['loan_prob'] = cond_table('loan', 'Loan')

    # Joint probabilities
    res['joint_job_deposit'] = pd.crosstab(df['job'], df['deposit'], normalize='all') if 'job' in df.columns else pd.DataFrame()
    res['joint_edu_deposit'] = pd.crosstab(df['education'], df['deposit'], normalize='all') if 'education' in df.columns else pd.DataFrame()

    # Continuous statistics
    continuous_vars = [c for c in ['age', 'balance', 'duration'] if c in df.columns]
    continuous_stats = []
    for var in continuous_vars:
        series = _safe_numeric(df[var]).dropna()
        if series.empty:
            continue
        continuous_stats.append({
            'Variable': var,
            'Mean': series.mean(),
            'Median': series.median(),
            'Std': series.std(),
            'Skewness': series.skew(),
            'Kurtosis': series.kurtosis(),
            'Min': series.min(),
            'Max': series.max(),
        })
    res['continuous_stats'] = pd.DataFrame(continuous_stats)

    # Discrete distributions
    discrete_vars = [c for c in ['job', 'education', 'marital', 'housing', 'loan'] if c in df.columns]
    discrete_tables = {}
    for var in discrete_vars:
        dist = df[var].value_counts(normalize=True).reset_index()
        dist.columns = [var, 'Probability']
        dist['Count'] = df[var].value_counts().values
        discrete_tables[var] = dist
    res['discrete_tables'] = discrete_tables

    # Risk model
    if {'job', 'education', 'balance'}.issubset(df.columns):
        risk_model = df.groupby(['job', 'education']).agg(
            p_deposit_yes=('deposit', _mean_yes),
            avg_age=('age', 'mean') if 'age' in df.columns else ('deposit', 'size'),
            avg_balance=('balance', 'mean'),
        ).reset_index()
        if 'age' not in df.columns:
            risk_model = risk_model.rename(columns={'avg_age': 'segment_size'})
        risk_model['risk_level'] = risk_model['p_deposit_yes'].apply(risk_category)
        risk_model['risk_score'] = 1 - risk_model['p_deposit_yes']
    else:
        risk_model = pd.DataFrame(columns=['job', 'education', 'p_deposit_yes', 'avg_age', 'avg_balance', 'risk_level', 'risk_score'])
    res['risk_model'] = risk_model

    # Prediction engine, if data available
    base_prob = p_yes
    job_prob_dict = df.groupby('job')['deposit'].apply(_mean_yes).to_dict() if 'job' in df.columns else {}
    edu_prob_dict = df.groupby('education')['deposit'].apply(_mean_yes).to_dict() if 'education' in df.columns else {}
    housing_prob_dict = df.groupby('housing')['deposit'].apply(_mean_yes).to_dict() if 'housing' in df.columns else {}

    def predict_deposit_probability(job, education, balance, housing):
        job_p = job_prob_dict.get(job, base_prob)
        edu_p = edu_prob_dict.get(education, base_prob)
        housing_p = housing_prob_dict.get(housing, base_prob)
        final_prob = (job_p * 0.4) + (edu_p * 0.3) + (housing_p * 0.3)
        try:
            balance = float(balance)
        except Exception:
            balance = 0.0
        if balance > 5000:
            final_prob = min(final_prob * 1.1, 1.0)
        elif balance < 0:
            final_prob = final_prob * 0.8
        return float(np.clip(final_prob, 0, 1))

    res['prediction_function'] = predict_deposit_probability

    # Build example predictions like notebook, but only for columns we have
    test_customers = [
        {'job': 'student', 'education': 'tertiary', 'balance': 3000, 'housing': 'no'},
        {'job': 'blue-collar', 'education': 'secondary', 'balance': 500, 'housing': 'yes'},
        {'job': 'retired', 'education': 'primary', 'balance': 2000, 'housing': 'no'},
        {'job': 'admin.', 'education': 'tertiary', 'balance': 1000, 'housing': 'yes'},
    ]
    pred_rows = []
    for customer in test_customers:
        prob = predict_deposit_probability(customer['job'], customer['education'], customer['balance'], customer['housing'])
        pred_rows.append({**customer, 'P(deposit=yes)': prob, 'Category': prediction_category(prob), 'Risk': risk_category(prob)})
    res['test_predictions'] = pd.DataFrame(pred_rows)

    # DSS output for actual customers
    out = df.copy()
    if {'job', 'education', 'balance', 'housing'}.issubset(out.columns):
        out['p_deposit_pred'] = out.apply(lambda x: predict_deposit_probability(x['job'], x['education'], x['balance'], x['housing']), axis=1)
        out['risk_category'] = out['p_deposit_pred'].apply(risk_category)
        out['recommendation'] = out['p_deposit_pred'].apply(prediction_category)
    else:
        out['p_deposit_pred'] = np.nan
        out['risk_category'] = 'Unknown'
        out['recommendation'] = 'Unknown'
    res['df_prob'] = out

    return res


def simulation_sensitivity(df: pd.DataFrame, gain_yes: float = 100, loss_no: float = -20, n_simulations: int = 10000) -> Dict[str, Any]:
    df = _ensure_deposit(df)
    p_yes = _mean_yes(df['deposit'])
    p_no = 1 - p_yes
    n_customers = len(df)

    sim_deposit_rate = np.random.binomial(n_customers, p_yes, size=n_simulations) / n_customers
    sim_ev = np.array([
        ((np.random.binomial(n_customers, p_yes) * gain_yes) + ((n_customers - np.random.binomial(n_customers, p_yes)) * loss_no)) / n_customers
        for _ in range(n_simulations)
    ])

    ev_deterministic = calculate_ev(p_yes, gain_yes, loss_no)
    be_threshold = abs(loss_no) / (gain_yes - loss_no) if (gain_yes - loss_no) != 0 else np.nan

    # Sensitivity matrix gain vs loss
    gain_range = np.arange(50, 201, 25)
    loss_range = np.arange(-100, -10, 10)
    sensitivity_matrix = np.zeros((len(gain_range), len(loss_range)))
    for i, gain in enumerate(gain_range):
        for j, loss in enumerate(loss_range):
            sensitivity_matrix[i, j] = calculate_ev(p_yes, gain, loss)
    sensitivity_df = pd.DataFrame(sensitivity_matrix, index=gain_range, columns=loss_range)

    # EV vs probability sensitivity
    p_range = np.arange(0.1, 0.9, 0.05)
    ev_by_prob = [calculate_ev(p, gain_yes, loss_no) for p in p_range]

    # Segment sensitivity (job)
    if 'job' in df.columns:
        job_probs = df.groupby('job')['deposit'].apply(_mean_yes).sort_values(ascending=False)
    else:
        job_probs = pd.Series(dtype=float)
    loss_scenarios = [-20, -50, -80, -100]
    scenario_labels = ['Optimis (Loss=20)', 'Moderat (Loss=50)', 'Pesimis (Loss=80)', 'Kritis (Loss=100)']
    job_sensitivity = []
    if not job_probs.empty:
        for job, p in job_probs.items():
            row = {'Job': job, 'P(yes)': p}
            for loss in loss_scenarios:
                row[f'EV_Loss_{abs(loss)}'] = calculate_ev(p, gain_yes, loss)
            job_sensitivity.append(row)
    job_sensitivity_df = pd.DataFrame(job_sensitivity).sort_values('P(yes)', ascending=False) if job_sensitivity else pd.DataFrame()

    # Tornado ranking
    parameters = {
        'Gain (Deposit)': gain_yes,
        'Loss (No Deposit)': abs(loss_no),
        'Probabilitas (P)': p_yes,
        'Jumlah Nasabah': n_customers,
    }
    tornado_rows = []
    base_ev = ev_deterministic
    for param_name, base_value in parameters.items():
        if param_name == 'Gain (Deposit)':
            low_ev = calculate_ev(p_yes, base_value * 0.8, loss_no)
            high_ev = calculate_ev(p_yes, base_value * 1.2, loss_no)
        elif param_name == 'Loss (No Deposit)':
            low_ev = calculate_ev(p_yes, gain_yes, -(base_value * 0.8))
            high_ev = calculate_ev(p_yes, gain_yes, -(base_value * 1.2))
        elif param_name == 'Probabilitas (P)':
            low_p = max(min(base_value * 0.8, 1), 0)
            high_p = max(min(base_value * 1.2, 1), 0)
            low_ev = calculate_ev(low_p, gain_yes, loss_no)
            high_ev = calculate_ev(high_p, gain_yes, loss_no)
        else:
            low_ev = base_ev * 0.8
            high_ev = base_ev * 1.2
        tornado_rows.append({'Parameter': param_name, 'EV_start': min(low_ev, high_ev), 'EV_end': max(low_ev, high_ev), 'Range': abs(high_ev - low_ev)})
    tornado_df = pd.DataFrame(tornado_rows).sort_values('Range', ascending=False).reset_index(drop=True)

    # Scenarios
    scenarios = {
        'Pessimistic': {'gain': 50, 'loss': -80, 'p': p_yes * 0.7},
        'Moderate': {'gain': 100, 'loss': -50, 'p': p_yes},
        'Optimistic': {'gain': 150, 'loss': -20, 'p': min(p_yes * 1.3, 0.95)},
        'Best Case': {'gain': 200, 'loss': -10, 'p': min(p_yes * 1.5, 0.95)},
        'Worst Case': {'gain': 30, 'loss': -100, 'p': p_yes * 0.5},
    }
    scenario_rows = []
    for name, params in scenarios.items():
        ev = calculate_ev(params['p'], params['gain'], params['loss'])
        scenario_rows.append({'Scenario': name, 'EV': ev, 'P(yes)': params['p'], 'Gain': params['gain'], 'Loss': params['loss']})
    scenario_df = pd.DataFrame(scenario_rows)

    # Robustness check
    p_test = np.arange(0.3, 0.7, 0.05)
    gain_test = [80, 100, 120]
    loss_test = [-30, -20, -10]
    robustness_rows = []
    for p in p_test:
        for gain in gain_test:
            for loss in loss_test:
                ev = calculate_ev(p, gain, loss)
                robustness_rows.append({'P(yes)': p, 'Gain': gain, 'Loss': loss, 'EV': ev, 'Decision': 'Offer' if ev > 0 else 'Not Offer'})
    robustness_df = pd.DataFrame(robustness_rows)

    return {
        'p_yes': p_yes,
        'p_no': p_no,
        'sim_deposit_rate': sim_deposit_rate,
        'sim_ev': sim_ev,
        'ev_deterministic': ev_deterministic,
        'be_threshold': be_threshold,
        'gain_range': gain_range,
        'loss_range': loss_range,
        'sensitivity_df': sensitivity_df,
        'p_range': p_range,
        'ev_by_prob': ev_by_prob,
        'job_probs': job_probs,
        'job_sensitivity_df': job_sensitivity_df,
        'tornado_df': tornado_df,
        'scenario_df': scenario_df,
        'robustness_df': robustness_df,
        'gain_yes': gain_yes,
        'loss_no': loss_no,
        'n_customers': n_customers,
    }


def dynamic_segment_comparison(df: pd.DataFrame, dimension: str = 'job', top_n: int = 5, gain_yes: float = 100, loss_no: float = -20) -> pd.DataFrame:
    """Create a comparison table for the requested dimension.

    Used by Streamlit sliders to compare top-N segments dynamically.
    """
    df = _ensure_deposit(df)
    if dimension not in df.columns:
        return pd.DataFrame(columns=[dimension, 'P(yes)', 'EV', 'Count'])
    rows = []
    for value in df[dimension].dropna().unique():
        mask = df[dimension] == value
        p = _mean_yes(df.loc[mask, 'deposit'])
        rows.append({dimension: value, 'P(yes)': p, 'EV': calculate_ev(p, gain_yes, loss_no), 'Count': int(mask.sum())})
    if not rows:
        return pd.DataFrame(columns=[dimension, 'P(yes)', 'EV', 'Count'])
    table = pd.DataFrame(rows).sort_values('EV', ascending=False).reset_index(drop=True)
    return table.head(top_n)


def stress_summary(p_yes: float, gain_yes: float, loss_no: float, probability_shift_pct: float = 0.0, gain_shift_pct: float = 0.0, loss_shift_pct: float = 0.0) -> Dict[str, float]:
    """Return a simple stress-test EV summary under slider-controlled perturbations."""
    stressed_p = float(np.clip(p_yes * (1 + probability_shift_pct), 0, 1))
    stressed_gain = gain_yes * (1 + gain_shift_pct)
    stressed_loss = loss_no * (1 + loss_shift_pct)
    return {
        'stressed_p_yes': stressed_p,
        'stressed_gain': stressed_gain,
        'stressed_loss': stressed_loss,
        'stressed_ev': calculate_ev(stressed_p, stressed_gain, stressed_loss),
        'base_ev': calculate_ev(p_yes, gain_yes, loss_no),
    }


def full_analysis(df: pd.DataFrame, gain_yes: float = 100, loss_no: float = -20, n_simulations: int = 10000) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    result.update(decision_under_risk(df, gain_yes=gain_yes, loss_no=loss_no))
    result.update(probabilistic_modeling(df))
    result.update(simulation_sensitivity(df, gain_yes=gain_yes, loss_no=loss_no, n_simulations=n_simulations))
    return result
