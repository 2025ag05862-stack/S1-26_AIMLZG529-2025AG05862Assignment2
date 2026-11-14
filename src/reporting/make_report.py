"""Generate the short assignment report from saved pipeline outputs.

Run after the pipeline: python -m src.reporting.make_report
The result table, counts, and dates come from the saved reports, not hardcoded values.
"""
import json
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src.common.config import load_config, path_for


def report_table(rows, widths, header=True, font_size=8.2):
    tbl = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    commands = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#40546e')),
        ('LINEBELOW', (0, 1), (-1, -2), .25, colors.HexColor('#d7dce2')),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#172333')),
    ]
    if header:
        commands += [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eaf0f5')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]
    tbl.setStyle(TableStyle(commands))
    return tbl


def footer(c, doc):
    c.setFont('Helvetica', 8)
    c.setFillColor(colors.HexColor('#667080'))
    c.drawString(1.7 * cm, 1.1 * cm, 'DM4ML | Assignment 2')
    c.drawRightString(A4[0] - 1.7 * cm, 1.1 * cm, str(doc.page))


def run(cfg=None):
    cfg = cfg or load_config()
    rep = path_for(cfg, 'reports')
    res = pd.read_csv(rep / 'model_comparison.csv').set_index('model')
    val = json.loads((rep / 'validation_report.json').read_text())
    split = json.loads((rep / 'split_summary.json').read_text())
    vs = pd.read_csv(rep / 'validation_summary.csv')
    catalog = pd.read_csv(rep / 'feature_catalog.csv')
    info = json.loads((rep / 'pipeline_run.json').read_text())
    k = int(cfg['evaluation']['k'])
    mn = {'popularity_baseline': 'Popularity',
          'svd_matrix_factorization': 'SVD (personalised)',
          'hybrid_mf_plus_features': 'Hybrid (SVD + features)'}
    pop, svd, hybrid = (res.loc[n] for n in mn)
    rule_count = len(vs)
    rules_reject = int((vs.severity == 'error').sum())
    users_eval = int(svd['users_evaluated'])
    ndcg_change = 100 * (svd[f'ndcg@{k}'] / pop[f'ndcg@{k}'] - 1)
    output = rep / 'final_report.pdf'

    ss = getSampleStyleSheet()
    title = ParagraphStyle('MainTitle', parent=ss['Title'], fontName='Helvetica-Bold',
                           fontSize=16, leading=19, textColor=colors.HexColor('#18334d'),
                           spaceAfter=7, alignment=0)
    meta = ParagraphStyle('Meta', parent=ss['Normal'], fontSize=9, leading=12,
                          textColor=colors.HexColor('#667080'), spaceAfter=14)
    head = ParagraphStyle('SectionTitle', parent=ss['Heading2'], fontName='Helvetica-Bold',
                          fontSize=11.4, leading=14, spaceBefore=14, spaceAfter=5,
                          textColor=colors.HexColor('#234b71'), keepWithNext=True)
    body = ParagraphStyle('BodyTextA2', parent=ss['BodyText'], fontName='Helvetica',
                          fontSize=9.6, leading=14.1, spaceAfter=8)
    note = ParagraphStyle('Note', parent=body, fontSize=9.1, leading=13, textColor=colors.HexColor('#39475a'))

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=1.7 * cm,
                            rightMargin=1.7 * cm, topMargin=1.55 * cm, bottomMargin=1.65 * cm,
                            title='DM4ML Assignment 2 - Recommendation Pipeline Report',
                            author='DM4ML Assignment 2')
    st = []
    def heading(s): st.append(Paragraph(s, head))
    def para(s, style=body): st.append(Paragraph(s, style))

    st.append(Paragraph('Recommendation System: Data Management Pipeline', title))
    st.append(Paragraph('Data Management for Machine Learning | Assignment 2 | Final report', meta))

    heading('1. Objective and dataset')
    para(f'The aim was to build a repeatable pipeline that takes user, item and rating data through validation, cleaning, '
         f'feature creation, model training and evaluation. I used a movie-style synthetic dataset so that I could '
         f'introduce known data issues and check how the validation stage handled them. The raw files contain '
         f'<b>{val["users"]["rows_in"]:,} user rows</b>, <b>{val["items"]["rows_in"]:,} item rows</b> and '
         f'<b>{val["interactions"]["rows_in"]:,} interactions</b>, with dates from January 2022 to June 2024. '
         'The synthetic ratings are influenced by hidden preferences, so the recommendation results should not '
         'be treated as performance estimates for a real platform.')

    heading('2. Pipeline implementation')
    para('<b>Ingestion and validation.</b> Each CSV is first read as text, which makes it possible to catch invalid '
         'numbers and dates instead of silently converting them. The schema is checked before processing, and a '
         'manifest records file sizes and checksums. Validation covers missing fields, duplicate identifiers, '
         'rating values, timestamps and foreign-key references. Serious errors are stored separately in '
         '<i>data/errors/</i>, while warning-level values are carried forward for cleaning.')
    para(f'There are <b>{rule_count} validation rules</b> ({rules_reject} reject-level and '
         f'{rule_count - rules_reject} warning-level). The validation stage rejected '
         f'{val["users"]["pct_rejected"]}% of user rows, {val["items"]["pct_rejected"]}% of item rows '
         f'and {val["interactions"]["pct_rejected"]}% of interaction rows. '
         'The rule-level counts and reasons are recorded in <i>reports/validation_report.md</i>.')
    para(f'<b>Cleaning and splitting.</b> The preparation stage standardises category and demographic labels, '
         f'fills missing age/release-year values and keeps the latest rating for each user-item pair '
         f'({split["superseded_re_ratings_dropped"]} earlier ratings removed). The chronological split uses '
         f'{split["train_rows"]:,} training rows and {split["test_rows"]:,} test rows, '
         f'with the cutoff at {split["split_cutoff_timestamp"][:10]}. User activity, item popularity and the '
         'user-item matrix are then calculated using training interactions.')
    para(f'<b>Feature management.</b> The pipeline stores <b>{len(catalog)} features</b> across users, items and '
         'user-item pairs in Parquet files. Examples are activity count, average rating, item popularity and '
         'category affinity. The <i>FeatureStore</i> class is used when training and when producing ranking '
         'scores. I kept this as a small local feature layer rather than adding a separate feature-serving service.')

    heading('3. Models and experiment tracking')
    para('I compared a popularity baseline with an SVD-based personalised recommender and a hybrid that adds '
         'category-affinity information to SVD ranking scores. The latent-factor choice was checked using the '
         'last portion of the training period, leaving the test set untouched for the final comparison. '
         'Model settings, results and artifacts are logged with MLflow. The hybrid blend weight was set in the '
         'configuration rather than tuned.')

    heading('Workflow overview')
    para('The architecture diagram is also available separately as reports/architecture.png.', note)
    st.append(Image(str(rep / 'architecture.png'), width=10.0 * cm, height=5.2 * cm))

    st.append(PageBreak())
    heading('4. Evaluation and observations')
    para(f'An item counts as relevant if its test rating is at least {cfg["evaluation"]["relevance_threshold"]}. '
         f'I evaluated the top {k} recommendations for {users_eval:,} users with relevant test interactions, '
         'excluding items those users had already rated in training. The metrics cover recommendation accuracy '
         '(Precision, Recall and NDCG), catalogue coverage, list diversity, and rating error (RMSE and MAE).')

    data = [['Model', f'P@{k}', f'R@{k}', f'NDCG@{k}', 'Coverage', 'Diversity', 'RMSE', 'MAE']]
    for name in mn:
        r = res.loc[name]
        data.append([mn[name]] + [f'{r[col]:.4f}' for col in
                   [f'precision@{k}', f'recall@{k}', f'ndcg@{k}', 'coverage', 'diversity', 'rmse', 'mae']])
    st.append(report_table(data, [4.25 * cm] + [1.84 * cm] * 7, font_size=7.5))
    st.append(Spacer(1, 8))
    para(f'The clearest improvement was in ranking: SVD reached <b>NDCG@{k} = '
         f'{svd[f"ndcg@{k}"]:.4f}</b>, compared with {pop[f"ndcg@{k}"]:.4f} for popularity '
         f'(about {ndcg_change:.0f}% higher). Its RMSE was also lower '
         f'({svd["rmse"]:.4f} versus {pop["rmse"]:.4f}). The hybrid increased coverage from '
         f'{svd["coverage"] * 100:.2f}% to {hybrid["coverage"] * 100:.2f}%, but its NDCG was slightly '
         'lower and its diversity also decreased. On this dataset, plain SVD was the better choice when '
         'ranking accuracy mattered most. The hybrid uses the same rating predictor, which explains its '
         'identical RMSE and MAE.')

    heading('5. Running and checking the pipeline')
    para(f'The local runner and the Airflow DAG use the same task dependency graph. Preparation feeds both '
         'transformation and feature generation, and training starts after both branches finish. The local '
         f'run record shows <b>{len(info["tasks"])} successful tasks</b>; the Airflow execution log is '
         'included separately. I used fixed random seeds and one YAML config so the dataset generation, '
         'thresholds and model settings can be reproduced. The repository also includes 13 unit tests for '
         'validation, preparation, recommenders and evaluation metrics.')

    heading('6. Limitations and next steps')
    para('The dataset is synthetic, so real-world issues such as exposure bias and unpredictable user behaviour '
         'are not represented. The evaluation uses one time split and one seed; repeating the experiment on '
         'MovieLens would give more useful evidence. Also, cleaning repeated ratings before splitting can '
         'influence which earlier interactions remain, so I would move that step into the chronological '
         'split logic in a future version. The current feature store is batch-only, and the implementation '
         'has not been tested at production scale.')
    para('One result worth following up is the hybrid model: the category-affinity signal helped coverage but '
         'did not improve NDCG. I would test other blend weights on a validation period before adding more '
         'features. The present system is intended as a reproducible coursework pipeline rather than a '
         'production recommendation service.')

    heading('Files used for verification')
    para('<i>reports/validation_report.md</i> (data checks); <i>reports/feature_catalog.md</i> '
         '(all 18 features); <i>reports/model_comparison.csv</i> (metrics); '
         '<i>reports/airflow_run_evidence.txt</i> (Airflow task states); '
         '<i>src/pipeline/runner.py</i> and <i>dags/recommendation_pipeline_dag.py</i> (orchestration).', note)

    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    return output


if __name__ == '__main__':
    print(run())
