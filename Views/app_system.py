import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from io import BytesIO

# === UTILITAS & PERSIAPAN ===

def extract_model_keyword(text): ## Identifikasi Model
    words = str(text).split()
    if len(words) >= 2:
        if words[0].upper() == 'COROLLA' and words[1].upper() == 'CROSS':
            return 'COROLLA CROSS'
        if words[0].upper() == 'GR':
            return ' '.join(words[:2])
        if words[0].upper() == 'YARIS' and words[1].upper() == 'CROSS':
            return 'YARIS CROSS'
        if words[0].upper() == 'INNOVA' and words[1].upper() == 'ZENIX':
            return 'INNOVA ZENIX'
        if words[0].upper() == 'HILUX' and words[1].upper() == 'CAB-CHASSIS':
            return 'HILUX CAB-CHASSIS'
        if words[0].upper() == 'HILUX' and words[1].upper() == 'PICK':
            return 'HILUX PICK'
    return words[0].upper() if words else ''

def group_parts(df, grouping_type="Electrical"): # Function untuk grouping part name
    grouped = defaultdict(list)
    part_names = df['Part Name'].unique().tolist()
    grouping_type = grouping_type.lower()

    for name in part_names:
        original = name.strip()
        cleaned = name.replace('-', ' ')
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = cleaned.lower().strip()
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        tokens = cleaned.split()

        if grouping_type == "textile":
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            tokens = cleaned.split()
            if 'floor' in tokens and 'mat' in tokens:
                is_gr = 'gr' in tokens
                has_fr = 'fr' in tokens or 'front' in tokens
                has_rr = 'rr' in tokens
                if has_fr:
                    base = 'floor mat gr fr' if is_gr else 'floor mat fr'
                elif has_rr:
                    base = 'floor mat gr rr' if is_gr else 'floor mat rr'
                else:
                    base = 'floor mat gr' if is_gr else 'floor mat'
            elif re.search(r',\s*set', name.lower()):
                base = re.split(r',\s*set', name.lower())[0].strip()
                base = re.sub(r'[^\w\s]', ' ', base)
                base = ' '.join(base.split()[:3])
            else:
                base = ' '.join(tokens[:3])
        elif grouping_type == "plastic":
            if re.search(r',\s*set', cleaned):
                base = re.split(r',\s*set', cleaned)[0].strip()
            else:
                cleaned_no_comma = re.sub(r'[^\w\s]', '', cleaned)
                tokens = cleaned_no_comma.split()
                base = ' '.join(tokens[:3])
        elif grouping_type == "multimedia":
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            tokens = cleaned.split()
            if '21cy' in tokens:
                base = ' '.join(tokens[:6])
            elif 'tam' in tokens and 'audio' in tokens and 'kit' in tokens:
                base = 'tam audio kit'
                if 'g' in tokens and 'grade' in tokens:
                    base = 'tam audio kit g grade'
                elif 'v' in tokens and 'grade' in tokens:
                    base = 'tam audio kit v grade'
                elif 'q' in tokens and 'grade' in tokens:
                    base = 'tam audio kit q grade'
            elif 'receiver' in tokens and 'assy' in tokens:
                base = 'receiver assy display' if 'display' in tokens else 'receiver assy'
            else:
                base = ' '.join(tokens[:3])
                
        else:  # Default termasuk electrical, safety
            if 'dvr' in tokens:
                base = 'wire harness dvr' if 'wire' in tokens or 'harness' in tokens else 'dvr'
            else:
                clean = re.sub(r'\([^)]*\)', '', name.replace('-', ' '))
                clean = re.sub(r'[^\w\s]', '', clean).lower().strip()
                base = ' '.join(clean.split()[:2])                

        grouped[base].append(original)
    return dict(grouped)

def Dataframe(dsrp_file, pio_file, segment_file, tooling_file): ## Function data cleaning & integration
    df_dsrp = pd.concat(pd.read_excel(dsrp_file, sheet_name=None).values(), ignore_index=True)
    df_pio = pd.concat(pd.read_excel(pio_file, sheet_name=None).values(), ignore_index=True)
    df_segment = pd.concat(pd.read_excel(segment_file, sheet_name=None).values(), ignore_index=True)
    df_tooling= pd.read_excel(tooling_file)

    exclude_keywords = ['two tone', '(premium color)', 'bitone']
    df_dsrp_filtered = df_dsrp[~df_dsrp['Model Name DSRP'].apply(lambda x: any(k in str(x).lower() for k in exclude_keywords))].copy()

    trim_keywords = ['one tone', 'non premium color', 'monotone color']
    def trim_model_name(name):
        name = str(name).strip().lower()
        for kw in trim_keywords:
            if kw in name:
                name = name[:name.find(kw)]
        name = re.sub(r'\(+\s*$', '', name)
        return name.strip()

    df_dsrp_filtered['Model Name DSRP'] = df_dsrp_filtered['Model Name DSRP'].apply(trim_model_name)
    df_dsrp_filtered['Model Name Clean'] = df_dsrp_filtered['Model Name DSRP'].str.lower().str.strip().str.replace(' +', ' ', regex=True)

    df_pio['Model Type Clean'] = df_pio['Model Type'].str.lower().str.strip().str.replace(' +', ' ', regex=True)
    df_combine_1 = pd.merge(
        df_pio,
        df_dsrp_filtered[['Model Name DSRP', 'Model Name Clean', 'DKI', 'Volume Wholesales']],
        left_on='Model Type Clean',
        right_on='Model Name Clean',
        how='left'
    ).drop(columns=['Model Type Clean', 'Model Name DSRP', 'Model Name Clean'])
    df_combine_1 = df_combine_1.rename(columns={'DKI': 'OTR'})

    model_keywords = df_combine_1['Model Type'].str.replace('Toyota', '', case=False, regex=False).str.strip().apply(extract_model_keyword)
    model_to_segment = dict(zip(df_segment['Model'], df_segment['Segment']))
    model_to_type = dict(zip(df_segment['Model'], df_segment['Type']))
    model_to_margin = dict(zip(df_segment['Model'], df_segment['Total Margin']))

    df_combine_1['Segment'] = model_keywords.map(model_to_segment)
    df_combine_1['Type'] = model_keywords.map(model_to_type)
    df_combine_1['Total Margin'] = model_keywords.map(model_to_margin)

    df_tooling = df_tooling.iloc[2:].reset_index(drop=True)
    df_tooling = df_tooling.drop(df_tooling.columns[0], axis=1)
    df_tooling.columns = df_tooling.iloc[0]
    df_tooling = df_tooling[1:].reset_index(drop=True)
    cols_to_clean = ['Monthly \nVolume\nPlanning','Volume Achievement (%)', 'Total\nTooling Cost', 'Tooling Cost/Unit']
    df_tooling[cols_to_clean] = df_tooling[cols_to_clean].replace({0: np.nan, '-': np.nan})
    df_tooling[cols_to_clean] = df_tooling[cols_to_clean].apply(pd.to_numeric, errors='coerce')

    df_combine = pd.merge(
        df_combine_1,
        df_tooling[['Part Number', 'Monthly \nVolume\nPlanning','Volume Achievement (%)', 'Total\nTooling Cost', 'Tooling Cost/Unit']],
        on='Part Number',
        how='left'  # gunakan 'left' supaya data dari df_combine tetap lengkap)
    )
    df_combine['Volume Achievement (%)'] = df_combine['Volume Achievement (%)'].apply(lambda x: float(round(x*100)) if pd.notna(x) else float('NaN'))
    
    df_combine.columns = df_combine.columns.str.replace('\n', ' ', regex=True)
    
    return df_combine, model_to_margin

# === REKOMENDASI ===

def Recommendation_Program(df, model_to_margin, grouping_type="Electrical"):
    grouped = group_parts(df, grouping_type)
    partname_to_group = {str(part).strip().casefold(): group for group, parts in grouped.items() for part in parts}
    df['Group'] = df['Part Name'].apply(lambda x: partname_to_group.get(str(x).strip().casefold()))

    segment_order = ['A', 'B', 'C', 'D', 'Comm']
    df['Segment'] = pd.Categorical(df['Segment'], categories=segment_order, ordered=True)
    df = df.sort_values(by=['Group', 'Segment', 'Total Margin', 'OTR'])

    final_rows = []
    group_number = 1
    for group_name, df_group in df.groupby('Group'):
        used_focus_keys = set()
        for _, focus_row in df_group.iterrows():
            model = focus_row['Model Type']
            key = (model, group_name)
            if key in used_focus_keys:
                continue

            focus_cost = focus_row['Part Cost']
            focus_otr = focus_row['OTR']
            focus_segment = focus_row['Segment']
            focus_type = focus_row['Type']
            focus_wholesales = focus_row['Volume Wholesales']

            candidates = df_group[
                (df_group['Model Type'] != model) &
                (df_group['Part Cost'] < focus_cost) &
                (
                    ((df_group['Segment'] == focus_segment) & (df_group['OTR'] > focus_otr)) |
                    ((df_group['Segment'] != focus_segment) & (df_group['Type'] == focus_type) & (df_group['OTR'] > focus_otr))
                )
            ]

            if not candidates.empty:
                avg_cost = candidates['Part Cost'].mean()
                gap = focus_cost - avg_cost
                potential_gain = focus_wholesales * gap

                focus_dict = focus_row.to_dict()
                focus_dict['No'] = group_number
                focus_dict['Recommendation (Average Cost Candidate)'] = round(avg_cost)
                focus_dict['Gap (Part Cost-Recommendation)'] = round(gap)
                focus_dict['Potential Gain based on Q1 25 (Volume Wholesales * Gap)'] = round (potential_gain)
                final_rows.append(focus_dict)

                for _, cand in candidates.iterrows():
                    cand_dict = cand.to_dict()
                    cand_dict['No'] = group_number
                    cand_dict['Recommendation (Average Cost Candidate)'] = '-'
                    cand_dict['Gap (Part Cost-Recommendation)'] = '-'
                    cand_dict['Potential Gain based on Q1 25 (Volume Wholesales * Gap)'] = '-'
                    final_rows.append(cand_dict)

                used_focus_keys.add(key)
                group_number += 1

    df_recom = pd.DataFrame(final_rows)

    fokus_df = df_recom[df_recom['Recommendation (Average Cost Candidate)'] != '-'].copy()
    fokus_df['Segment'] = pd.Categorical(fokus_df['Segment'], categories=segment_order, ordered=True)
    fokus_df['Total Margin'] = fokus_df['Model Type'].apply(lambda x: model_to_margin.get(extract_model_keyword(x), np.nan))
    fokus_df['Gap (Part Cost-Recommendation)'] = pd.to_numeric(fokus_df['Gap (Part Cost-Recommendation)'], errors='coerce')
    fokus_df = fokus_df.sort_values(by=['Segment', 'Total Margin', 'Potential Gain based on Q1 25 (Volume Wholesales * Gap)'], ascending=[True, True, False])
    fokus_df['Rank'] = range(1, len(fokus_df) + 1)

    df_recom = df_recom.merge(fokus_df[['Model Type', 'Group', 'Rank']], on=['Model Type', 'Group'], how='left')
    df_recom['Rank'] = df_recom.groupby('No')['Rank'].transform('first')
    df_recom = df_recom.sort_values(by=['Rank', 'No'])
    df_recom = df_recom.drop(columns=['No'])
    df_recom['Notes'] = ''  # Tambah kolom kosong

    # Kolom yang ingin dipindahkan ke belakang
    cols_to_move = ['Tooling Cost/Unit', 'Volume Achievement (%)']
    # Sisanya (tanpa dua kolom tadi)
    other_cols = [col for col in df_recom.columns if col not in cols_to_move]
    # Gabungkan: kolom lainnya dulu, baru dua kolom tersebut di akhir
    df_recom = df_recom[other_cols + cols_to_move]
    return df_recom

def Summary_Recommendation_Report(df, grouping_type="Electrical"):
    grouped = group_parts(df, grouping_type)
    partname_to_group = {str(part).strip().casefold(): group for group, parts in grouped.items() for part in parts}
    df['Group'] = df['Part Name'].apply(lambda x: partname_to_group.get(str(x).strip().casefold()))

    segment_order = ['A', 'B', 'C', 'D', 'Comm']
    df['Segment'] = pd.Categorical(df['Segment'], categories=segment_order, ordered=True)

    final_rows = []
    group_number = 1

    for group_name, df_group in df.groupby('Group'):
        df_group_sorted = df_group.sort_values(by='Part Cost')
        lowest_cost = df_group_sorted['Part Cost'].min()

        if any(df_group_sorted['Part Cost'] > lowest_cost):
            focus_models = df_group_sorted[df_group_sorted['Part Cost'] == lowest_cost].copy()
            for _, row in focus_models.iterrows():
                row_dict = row.to_dict()
                row_dict['No'] = group_number
                row_dict['Remarks'] = 'Lowest'
                final_rows.append(row_dict)

            candidates = df_group_sorted[df_group_sorted['Part Cost'] > lowest_cost].copy()
            candidates = candidates.sort_values(by=['Part Cost', 'Segment'])
            for _, cand in candidates.iterrows():
                cand_dict = cand.to_dict()
                cand_dict['No'] = group_number
                cand_dict['Remarks'] = 'Candidate'
                final_rows.append(cand_dict)

            group_number += 1

    df_summary = pd.DataFrame(final_rows)
    df_summary = df_summary.drop(columns=['No'])
    df_summary['Notes'] = ''  # Tambah kolom kosong

    # Kolom yang ingin dipindahkan ke belakang
    cols_to_move = ['Monthly  Volume Planning', 'Volume Achievement (%)', 'Total Tooling Cost', 'Tooling Cost/Unit']
    # Sisanya (tanpa dua kolom tadi)
    other_cols = [col for col in df_summary.columns if col not in cols_to_move]
    # Gabungkan: kolom lainnya dulu, baru dua kolom tersebut di akhir
    df_summary = df_summary[other_cols + cols_to_move]

    # Tambahkan kolom tambahan sesuai grouping_type
    if grouping_type == 'Electrical':
        df_summary['Wire'] = ''
        df_summary['Spec'] = ''
    elif grouping_type == 'Multimedia':
        df_summary['Wire'] = ''
        df_summary['Spec'] = ''
    elif grouping_type == 'Textile':
        df_summary['Dimention'] = ''
        df_summary['Material'] = ''
        df_summary['Spec'] = ''
    elif grouping_type == 'Plastic':
        df_summary['Dimention'] = ''
        df_summary['Material'] = ''
        df_summary['Painting Condition'] = ''
        df_summary['Spec'] = ''
    elif grouping_type == 'Safety':
        df_summary['Part Type'] = ''
        df_summary['Spec'] = ''
    return df_summary

# === PEMBUATAN USER INTERFACE ===

st.set_page_config(page_title="Cost Analysis and Recommendation Program", layout="centered")
st.title("Cost Analysis and Recommendation Program")
st.markdown("### 📂 Upload Required Files")

pio_file = st.file_uploader("📦 PIO Parts Master", type="xlsx", key="pio")
dsrp_file = st.file_uploader("💰 DSRP CAL PA Latest", type="xlsx", key="dsrp")
tooling_file = st.file_uploader("📋 Tooling Report Latest", type="xlsx", key="tooling")
segment_file = st.file_uploader("📋 Segment, Type & Margin Mapping", type="xlsx", key="segment")

if dsrp_file and pio_file and segment_file and tooling_file:
    if st.button("🚀 Generate Reports"):
        with st.spinner("Processing your files. This may take a moment..."):
            df_combine, model_to_margin = Dataframe(dsrp_file, pio_file, segment_file, tooling_file)

            # === DETAILED REPORT ===
            output_detailed = BytesIO()
            with pd.ExcelWriter(output_detailed, engine='xlsxwriter') as writer:
                for jenis in df_combine['Jenis'].dropna().unique():
                    clean_name = jenis.strip().title().replace(' ', '')[:31]
                    subset_df = df_combine[df_combine['Jenis'] == jenis].copy()
                    df_result = Recommendation_Program(subset_df, model_to_margin, grouping_type=jenis)

                    df_result.to_excel(writer, sheet_name=clean_name, index=False)
                    workbook = writer.book
                    worksheet = writer.sheets[clean_name]

                    header_format = workbook.add_format({'bold': True, 'bg_color': '#FFFF99', 'border': 1})
                    regular_format = workbook.add_format({'bg_color': '#DAECF4', 'border': 1})
                    focus_format = workbook.add_format({'bg_color': '#0F243E', 'border': 1, 'font_color': 'white'})
                    number_format = workbook.add_format({'num_format': '#,##0', 'bg_color': '#DAECF4', 'border': 1})
                    number_focus_format = workbook.add_format({'num_format': '#,##0', 'bg_color': '#0F243E', 'border': 1, 'font_color': 'white'})

                    for col_num, col_name in enumerate(df_result.columns):
                        max_len = max(df_result[col_name].astype(str).map(len).max(), len(col_name)) + 2
                        worksheet.set_column(col_num, col_num, max_len)
                        worksheet.write(0, col_num, col_name, header_format)

                    last_col_index = df_result.columns.get_loc('Volume Achievement (%)')
                    for row_num, rec in enumerate(df_result['Recommendation (Average Cost Candidate)'], start=1):
                        row_format = focus_format if rec != '-' else regular_format
                        for col in range(last_col_index + 1):
                            value = df_result.iloc[row_num - 1, col]
                            col_name = df_result.columns[col]
                            is_number_col = col_name in ['Part Cost', 'OTR']

                            if pd.isna(value) or (isinstance(value, float) and not np.isfinite(value)):
                                worksheet.write(row_num, col, '-', row_format)
                            else:
                                if is_number_col:
                                    fmt = number_focus_format if row_format == focus_format else number_format
                                    worksheet.write_number(row_num, col, value, fmt)
                                else:
                                    worksheet.write(row_num, col, value, row_format)


            # === SUMMARY REPORT ===
            output_summary = BytesIO()
            with pd.ExcelWriter(output_summary, engine='xlsxwriter') as writer:
                for jenis in df_combine['Jenis'].dropna().unique():
                    clean_name = jenis.strip().title().replace(' ', '')[:31]
                    subset_df = df_combine[df_combine['Jenis'] == jenis].copy()
                    df_summary = Summary_Recommendation_Report(subset_df.copy(), grouping_type=jenis)

                    df_summary.to_excel(writer, sheet_name=clean_name, index=False)
                    workbook = writer.book
                    worksheet = writer.sheets[clean_name]

                    header_format = workbook.add_format({'bold': True, 'bg_color': '#FFFF99', 'border': 1})
                    regular_format = workbook.add_format({'bg_color': '#0F243E', 'border': 1, 'font_color': 'white'})
                    focus_format = workbook.add_format({'bg_color': '#DAECF4', 'border': 1})
                    number_format = workbook.add_format({'num_format': '#,##0', 'bg_color': '#0F243E', 'border': 1, 'font_color': 'white'})
                    number_focus_format = workbook.add_format({'num_format': '#,##0', 'bg_color': '#DAECF4', 'border': 1})

                    for col_num, col_name in enumerate(df_summary.columns):
                        max_len = max(df_summary[col_name].astype(str).map(len).max(), len(col_name)) + 2
                        worksheet.set_column(col_num, col_num, max_len)
                        worksheet.write(0, col_num, col_name, header_format)

                    for row_num, rec in enumerate(df_summary['Remarks'], start=1):
                        row_format = focus_format if rec == 'Lowest' else regular_format
                        for col in range(len(df_summary.columns)):
                            value = df_summary.iloc[row_num - 1, col]
                            col_name = df_summary.columns[col]
                            is_number_col = col_name in ['Part Cost', 'OTR']

                            if pd.isna(value) or (isinstance(value, float) and not np.isfinite(value)):
                                worksheet.write(row_num, col, '-', row_format)
                            else:
                                if is_number_col:
                                    fmt = number_focus_format if row_format == focus_format else number_format
                                    worksheet.write_number(row_num, col, value, fmt)
                                else:
                                    worksheet.write(row_num, col, value, row_format)


            st.success("✅ Reports Generated!")
            
            st.download_button("📥 Download Summary Recommendation Report", data=output_summary.getvalue(), file_name="Summary Recommendation Report.xlsx")
            st.download_button("📥 Download Detailed Recommendation Report", data=output_detailed.getvalue(), file_name="Detailed Recommendation Report.xlsx")

else:
    st.info("Please upload all required files to enable report generation.")
