import streamlit as st
import pandas as pd
import operator
import tabula
import numpy as np
from tabulate import tabulate
from PyPDF2 import PdfReader
from pandas.api.types import is_numeric_dtype, is_string_dtype
from thefuzz import fuzz
import io
import os
import base64
import requests

KW_THRESHOLD = 65
ORIGIN_THRESHOLD = 0.5

paths = [
    #r"C:\Users\nperkins\Downloads\example_pdfs\example_pdfs\2022-agilent-esg-report.pdf",
    #r"C:\Users\nperkins\Downloads\example_pdfs\example_pdfs\News-Corp-2022-ESG-Report.pdf"
    #r"C:\Users\nperkins\Downloads\example_pdfs\example_pdfs\2022-Bio-Rad-Sustainability-Report.pdf",
    #r"C:\Users\nperkins\Downloads\example_pdfs\example_pdfs\bmy-2022-esg-report.pdf",
    #r"C:\Users\nperkins\Downloads\example_pdfs\example_pdfs\broadridge-sustainability-report-2022.pdf"
]
pdfs = []
year_reps = [y for x in ['19', '20', '21', '22', '23'] for y in ['20' + x, 'FY' + x, '\'' + x, 'FY 20' + x]]
options_col1 = ['Environmental', 'Social', 'Governance']
options_col2 = {'Environmental': ['Energy Total', 'Energy Management', 'GHG', 'Chemical Emissions', 'Waste Management', 'Water', 'Controversy'], 'Social': ['Diversity & Inclusion', 'Gender Equality', 'Health and Safety'], 'Governance': ['Board', 'Employee', 'Turnover', 'Supply Chain', 'Risk Management']}
options_col3 = {"Energy Total": ["Energy", "Renewable", "Electricity", "Heating"], "Energy Management": ["Natural Gas", "Crude Oil", "Diesel", "LPG", "Propane", "Jet Fuel", "Gasoline"], "GHG": ["GHG", "Greenhouse", "Emission", "Scope 1", "Scope 2", "Scope 3"], "Chemical Emissions": ["Sulfur", "SO", "Nitrogen", "NO", "Volatile"], "Waste Management": ["Waste", "Compost", "Recycl", "Landfill", "Disposal", "Hazardous Waste", "Incinerat"], "Water": ["Fresh", "Water", "Ground water", "Surface Water"], "Controversy": ["Controlled", "Hydrocarbon", "Spill", "Barrel"],
                "Diversity & Inclusion": ["Minority", "Race", "Ethnic", "Asian", "African", "Hispanic", "Latinx", "Latino", "Indian", "Native","Multiracial", "Veteran", "Disability", "LGBT"], "Gender Equality": ["Female", "Non-Binary"], "Health and Safety":  ["Injur", "Fatal", "Illness", "Incident"],
                "Board": ["Board"], "Employee": ["Full-time", "Part-time", "Permanent"], "Turnover": ["Turnover"], "Supply Chain": ["Suppl"], "Risk Management": ["Penalt", "Audit"]}
def processQuery(input):
    output = input.replace("Hello", "SELECT")
    data = {"Column1":[1,2,3,4], "Column2":["a","b","c","d"]}
    df = pd.DataFrame(data)
    return output

def outputDataframe():
    data = {"Column1":[1,2,3,4], "Column2":["a","b","c","d"]}
    df = pd.DataFrame(data)
    return df

def process_pdf(path, keywords):
    reader = PdfReader(path)
    kw_data, d_max, page_num, kw_found = find_keywords(reader, keywords)
    kw_full = {kw for cat in keywords for sub_cat in keywords[cat] for kw in keywords[cat][sub_cat]}
    kw_not_found = kw_full.difference(kw_found)
    pages = []
    for cat in d_max:
        for sub_cat in d_max[cat]:
            num_words = len(keywords[cat][sub_cat])
            pages += [f + x[0] if 0 < f + x[0] < page_num else x[0] for x in d_max[cat][sub_cat] if x[1] > num_words / 2
                      for f in [-1, 0, 1]]
    pages = list(dict.fromkeys(pages))
    print('running tabula')
    dfs = tabula.read_pdf(path, pages=list(pages), pandas_options={'header': None}, silent=True)
    print('finished')
    return dfs, kw_not_found


def find_keywords(reader, keywords):
    d = {cat: {sub_cat: {} for sub_cat in keywords[cat]} for cat in keywords}
    kw_found = set()
    for i in range(len(reader.pages)):
        content = reader.pages[i].extract_text()
        for cat in keywords:
            category = keywords[cat]
            for sub_cat in category:
                for kw in category[sub_cat]:
                    if kw.lower() in content.lower():
                        kw_found.add(kw)
                        if i + 1 in d[cat][sub_cat]:
                            d[cat][sub_cat][i + 1] = d[cat][sub_cat][i + 1] + 1
                        else:
                            d[cat][sub_cat][i + 1] = 1
    d_max = {
        cat: {sub_cat: list(sorted(d[cat][sub_cat].items(), key=operator.itemgetter(1), reverse=True)) for sub_cat in
              d[cat]} for cat in d}
    return d, d_max, i, kw_found


def filter_tables(dfs, kws):
    r = []
    kw_list = [kw for cat in kws for sub_cat in cat for kw in sub_cat]
    filter_conds = [
        lambda df: df.shape[0] > 1,
        lambda df: df.shape[1] > 1,
        lambda df: df.applymap(lambda x: True if len(str(x)) < 300 else False).all().all(),
        lambda df: df.isna().sum().sum() < df.count().sum(),
        lambda df: df.apply(
            lambda col: not is_string_dtype(col) or any([col.str.contains(kw).any() for kw in kw_list])).any(),
        lambda df: df.apply(lambda x: is_numeric_dtype(x) or x.str.isnumeric().any()).any()
    ]
    for df in dfs:
        df_passed = True
        for cond in filter_conds:
            if not cond(df):
                df_passed = False
                break
        if df_passed:
            r.append(df)
    return r

def filter_tables_with_meta(dfs, kws):
    r = []
    kw_list = [kw for cat in kws for sub_cat in cat for kw in sub_cat]
    filter_conds = [
        lambda df: df.shape[0] > 1,
        lambda df: df.shape[1] > 1,
        lambda df: df.applymap(lambda x: True if len(str(x)) < 60 else False).all().all(),
        lambda df: df.isna().sum().sum() * 2 < df.count().sum(),
        lambda df: df.apply(
            lambda col: not is_string_dtype(col) or any([col.astype(str).str.contains(kw).any() for kw in kw_list])).any(),
        lambda df: df.apply(lambda x: is_numeric_dtype(x) or x.astype(str).str.isnumeric().any()).any()
    ]
    rejected = []
    for df, meta in dfs:
        df_passed = True
        for cond in filter_conds:
            if not cond(df):
                df_passed = False
                break
        if df_passed:
            r.append((df, meta))
        else:
            rejected.append(df)
    return r


def fuzzy_match_kws(entry, kws):
    ratios = [(kw, fuzz.ratio(entry, kw)) for cat in kws for subcat in kws[cat] for kw in kws[cat][subcat]]
    return max(ratios, key=lambda x: x[1])

def entry_contains_kw(entry, kws):
    return [x for cat in kws for subcat in kws[cat] for x in kws[cat][subcat] if x.lower() in str(entry).lower()]

def find_year(entry):
    for year in year_reps:
        if year in str(entry):
            return True
    return False


def split_by_metric(df, metrics):
    #kws = [x for cat in metrics for subcat in metrics[cat] for x in metrics[cat][subcat]]
    cols_with_metric = []
    meta = {'col_category': {}}
    for name, series in df.items():
        if series.dtype == 'object':
            kw_fuzzy_match = series.apply(lambda x: fuzzy_match_kws(x, metrics))
            kw_entry_contains_match = series.apply(lambda x: entry_contains_kw(x, metrics))
            kw_fuzzy_mask = kw_fuzzy_match.apply(lambda x: x[1] > KW_THRESHOLD)
            kw_entry_mask = kw_entry_contains_match.apply(lambda x: len(x) > 0)

            if kw_fuzzy_mask.any() or kw_entry_mask.any():
                cols_with_metric.append(name)

                fuzz_matched = kw_fuzzy_match[kw_fuzzy_mask].apply(lambda x: x[0])
                entry_matched = kw_entry_contains_match[kw_entry_mask]


                fuzz_cat = {(cat, subcat, match) for cat in metrics for subcat in metrics[cat] for match in fuzz_matched if match in metrics[cat][subcat]}
                match_cat = {(cat, subcat, match) for cat in metrics for subcat in metrics[cat] for match_list in entry_matched for match in match_list if match in metrics[cat][subcat]}
                u = fuzz_cat.union(match_cat)
                cats = {m[0] for m in u}
                subcats = {m[1] for m in u}
                match = {m[2] for m in u}
                meta['col_category'][name] = (cats, subcats, match)
        split_dfs = []

    if len(cols_with_metric) > 0:
        for col_idx in range(len(cols_with_metric) - 1):
            split_dfs.append(df.iloc[:, cols_with_metric[col_idx]:cols_with_metric[col_idx + 1]].dropna(how='all'))
        split_dfs.append(df.iloc[:, cols_with_metric[len(cols_with_metric) - 1]:].dropna(how='all'))
        return split_dfs, meta
    return [], {}

def get_year_if_missing(series):
    year_mask = series.apply(lambda x: find_year(f'{x:.0f}') if series.dtype == 'float64' else find_year(x))
    if year_mask.any() and not year_mask.iloc[0]:
        #print('found missing', year_mask[year_mask].index[0])
        return series.loc[year_mask[year_mask].index[0]]
    return None
def split_rows_by_metric(df):
    #copy years if not found
    #print(df.iloc[: 1:].apply(get_year_if_missing))
    missing_years = df.apply(get_year_if_missing)
    if not missing_years.iloc[1:].isnull().any():
        #print('replacing top row')
        df.loc[-1] = missing_years
        df.index = df.index + 1
        df = df.sort_index()
    has_sub_metric = df.loc[:, 1:].isnull().all(1)
    parent_label = df.iloc[0, 0]
    if has_sub_metric.any():
        dfs = []
        cur_df = df.iloc[:0,:].copy()
        top_row = df.iloc[0,1:].copy()
        for idx, row in df.iterrows():
            if has_sub_metric[idx]:
                dfs.append(cur_df)
                cur_df = df.iloc[:0,:].copy()
                row.iloc[1:] = top_row
                cur_df.loc[0] = row
            else:
                cur_df.loc[len(cur_df.index)] = row
        dfs.append(cur_df)
        return dfs, {'parent_label': parent_label}
    return [df], {}

def extract_year(df):
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    cols_with_year = []
    rows = len(df)
    for name, series in df.items():
        # year_mask = series.apply(lambda x: f'{x:.0f}' if series.dtype == 'float64' else x).apply(find_year)
        year_mask = series.apply(lambda x: f'{x:.0f}' if series.dtype == 'float64' else x).isin(year_reps)

        if year_mask.any():
            cols_with_year.append(name)
            start_row = year_mask[year_mask].index[0]  # ugly trick to get first true value in the series
            rows = start_row if start_row < rows else rows
    if len(cols_with_year) > 0:
        # clean the table
        new_df = df[cols_with_year]
        new_df = new_df.iloc[rows:]
        new_df.loc[:, 0] = df.iloc[:, 0]
        # new_df = new_df.dropna(axis=0, how='all', subset=cols_with_year)
        new_df = new_df.reindex(sorted(new_df.columns), axis=1)
        label = new_df.iloc[0, 0]
        new_df.columns = new_df.iloc[0]
        new_df = new_df.rename(columns={new_df.columns[0]: "Metric"})  # .set_index('Metric')
        new_df = new_df.iloc[1:]  # .dropna(axis=1, how='all')
        # print_tables([new_df])
        return new_df, label
    return None, None


def format_json(json, meta):
    d = {'meta': meta, 'data': {}}
    for i in json:
        metric = json[i]['Metric']
        c = json[i].copy()
        c.pop('Metric')
        d['data'][metric] = c
    return d

def format_for_table(dfs):
    final_df = pd.DataFrame(columns=['File', 'Page', 'Category', 'Label', 'Row', 'Column', 'Value'])
    for path in dfs:
        for pair in dfs[path]:
            df = pair['df']
            meta = pair['meta']
            #print_tables([df])
            shape = df.shape
            if shape[0] < 2 or shape[1] < 2:
                continue
            top_row = df.iloc[0, :]
            label = [df.iloc[0, 0]]
            if 'parent_label' in meta and len(str(meta['parent_label'])) < 50:
                label.append(meta['parent_label'])
            for idx, row in df.iloc[1:].iterrows():
                metric_col = None
                for col in df.columns:
                    if col in meta['col_category']:
                        metric_col = col
                    else:
                        metric = row[metric_col] if metric_col is not None else 'NA'
                        metric_cat = meta['col_category'][metric_col] if metric_col is not None else 'NA'
                        final_df.loc[len(final_df.index)] = [path, meta['page'], metric_cat, label, metric, top_row[col], row[col]]
    final_df = final_df.dropna(subset=['Row', 'Column', 'Value'])
    return final_df
def trace_origin(dfs, path):
    d = []
    included_indexes = set()
    reader = PdfReader(path)
    for i in range(len(reader.pages)):
        page_content = reader.pages[i].extract_text()
        for idx, df_with_meta in enumerate(dfs):
            df = df_with_meta[0]
            meta = df_with_meta[1]
            stack = df.stack()
            tot_count = len(stack)
            found_count = df.stack().astype(str).apply(lambda x: x in page_content).sum()
            if idx not in included_indexes and found_count / tot_count > ORIGIN_THRESHOLD:
                d.append({'df': df, 'meta': dict(meta, page=i+1)})
                included_indexes.add(idx)
    for idx, df_with_meta in enumerate(dfs):
        df = df_with_meta[0]
        meta = df_with_meta[1]
        if idx not in included_indexes:
            d.append({'df': df, 'meta': dict(meta, page=-1)})
    return d


def print_tables(dfs):
    for df in dfs:
        print(tabulate(df, headers='keys', tablefmt='psql'))


def save_dfs_to_excel(dfs, writer, sheet_name):
    start_row = 0
    for df in dfs:
        df['df'].to_excel(writer, sheet_name=sheet_name, startrow=start_row)
        start_row += df['df'].shape[0] + 1
        meta_df = pd.DataFrame(df['meta'], index=[0])
        meta_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row)
        start_row += 3


def save_data_to_excel(data, path):
    #print(data)
    with pd.ExcelWriter(path) as writer:
        for company in data:
            save_dfs_to_excel(data[company], writer, company)


def process(path, kws):
    dfs_found, kws_not_found = process_pdf(path, kws)
    filtered_tables = filter_tables(dfs_found, kws)
    #print('FILTER', len(filtered_tables))
    #print_tables(filtered_tables)
    #print('FILTER DONE')
    dfs = []
    for idx, ft in enumerate(filtered_tables):
        dfs_split, meta = split_by_metric(ft, kws)
        for df in dfs_split:
            second_split, inner_meta = split_rows_by_metric(df)
            for metric_split_df in second_split:
                dfs.append((metric_split_df, {**meta, **inner_meta}))
    fdfs = filter_tables_with_meta(dfs, kws)
    return trace_origin(fdfs, path)
def main():
    st.title("KPI Extraction")
    uploadedFiles = st.file_uploader("Choose PDF Files", accept_multiple_files=True, type='pdf')
    for uploadedFile in uploadedFiles:
        if uploadedFile is not None:
            pdfs.append(uploadedFile)
    if 'selections' not in st.session_state:
        st.session_state.selections = {'col1':None,'col2':None,'col3':[]}
    if 'final_selections' not in st.session_state:
        st.session_state.final_selections = {}
    if 'checkbox_states' not in st.session_state:
        st.session_state.checkbox_states = {}
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.selections['col1'] = st.radio("Choose E/S/G", options_col1)
    with col2:
        if st.session_state.selections['col1']:
            st.session_state.selections['col2'] = st.radio("Choose Category", options_col2[st.session_state.selections['col1']])
    with col3:
        st.write("Choose Keywords")
        col1_choice = st.session_state.selections['col1']
        col2_choice = st.session_state.selections['col2']
        if col1_choice and col2_choice:
            unique_key = f"{col1_choice}_{col2_choice}"
            if unique_key not in st.session_state.checkbox_states:
                st.session_state.checkbox_states[unique_key] = []

            col3_changed = False
            for option in options_col3[col2_choice]:
                checkbox_state = st.checkbox(option, key=f"{unique_key}_{option}", value = option in st.session_state.checkbox_states[unique_key])
                if checkbox_state != (option in st.session_state.checkbox_states[unique_key]):
                    col3_changed = True
                    if checkbox_state:
                        st.session_state.checkbox_states[unique_key].append(option)
                    else:
                        st.session_state.checkbox_states[unique_key].remove(option)
               
            if col3_changed:
                if st.session_state.checkbox_states[unique_key] and len(st.session_state.checkbox_states[unique_key]) != 0:
                    st.session_state.final_selections[unique_key] = st.session_state.checkbox_states[unique_key]
                else:
                    st.session_state.final_selections.pop(unique_key, None)
                    col1, col2 = unique_key.split('_', 1)
                    if not any(key.startswith(f"{col1}_") for key in st.session_state.final_selections):
                        st.session_state.final_selections.pop(col1, None)
                
    
    

    if st.button("Search for KPIs"):
        kpis = {}
        for unique_key, choices in st.session_state.final_selections.items():
            col1_choice, col2_choice = unique_key.split("_", 1)
            if col1_choice not in kpis:
                kpis[col1_choice] = {}
            kpis[col1_choice][col2_choice] = choices
        final_data = {}
        st.write(kpis)
        if not os.path.exists(r"C:\Users\nperkins\Downloads\output_table_strict_add_row.xlsx"):
            #for p in paths:
                #name = p.split('\\')[-1]
                #print(name)
                #final_data[name] = process(p, kpis)
            for p in pdfs:
                print(p)
                with st.spinner("Processing..."):
                    final_data[p.name] = process(p, kpis)
                    df = format_for_table(final_data)
        else:
            oldData = pd.read_excel(r"C:\Users\nperkins\Downloads\output_table_strict_add_row.xlsx")
            stringPDFs = [x.name for x in pdfs]
            if len(oldData) > 0 and oldData['KPIs_Used'].iloc[0] == str(kpis):
                print('KPIs found')
                oldPDFlist = oldData["PDFs_Searched"].iloc[0]
                print(oldPDFlist)
                print(type(oldPDFlist))
                counter = 0
                for p in pdfs:
                    if p.name in oldPDFlist:
                        counter += 1
                if len(pdfs) == counter:
                    df = oldData.drop('KPIs_Used', axis=1)
                    df = df.drop('PDFs_Searched', axis=1)
                    #else if more remove the present pdfs and only search new ones
                    #else if less then filter out non-present ones and display current ones
                else:
                    for p in pdfs:
                        print(p)
                        with st.spinner("Processing..."):
                            final_data[p.name] = process(p, kpis)
                            df = format_for_table(final_data)
            else:
                for p in pdfs:
                    print(p)
                    with st.spinner("Processing..."):
                        final_data[p.name] = process(p, kpis)
                        df = format_for_table(final_data)
        st.dataframe(df)
        df['KPIs_Used'] = str(kpis)
        df['PDFs_Searched'] = str([x.name for x in pdfs])
        df.to_excel('output_table_strict_add_row.xlsx', index=False)
    #Connected Streamlit and Django
        #How does current Django work? Is it good enough to have a link between pages? Data will hopefully be stored on Databricks so it can be shared.
    #Allowed PDF uploads
        #Should we also keep a list of PDFs that we search through every time in addition to user uploads, or only do uploads from user?
    #Looked into Databricks connection, not 100% sure how it works. Databricks connected is installed but needs Auth token, it said I do not have access.
        #Currently saving locally as an excel, if new KPIs are detected, or old ones removed, then Tabula will be run again. Same with PDFs.

    #Should most likely be: If you remove a KPI, filter it out of the table then load the table, not re-writing in case someone wants the other KPI later.
    #If there is a new KPI, should I re-run all KPIs or just pass the new one to the function for Tabular to search for?
    #If you remove a PDF, filter it out of the table then load the table, not re-writing in case someones wants the data later
    #If there is a new PDf, only pass it through to the function to save time, then append to loaded table.
if __name__ == '__main__':
    main()