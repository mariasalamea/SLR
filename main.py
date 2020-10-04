# Developed by Maria Jose Salamea
# salamea@essi.upc.edu

import os
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

tqdm.pandas(desc="Quartil")


# Working Directories
dir_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(dir_path, 'results')
# setup style
sns.set(font_scale=1.2, style="whitegrid",
        palette=sns.palplot(sns.color_palette('Paired')), color_codes=True)


def database_filter(datastructure):
    datastructure = datastructure.filter(['Title',
                    'Abstract',
                    'Keywords',
                    'Authors',
                    'Year',
                    'DocumentType',
                    'PublicationTitle',
                    'DOI',
                    'Link',
                    'Affiliations',
                    'Publisher',
                    'Language',
                    'ISSN',
                    'ISBN'
                    ],
                   axis=1)
    return datastructure


def sort_ieee(df):
    """Index(['Document Title', 'Authors', 'Author Affiliations', 'Publication Title',
       'Date Added To Xplore', 'Publication Year', 'Volume', 'Issue',
       'Start Page', 'End Page', 'Abstract', 'ISSN', 'ISBNs', 'DOI',
       'Funding Information', 'PDF Link', 'Author Keywords', 'IEEE Terms',
       'INSPEC Controlled Terms', 'INSPEC Non-Controlled Terms', 'Mesh_Terms',
       'Article Citation Count', 'Reference Count', 'License', 'Online Date',
       'Issue Date', 'Meeting Date', 'Publisher', 'Document Identifier'],
      dtype='object')
    """
    # Normalize column names
    df.rename(columns={'Document Title': 'Title',
                       'Author Affiliations': 'Affiliations',
                       'Publication Year': 'Year',
                       'Publication Title': 'PublicationTitle',
                       'Author Keywords': 'Keywords',
                       'Document Identifier': 'DocumentType',
                       'PDF Link': 'Link',
                       'ISBNs': 'ISBN'
                       },
              inplace=True)

    # Normalize IEEE DocumentType [IEEE Journals, IEEE Conferences, Review]
    df.loc[df.DocumentType == "IEEE Journals", "DocumentType"] = "Journal"
    df.loc[df.DocumentType == "IEEE Conferences", "DocumentType"] = "Conference"
    df.loc[df.DocumentType == "IEEE Magazines", "DocumentType"] = "Magazine"
    df.loc[df.DocumentType == "IET Conferences", "DocumentType"] = "Conference"
    df.loc[df.DocumentType == "IET Journals", "DocumentType"] = "Journal"

    # Default language English ******
    df.insert(5, 'Language', 'English')

    # Filter and save
    filter_and_save_DB(df, 'IEEE')


def sort_scopus(df):
    """Index(['Authors', 'Author(s) ID', 'Title', 'Year', 'PublicationTitle',
       'Volume', 'Issue', 'Art. No.', 'Page start', 'Page end', 'Page count',
       'Cited by', 'DOI', 'Link', 'Affiliations', 'Authors with affiliations',
       'Abstract', 'AuthorKeywords', 'Index Keywords',
       'Molecular Sequence Numbers', 'Chemicals/CAS', 'Tradenames',
       'Manufacturers', 'Funding Details', 'References',
       'Correspondence Address', 'Editors', 'Sponsors', 'Publisher',
       'Conference name', 'Conference date', 'Conference location',
       'Conference code', 'ISSN', 'ISBN', 'CODEN', 'PubMed ID',
       'Language of Original Document', 'Abbreviated Source Title',
       'DocumentType', 'Publication Stage', 'Access Type', 'Source', 'EID'],
      dtype='object')
    """
    df.rename(columns={'Source title': 'PublicationTitle',
                       'Author Keywords': 'Keywords',
                       'Document Type': 'DocumentType',
                       'Language of Original Document': 'Language'
                       },
              inplace=True)

    # Normalize Scopus DocumentType [Article, Conference Paper, Review, Book Chapter, Book]
    df.loc[df.DocumentType == "Article", "DocumentType"] = "Journal"
    df.loc[df.DocumentType == "Conference Paper", "DocumentType"] = "Conference"

    # Filter and save
    filter_and_save_DB(df, 'Scopus')


def sort_ACM(df):
    """Index(['Key', 'Item Type', 'Publication Year', 'Author', 'Title',
       'Publication Title', 'ISBN', 'ISSN', 'DOI', 'Url', 'Abstract Note',
       'Date', 'Date Added', 'Date Modified', 'Access Date', 'Pages',
       'Num Pages', 'Issue', 'Volume', 'Number Of Volumes',
       'Journal Abbreviation', 'Short Title', 'Series', 'Series Number',
       'Series Text', 'Series Title', 'Publisher', 'Place', 'Language',
       'Rights', 'Type', 'Archive', 'Archive Location', 'Library Catalog',
       'Call Number', 'Extra', 'Notes', 'File Attachments', 'Link Attachments',
       'Manual Tags', 'Automatic Tags', 'Editor', 'Series Editor',
       'Translator', 'Contributor', 'Attorney Agent', 'Book Author',
       'Cast Member', 'Commenter', 'Composer', 'Cosponsor', 'Counsel',
       'Interviewer', 'Producer', 'Recipient', 'Reviewed Author',
       'Scriptwriter', 'Words By', 'Guest', 'Number', 'Edition',
       'Running Time', 'Scale', 'Medium', 'Artwork Size', 'Filing Date',
       'Application Number', 'Assignee', 'Issuing Authority', 'Country',
       'Meeting Name', 'Conference Name', 'Court', 'References', 'Reporter',
       'Legal Status', 'Priority Numbers', 'Programming Language', 'Version',
       'System', 'Code', 'Code Number', 'Section', 'Session', 'Committee',
       'History', 'Legislative Body'],
      dtype='object')
    """

    df.rename(columns={'Author': 'Authors',
                       'Abstract Note': 'Abstract',
                       'Publication Year': 'Year',
                       'Url': 'Link',
                       'Automatic Tags': 'Keywords',
                       'Item Type': 'DocumentType',
                       'Publication Title': 'PublicationTitle'},
              inplace=True)

    # Default language English ******
    df['Language'] = 'English'

    # Normalize ACM DocumentType [conferencePaper, journalArticle]
    df = zotero(df)

    # Filter and save
    filter_and_save_DB(df, 'ACM')


def sort_SD(df):
    """Index(['Key', 'Item Type', 'Publication Year', 'Author', 'Title',
       'Publication Title', 'ISBN', 'ISSN', 'DOI', 'Url', 'Abstract Note',
       'Date', 'Date Added', 'Date Modified', 'Access Date', 'Pages',
       'Num Pages', 'Issue', 'Volume', 'Number Of Volumes',
       'Journal Abbreviation', 'Short Title', 'Series', 'Series Number',
       'Series Text', 'Series Title', 'Publisher', 'Place', 'Language',
       'Rights', 'Type', 'Archive', 'Archive Location', 'Library Catalog',
       'Call Number', 'Extra', 'Notes', 'File Attachments', 'Link Attachments',
       'Manual Tags', 'Automatic Tags', 'Editor', 'Series Editor',
       'Translator', 'Contributor', 'Attorney Agent', 'Book Author',
       'Cast Member', 'Commenter', 'Composer', 'Cosponsor', 'Counsel',
       'Interviewer', 'Producer', 'Recipient', 'Reviewed Author',
       'Scriptwriter', 'Words By', 'Guest', 'Number', 'Edition',
       'Running Time', 'Scale', 'Medium', 'Artwork Size', 'Filing Date',
       'Application Number', 'Assignee', 'Issuing Authority', 'Country',
       'Meeting Name', 'Conference Name', 'Court', 'References', 'Reporter',
       'Legal Status', 'Priority Numbers', 'Programming Language', 'Version',
       'System', 'Code', 'Code Number', 'Section', 'Session', 'Committee',
       'History', 'Legislative Body'],
      dtype='object')
    """

    df.rename(columns={'Author': 'Authors',
                       'Abstract Note': 'Abstract',
                       'Publication Year': 'Year',
                       'Url': 'Link',
                       'Automatic Tags': 'Keywords',
                       'Item Type': 'DocumentType',
                       'Publication Title': 'PublicationTitle'},
              inplace=True)
    # Normalize SD DocumentType as zotero  [conferencePaper, journalArticle]
    df = zotero(df)

    # Filter and save
    filter_and_save_DB(df, 'SD')


def sort_WoS(df):
    """[Index(['Authors', 'Book Authors', 'Book Editors', 'Book Group Authors',
       'Author Full Names', 'Book Author Full Names', 'Group Authors',
       'Article Title', 'Source Title', 'Book Series Title',
       'Book Series Subtitle', 'Language', 'Document Type', 'Conference Title',
       'Conference Date', 'Conference Location', 'Conference Sponsor',
       'Conference Host', 'Author Keywords', 'Keywords Plus', 'Abstract',
       'Addresses', 'Reprint Addresses', 'Email Addresses', 'Researcher Ids',
       'ORCIDs', 'Funding Orgs', 'Funding Text', 'Cited References',
       'Cited Reference Count', 'Times Cited, WoS Core',
       'Times Cited, All Databases', '180 Day Usage Count',
       'Since 2013 Usage Count', 'Publisher', 'Publisher City',
       'Publisher Address', 'ISSN', 'eISSN', 'ISBN', 'Journal Abbreviation',
       'Journal ISO Abbreviation', 'Publication Date', 'Publication Year',
       'Volume', 'Issue', 'Part Number', 'Supplement', 'Special Issue',
       'Meeting Abstract', 'Start Page', 'End Page', 'Article Number', 'DOI',
       'Book DOI', 'Early Access Date', 'Number of Pages', 'WoS Categories',
       'Research Areas', 'IDS Number', 'UT (Unique WOS ID)', 'Pubmed Id',
       'Open Access Designations', 'Highly Cited Status', 'Hot Paper Status',
       'Date of Export', 'Unnamed: 67'],
      dtype='object')
    """
    df.rename(columns={'Article Title': 'Title',
                       'Addresses': 'Affiliations',
                       'Publication Year': 'Year',
                       'Url': 'Link',
                       'Author Keywords': 'Keywords',
                       'Document Type': 'DocumentType',
                       'Source Title': 'PublicationTitle'},
              inplace=True)

    # Normalize WoS DocumentType [conferencePaper, journalArticle]
    df.loc[df.DocumentType == "Article", "DocumentType"] = "Journal"
    df.loc[df.DocumentType == "Article; Early Access", "DocumentType"] = "JournalEA"
    df.loc[df.DocumentType == "Article; Data Paper", "DocumentType"] = "Journal"
    df.loc[df.DocumentType == "Review; Early Access", "DocumentType"] = "ReviewEA"
    df.loc[df.DocumentType == "Article; Proceedings Paper", "DocumentType"] = "Conference"
    df.loc[df.DocumentType == "Software Review; Early Access", "DocumentType"] = "SoftwareReviewEA"
    df.loc[df.DocumentType == "Editorial Material; Early Access", "DocumentType"] = "Editorial"
    df.loc[df.DocumentType == "Editorial Material", "DocumentType"] = "Editorial"
    df.loc[df.DocumentType == "News Item; Early Access", "DocumentType"] = "NewItemEA"

    # Merge conference title with publication title
    df.loc[df.DocumentType == 'Conference', 'PublicationTitle'] = df['Conference Title']

    # Fill current year for Early access articles
    now = datetime.datetime.now()
    df.loc[df.DocumentType == 'JournalEA', 'Year'] = now.year
    df.loc[df.DocumentType == 'ReviewEA', 'Year'] = now.year

    # Filter and save
    filter_and_save_DB(df, 'WoS')


def sort_Compendex(df):
    """Index(['Title', 'Accession number', 'Title of translation', 'Author',
       'Author affiliation', 'Corresponding author', 'Source',
       'Abbreviated source title', 'Sponsor', 'Publisher', 'Volume', 'Issue',
       'Pages', 'Issue date', 'Monograph title', 'Volume title', 'Part number',
       'Publication date', 'Publication year', 'Language', 'ISSN', 'E-ISSN',
       'ISBN', 'ISBN13', 'DOI', 'Article number', 'Conference name',
       'Conference date', 'Conference location', 'Conference code', 'CODEN',
       'Country of publication', 'Document type', 'Abstract',
       'Number of references', 'Main Heading', 'Controlled/Subject terms',
       'Uncontrolled terms', 'Classification code', 'IPC code', 'Treatment',
       'Discipline', 'Funding details', 'Funding text', 'Access type',
       'Database', 'Copyright', 'Data Provider'],
      dtype='object')
    """
    df.rename(columns={'Author': 'Authors',
                       'Author affiliation': 'Affiliations',
                       'Publication year': 'Year',
                       'Url': 'Link',
                       'Uncontrolled terms': 'Keywords',
                       'Document type': 'DocumentType',
                       'Source': 'PublicationTitle'},
              inplace=True)

    # Normalize WoS DocumentType [conferencePaper, journalArticle]
    df.loc[df.DocumentType == "Conference article (CA)", "DocumentType"] = "Conference"
    df.loc[df.DocumentType == "Journal article (JA)", "DocumentType"] = "Journal"
    df.loc[df.DocumentType == "Book chapter (CH)", "DocumentType"] = "Book Chapter"
    df.loc[df.DocumentType == "Book (BK)", "DocumentType"] = "Book"

    # filter and save Compendex DB
    df_compendex = df[df.Database == 'Compendex']
    filter_and_save_DB(df_compendex, 'Compendex')

    # filter and save inspec DB includes arXiv
    df_inspec = df[df.Database == 'Inspec']
    # Inspec database update year

    # iterate over the dataframe row by row
    for index_label, row_series in df_inspec.iterrows():
        # For each row update the 'Year' value at Publication date
        df_inspec.at[index_label, 'Year'] = find_year_inspec_db(row_series['Publication date'])
        # Find arXiv
        df_inspec.at[index_label, 'DocumentType'] = find_arXiv_inspec_db(row_series)
    # Save
    filter_and_save_DB(df_inspec, 'Inspec')


def find_year_inspec_db(publication_date):
    pub_date = str(publication_date).split(' ')
    year = [s for s in pub_date if len(s) == 4 and s.startswith(("19", "20"))]
    return year[0]


def find_arXiv_inspec_db(row):
    return 'arXiv' if str(row['PublicationTitle']) == 'arXiv' else row.DocumentType


def zotero(df_normalize):
    df_normalize.loc[df_normalize.DocumentType == "conferencePaper", "DocumentType"] = "Conference"
    df_normalize.loc[df_normalize.DocumentType == "journalArticle", "DocumentType"] = "Journal"
    df_normalize.loc[df_normalize.DocumentType == "bookSection", "DocumentType"] = "Book Chapter"
    df_normalize.loc[df_normalize.DocumentType == "book", "DocumentType"] = "Book"

    df_normalize['Language'] = df_normalize['Language'].astype(str)
    df_normalize.loc[df_normalize.Language == "en", "Language"] = "English"

    return df_normalize


def filter_and_save_DB(data, db_name):
    # Filter field and save
    data = database_filter(data)
    data.insert(0, 'DataBase', db_name)
    data.to_csv(os.path.join(output_dir, '{}_sorted.csv'.format(db_name)), index=False)


def read_subfolder(curr_DB):
    # read subfolder DB pages
    aux_datasets_list = [f for f in os.listdir(curr_DB) if not f.startswith(('.', '~$', '~'))]
    aux_df = pd.DataFrame()
    pages_number = []

    for aux_dataset in aux_datasets_list:
        pages_number.append(aux_dataset)
        file_format = aux_dataset.split('.')[-1]

        # Supported extensions xls, xlsx and csv
        page = pd.DataFrame()
        if file_format in 'xlsx':
            page = pd.read_excel(os.path.join(curr_DB, aux_dataset))

        elif file_format == 'csv':
            page = pd.read_csv(os.path.join(curr_DB, aux_dataset))
        else:
            print('ERROR unknow file extension. Supported (.xls , .csv)')

        aux_df = aux_df.append(page, ignore_index=True, verify_integrity=True)
    return aux_df, pages_number


def read_datasets(dataset_dir, databases):
    # db_list datasets folder. Ignore hidden files begining with .
    datasets_list = [f for f in os.listdir(dataset_dir) if not f.startswith('.')]

    # iterate over all dataset in datasets' folder
    for dataset in datasets_list:
        # Sort folders
        curr_DB = os.path.join(dataset_dir, dataset)

        pages = []
        database_name = ''
        temp_df_dataset = pd.DataFrame()

        if dataset in databases:
            temp_df_dataset, pages = read_subfolder(curr_DB)
            database_name = str(dataset)
        else:
            print('ERROR unknow Database set. Supported {}'.format(databases))

        # Normalize databases fields
        if database_name == 'Scopus':
            sort_scopus(temp_df_dataset)
        elif database_name == 'IEEE':
            sort_ieee(temp_df_dataset)
        elif database_name == 'ACM':
            sort_ACM(temp_df_dataset)
        elif database_name == 'WoS':
            sort_WoS(temp_df_dataset)
        elif database_name == 'SD':
            sort_SD(temp_df_dataset)
        elif database_name == 'Compendex':
            sort_Compendex(temp_df_dataset)
        else:
            print('ERROR unknow DB')

        # Summary
        print('Read {} DB: ok  Pages:{}  Files:{}'.format(database_name, len(pages), pages))


def merge_datasets():
    # read results sorted csv
    ieee = pd.read_csv(os.path.join(output_dir, 'ieee_sorted.csv'))
    scopus = pd.read_csv(os.path.join(output_dir, 'scopus_sorted.csv'))
    ACM = pd.read_csv(os.path.join(output_dir, 'ACM_sorted.csv'))
    WoS = pd.read_csv(os.path.join(output_dir, 'WoS_sorted.csv'))
    SD = pd.read_csv(os.path.join(output_dir, 'SD_sorted.csv'))
    compendex = pd.read_csv(os.path.join(output_dir, 'Compendex_sorted.csv'))
    inspec = pd.read_csv(os.path.join(output_dir, 'Inspec_sorted.csv'))

    # concat frames
    frames = [ieee, scopus, ACM, WoS, SD, compendex, inspec]
    df = pd.concat(frames)

    # Reindex
    df.insert(0, 'Index', range(0, df.index.size))

    # Get document type [academic, industry]
    df['DocType'] = df.apply(lambda row: DocumentType(row), axis=1)

    # Reorder column for easy read
    df.insert(11, 'Type', df['DocType'])
    df.pop('DocType')

    # Filter milti - Language looking for English
    df['Language'] = df.apply(lambda row: SetLanguage(row), axis=1)
    # Set columns type
    df['Year'] = df['Year'].astype('Int64')
    # New Relevance column to fill with Y = Yes, N = No
    df.insert(3, 'Relevant', 'N')
    # Normalize case
    df['PublicationTitle'] = df['PublicationTitle'].str.upper()
    # Normalize ISSN/ISBN
    df['ISSN'] = df['ISSN'].str.replace('-', '')
    df['ISBN'] = df['ISBN'].str.replace('-', '')

    # Print summary
    # Header
    print('-' * 50, '\nSummary\n', '-' * 50)
    [print(DB, (df[df.DataBase == DB]).shape[0]) for DB in df.DataBase.unique()]
    print('\nTotal', df.DataBase.shape[0])
    return df


def databases_update(JCR_journals_directory, conferences_directory, years_range):
    """
    JCR database
    Index(['TITLE ABRV', 'YEAR', 'ISO_ABREV', 'ISSN', 'LANGUAGE', 'CATEGORY',
       'TITLE', 'QUARTIL', 'CAT SEARCH'],
      dtype='object')

    Conferences database
    'TITLE;TITLE ABRV;Catalog;Rank;N;M;O;P']

    """
    # Prepare JCR database
    JCR_DB = prepare_JCR_DB(JCR_journals_directory)

    # Prepare conferences database
    Conf_DB = prepare_conference_DB(conferences_directory, years_range)

    # save DBs
    Conf_DB.to_csv(os.path.join(os.getcwd(), 'Processed_DB', 'conferences_processed.csv'), index=False)
    JCR_DB.to_csv(os.path.join(os.getcwd(), 'Processed_DB', 'journals_processed.csv'), index=False)
    return JCR_DB, Conf_DB


def prepare_conference_DB(conferences_db, years_range):
    df_conferences = pd.read_excel(conferences_db)
    year_list = np.arange(years_range[0], years_range[1] + 1, 1)
    # Obtain year column from catalog column
    df_conferences['YEAR'] = df_conferences.apply(lambda row: get_conference_year(row, year_list), axis=1)
    # Normalize case
    df_conferences['TITLE'] = df_conferences['TITLE'].str.upper()

    return df_conferences


def prepare_JCR_DB(journals_db):
    df_journals = pd.read_excel(journals_db)
    # Sort journals by QUARTIL for later delete duplicates mantainig higher Q1, Q2, Q3, Q4
    df_journals.sort_values(by='QUARTIL', ascending=True, inplace=True)
    # Dropping duplicate values
    df_journals.drop_duplicates(subset=["TITLE", 'YEAR', 'ISSN'], keep='first', inplace=True)
    # sort by TITTLE ABRV BEFORE save
    df_journals.sort_values(by=['TITLE ABRV', 'YEAR'], ascending=True, inplace=True)
    # Normalize case
    df_journals['TITLE'] = df_journals['TITLE'].str.upper()
    # Normalize ISSN. Delete -
    df_journals['ISSN'] = df_journals['ISSN'].str.replace('-', '')

    return df_journals


def get_conference_year(row, year_list):
    conference_year = [year for year in year_list if str(year) in row.Catalog]
    return conference_year[0] if conference_year else ''


def get_quartile(JCR_DB, Conf_DB, data, DocType):
    with open(os.path.join(os.getcwd(), 'not_found_journals.txt'), 'w') as myfile:
        # Update Quartil and conference ranks
        data['Quartil'] = data.progress_apply(lambda row: fill_quartil(row, JCR_DB, Conf_DB, DocType, myfile), axis=1)
    return data


def fill_quartil(article_row, df_journals, df_conferences, DocType, myfile):

    if article_row.DocumentType in DocType:
        # known document types
        journal_fit = ['Journal', 'JournalEA']
        conference_fit = ['Conference', 'ConferenceEA']

        # JOURNALS
        if article_row.DocumentType in journal_fit:
            # find journal per ISSN  or TITLE
            df_journals = df_journals.loc[(df_journals['ISSN'] == article_row.ISSN) |
                                          (df_journals['TITLE'] == article_row.PublicationTitle)]
            if df_journals.empty:
                # journal ISSN or TITLE not found
                myfile.write('Journal' + ',' + str(article_row.PublicationTitle) + '\n')
                return 'NF'
            else:
                # find journal per YEAR
                journal_search = df_journals.loc[df_journals['YEAR'] == article_row.Year]
                if journal_search.QUARTIL.values.size > 0:
                    return journal_search.QUARTIL.values[0]
                else:
                    # asignate Quartil of the closest year found in database
                    closest_year_found = min(df_journals.YEAR, key=lambda x: abs(x - article_row.Year))
                    df_journals = df_journals.loc[(df_journals['YEAR'] == int(closest_year_found))]
                    return df_journals.QUARTIL.values[0]

        # CONFERENCES
        elif article_row.DocumentType in conference_fit:

            # find conference per  TITLE and year
            rank = [[article_row.PublicationTitle, df_conferences.TITLE[index], df_conferences.Rank[index], df_conferences.YEAR[index]] for index, conference in df_conferences.iterrows()
                    if str(conference['TITLE']) in str(article_row.PublicationTitle)]
            # build a new df with titles found in conferneces database
            df_rank = pd.DataFrame(rank, columns=['ARTICLE_TITLE', 'CONF_TITLE', 'RANK', 'YEAR'])

            if df_rank.empty:
                # conference TITLE not found
                myfile.write('Conference' + ',' + str(article_row.PublicationTitle) + '\n')
                return 'NF'
            else:
                # asignate rank to the closest year of conference in database
                closest_year_found = min(df_rank.YEAR, key=lambda x: abs(x - article_row.Year))
                df_rank = df_rank.loc[(df_rank['YEAR'] == int(closest_year_found))]
                return df_rank.RANK.values[0]

        else:
            print('ERROR: Unknow document type')


def temp(df_conf_row, curr_pub_title):

    return True if df_conf_row['TITLE'] in curr_pub_title else False


def SetLanguage(row):
    curr_row = str(row.Language)
    language_list = curr_row.split(';')

    # In case language English
    matchers_language = ['English']
    matching_language = [language for language in language_list if
                         any(matcher in language for matcher in matchers_language)]
    if matching_language:
        language_found = 'English'
    else:
        language_found = language_list[0]
    return language_found


def DocumentType(row):
    curr_row = str(row.Affiliations)
    affiliations_list = curr_row.split(' ')

    if affiliations_list[0] != 'nan':
        # In case Affiliation is not empty ask for University matchers_academy
        matchers_academy = ['University', 'university', 'Univ', 'Univ.', 'Institute', 'Inst',
                            'Inst.', 'Institut', 'College', 'Politecnico',
                            'Politécnico', 'Academy', 'Faculty', 'School']
        matching_academy = [affiliation for affiliation in affiliations_list if
                            any(matcher in affiliation for matcher in matchers_academy)]
        if matching_academy:
            DocType = 'Academic'
        else:  # TO DO NA fileds
            DocType = 'Industry'
    else:
        DocType = ''  # empty field
    return DocType


def counts_per_year(data):
    df = data.groupby(['Year']).size().reset_index(name='counts')

    # convert column df to int
    df = df.astype('int32')

    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 1, figsize=(8, 5), sharex=False, sharey=False)
    sns.despine(left=False, top=False, right=False)
    # Plot duration
    py_plot = sns.pointplot(x=r'Year', y=r'counts', data=df)

    py_plot.set_xticklabels(
        py_plot.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )

    # Axes names
    plt.setp(axes, xlabel=r'Year', ylabel=r'Total number of literature')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'plots', 'year.pdf'), bbox_inches="tight")


def counts_per_repositories(data):
    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 1, figsize=(7, 5), sharex=False, sharey=False)
    sns.despine(left=False, top=False, right=False)

    # Plot
    current_palette = sns.color_palette()
    ax = sns.countplot(x=r'DataBase', palette=sns.color_palette(current_palette, 1), data=data)
    without_hue(ax, data.DataBase)
    # Axes names
    plt.setp(axes, xlabel=r'Electronic repositories', ylabel=r'Total number of literature')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'plots', 'repositories.pdf'), bbox_inches="tight")


def counts_per_document_type(data, doctype):
    # Filter by conferences and journals
    data = data[data.DocumentType.isin(doctype)]

    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 1, figsize=(5, 5), sharex=False, sharey=False)
    sns.despine(left=False, top=False, right=False)

    # Plot
    current_palette = sns.color_palette()
    ax = sns.countplot(x=r'DocumentType', palette=sns.color_palette(current_palette, 1), data=data)
    without_hue(ax, data.DataBase)
    # Axes names
    plt.setp(axes, xlabel=r'Document type', ylabel=r'Total number of literature')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'plots', 'doctype.pdf'), bbox_inches="tight")



def with_hue(plot, feature, Number_of_categories, hue_categories):
    a = [p.get_height() for p in plot.patches]
    patch = [p for p in plot.patches]
    for i in range(Number_of_categories):
        total = feature.value_counts().values[i]
        for j in range(hue_categories):
            percentage = '{:.1f}%'.format(100 * a[(j * Number_of_categories + i)] / total)
            x = patch[(j * Number_of_categories + i)].get_x() + patch[
                (j * Number_of_categories + i)].get_width() / 2 - 0.15
            y = patch[(j * Number_of_categories + i)].get_y() + patch[(j * Number_of_categories + i)].get_height()
            ax.annotate(percentage, (x, y), size=12)
    plt.show()


def without_hue(ax, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size=14)


def filter_criteria(data, criteria):
    # Dropping duplicate values
    count = len(data)
    data.drop_duplicates(subset="Title", keep=False, inplace=True)
    dup_count = count - len(data)

    # Year
    count = len(data)
    years = np.arange(criteria['Years'][0], criteria['Years'][1] + 1, 1)
    data = data[data.Year.isin(years)]
    y_count = count - len(data)

    # Language
    count = len(data)
    Language = criteria['Language']
    data = data[data.Language.isin(Language)]
    l_count = count - len(data)

    # Document type
    count = len(data)
    DocType = criteria['DocumentType']
    data = data[data.DocumentType.isin(DocType)]
    dt_count = count - len(data)

    print('-' * 50, '\nCriteria\n', '-' * 50)
    print('Duplicated = {}'.format(dup_count))
    print('Years = {}'.format(y_count))
    print('Language = {}'.format(l_count))
    print('DocumentType = {}'.format(dt_count))
    print('Total = {}\n'.format(len(data)))
    return data


if __name__ == '__main__':
    # datasets directory
    dataset_dir = os.path.join(dir_path, 'datasets')
    # Journals database
    journals = os.path.join(dir_path, 'DBs', 'Journals', 'db.xlsx')
    # Conferences database
    conferences = os.path.join(dir_path, 'DBs', 'Conferences', 'CORE_ALL.xlsx')
    # Set of databases - Folder names must be as the databases set array
    Database_Set = ['WoS', 'SD', 'Scopus', 'IEEE', 'ACM', 'Compendex']
    print('\nNumber of DB = {}\n'.format(len(Database_Set)))
    # read datasets .csv files from directory
    read_datasets(dataset_dir, Database_Set)

    # merge all results folder files
    data = merge_datasets()

    # Save bulk summary
    data.to_csv(os.path.join(os.getcwd(), 'bulk_summary.csv'), index=False)

    # criteria by dictionary
    # | data | Year: ini , fin | Language: English| DocumentType: Journal, Conference, EA | Quartil: 1 2 | CORE A A*|
    criteria = {'Years': [1999, 2020],
                'Language': ['English'],
                'DocumentType': ['Journal', 'Conference', 'JournalEA', 'ConferenceEA'],
                'JournalRank': [1, 2],
                'ConferenceRank': ['A', 'A*']
                }
    # FILTER CRITERIA RESULTS
    filtered_data = filter_criteria(data, criteria)

    # Prepare JCR Journal and Conferences databases
    JCR_DB, CONF_DB = databases_update(journals, conferences, criteria['Years'])

    # Get quartile and conference ranks
    df = get_quartile(JCR_DB, CONF_DB, data, criteria['DocumentType'])

    # Save bulk summary
    df.to_csv(os.path.join(os.getcwd(), 'summary.csv'), index=False)
    print(df['Quartil'].value_counts())
    # figures
    # counts_per_year(filtered_data)
    # counts_per_repositories(filtered_data)
    # counts_per_document_type(filtered_data, criteria['DocumentType'])
