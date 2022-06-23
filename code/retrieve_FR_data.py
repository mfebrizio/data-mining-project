# --------------------------------------------------
from datetime import date
import json
from pathlib import Path

import requests

# --------------------------------------------------
p = Path.cwd()
data_dir = p.parent.joinpath('data', 'raw')
if data_dir.exists():
    pass
else:
    try:
        data_dir.mkdir(parents=True)
    except:
        print("Cannot create data directory.")

# --------------------------------------------------
# configure requests: Federal Register API
# documentation: https://www.federalregister.gov/developers/documentation/api/v1/
# endpoint: /documents.{format} -- results since 1994

# define endpoint url
dcts_url = 'https://www.federalregister.gov/api/v1/documents.json?'

# define parameters
res_per_page = 1000
page_offset = 0  # both 0 and 1 return first page
sort_order = 'oldest'
fieldsList = ['citation', 'document_number',  # unique identifiers
              'type', 'subtype',  # labels of interest
              'agencies', 'agency_names', 'publication_date', 'president', 'corrections', 'correction_of',
              # other id info
              'action', 'title', 'page_length', 'effective_on', 'abstract', 'dates', 'toc_doc', 'topics', 'significant',
              'page_views', 'comments_close_on',  # potential features
              'regulation_id_numbers', 'regulation_id_number_info', 'regulations_dot_gov_info', 'cfr_references',
              # non-FR info
              'json_url', 'raw_text_url', 'full_text_xml_url'  # beyond the metadata
              ]

year_range = list(map(str, range(1994, 2000)))  # range end is exclusive
quarters = [("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")]

# dictionary of parameters
dcts_params = {'per_page': res_per_page,
               'page': page_offset,
               'order': sort_order,
               'fields[]': fieldsList,
               'conditions[publication_date][gte]': '',
               'conditions[publication_date][lte]': ''
               }

# check API configuration
print(year_range, '\n')
test_response = requests.get(dcts_url, params=dcts_params)
request_url = test_response.url
print(request_url)

# --------------------------------------------------
# retrieve data from Federal Register API
# create objects
dctsResults_all = []
dctsCount_all = 0

# for loop to pull data for each publication year
for year in year_range:
    print('\n', f'***** Retrieving results for year = {year} *****')

    dctsResults = []
    dctsCount = 0
    # for loop to pull data for each quarter
    for q in quarters:
        q_start = year + "-" + q[0]
        q_end = year + "-" + q[1]

        # update parameters for year
        dcts_params.update({'conditions[publication_date][gte]': q_start,
                            'conditions[publication_date][lte]': q_end})

        # get documents
        dcts_response = requests.get(dcts_url, params=dcts_params)
        print(dcts_response.status_code,
              dcts_response.headers['Date'],
              dcts_response.url, sep='\n')  # print request URL

        # set variables
        dctsCount += dcts_response.json()['count']
        dctsPages = dcts_response.json()['total_pages']  # number of pages to retrieve all results

        # for loop for grabbing results from each page
        for page in range(1, dctsPages + 1):
            dcts_params.update({'page': page})
            dcts_response = requests.get(dcts_url, params=dcts_params)
            results_this_page = dcts_response.json()['results']
            dctsResults.extend(results_this_page)
            print('Results retrieved = ' + str(len(dctsResults)))

    # create dictionary for year to export as json
    if len(dctsResults) == dctsCount:
        dctsRules_one_year = {'source': 'Federal Register API',
                              'endpoint': 'https://www.federalregister.gov/api/v1/documents.{format}',
                              'requestURL': request_url,
                              'dateRetrieved': str(date.today()),
                              'count': dctsCount,
                              'results': dctsResults}

        filePath = data_dir / rf'documents_endpoint_rules_{year}.json'
        with open(filePath, 'w', encoding='utf-8') as outfile:
            json.dump(dctsRules_one_year, outfile, indent=4)

        print('Retrieved all results for ' + str(year) + '!')

    else:
        print('Counts do not align for ' + str(year) + ': ' + str(len(dctsResults)) + ' =/= ' + str(dctsCount))

    # extend list of cumulative results and counts
    dctsResults_all.extend(dctsResults)
    dctsCount_all = dctsCount_all + dctsCount

# save params for export with metadata
save_params = dcts_params.copy()
save_params.pop('page')
save_params.pop('per_page')
save_params.pop('conditions[publication_date][gte]')
save_params.pop('conditions[publication_date][lte]')

# create dictionary of data with retrieval date
dctsRules = {'source': 'Federal Register API, https://www.federalregister.gov/reader-aids/developer-resources/rest-api',
             'endpoint': 'https://www.federalregister.gov/api/v1/documents.{format}',
             'requestURL': request_url,
             'dateUpdated': str(date.today()),
             'params': save_params,
             'dateRange': year_range[0] + ' - ' + year_range[-1],
             'count': dctsCount_all,
             'results': dctsResults_all}
if dctsRules['count'] == len(dctsRules['results']):
    print('\nDictionary with retrieval date created!')
else:
    print('\nError creating dictionary...')

# export json file
filePath = data_dir / rf'documents_endpoint_rules_combo_{year_range[0]}_{year_range[-1]}.json'
with open(filePath, 'w', encoding='utf-8') as outfile:
    json.dump(dctsRules, outfile, indent=4)

print('Exported as JSON!')
