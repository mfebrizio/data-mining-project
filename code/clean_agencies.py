# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:16:28 2022

@author: mark
"""

# import dependencies
import pandas as pd
import requests

# set default agency schema via Federal Register API
# source: https://www.federalregister.gov/developers/documentation/api/v1
defaultAgencySchema = ['action',
 'administration-office-executive-office-of-the-president',
 'administrative-conference-of-the-united-states',
 'administrative-office-of-united-states-courts',
 'advisory-council-on-historic-preservation',
 'advocacy-and-outreach-office',
 'agency-for-healthcare-research-and-quality',
 'agency-for-international-development',
 'agency-for-toxic-substances-and-disease-registry',
 'aging-administration',
 'agricultural-marketing-service',
 'agricultural-research-service',
 'agriculture-department',
 'air-force-department',
 'air-quality-national-commission',
 'air-transportation-stabilization-board',
 'alaska-power-administration',
 'alcohol-and-tobacco-tax-and-trade-bureau',
 'alcohol-tobacco-firearms-and-explosives-bureau',
 'american-battle-monuments-commission',
 'amtrak-reform-council',
 'animal-and-plant-health-inspection-service',
 'antitrust-division',
 'antitrust-modernization-commission',
 'appalachian-regional-commission',
 'appalachian-states-low-level-radioactive-waste-commission',
 'architect-of-the-capitol',
 'architectural-and-transportation-barriers-compliance-board',
 'arctic-research-commission',
 'armed-forces-retirement-home',
 'arms-control-and-disarmament-agency',
 'army-department',
 'assassination-records-review-board',
 'barry-m-goldwater-scholarship-and-excellence-in-education-foundation',
 'benefits-review-board',
 'bipartisan-commission-on-entitlement-and-tax-reform',
 'board-of-directors-of-the-hope-for-homeowners-program',
 'bonneville-power-administration',
 'broadcasting-board-of-governors',
 'bureau-of-the-fiscal-service',
 'census-bureau',
 'census-monitoring-board',
 'centers-for-disease-control-and-prevention',
 'centers-for-medicare-medicaid-services',
 'central-intelligence-agency',
 'chemical-safety-and-hazard-investigation-board',
 'child-support-enforcement-office',
 'children-and-families-administration',
 'christopher-columbus-quincentenary-jubilee-commission',
 'civil-rights-commission',
 'coast-guard',
 'commerce-department',
 'commercial-space-transportation-office',
 'commission-of-fine-arts',
 'commission-on-immigration-reform',
 'commission-on-protecting-and-reducing-government-secrecy',
 'commission-on-review-of-overseas-military-facility-structure-of-the-united-states',
 'commission-on-structural-alternatives-for-the-federal-courts-of-appeals',
 'commission-on-the-advancement-of-federal-law-enforcement',
 'commission-on-the-bicentennial-of-the-united-states-constitution',
 'commission-on-the-future-of-the-united-states-aerospace-industry',
 'commission-on-the-social-security-notch-issue',
 'committee-for-purchase-from-people-who-are-blind-or-severely-disabled',
 'committee-for-the-implementation-of-textile-agreements',
 'commodity-credit-corporation',
 'commodity-futures-trading-commission',
 'community-development-financial-institutions-fund',
 'community-living-administration',
 'competitiveness-policy-council',
 'comptroller-of-the-currency',
 'congressional-budget-office',
 'consumer-financial-protection-bureau',
 'consumer-product-safety-commission',
 'cooperative-state-research-education-and-extension-service',
 'coordinating-council-on-juvenile-justice-and-delinquency-prevention',
 'copyright-office-library-of-congress',
 'copyright-royalty-board',
 'copyright-royalty-judges',
 'corporation-for-national-and-community-service',
 'council-of-the-inspectors-general-on-integrity-and-efficiency',
 'council-on-environmental-quality',
 'counsel-to-the-president',
 'court-services-and-offender-supervision-agency-for-the-district-of-columbia',
 'crime-and-security-in-u-s-seaports-interagency-commission',
 'customs-service',
 'defense-acquisition-regulations-system',
 'defense-base-closure-and-realignment-commission',
 'defense-contract-audit-agency',
 'defense-criminal-investigative-service',
 'defense-department',
 'defense-information-systems-agency',
 'defense-intelligence-agency',
 'defense-investigative-service',
 'defense-logistics-agency',
 'defense-mapping-agency',
 'defense-nuclear-facilities-safety-board',
 'defense-special-weapons-agency',
 'delaware-river-basin-commission',
 'denali-commission',
 'disability-employment-policy-office',
 'drug-enforcement-administration',
 'economic-analysis-bureau',
 'economic-analysis-staff',
 'economic-development-administration',
 'economic-research-service',
 'economics-and-statistics-administration',
 'education-department',
 'election-assistance-commission',
 'electronic-commerce-advisory-commission',
 'emergency-oil-and-gas-guaranteed-loan-board',
 'emergency-steel-guarantee-loan-board',
 'employee-benefits-security-administration',
 'employees-compensation-appeals-board',
 'employment-and-training-administration',
 'employment-standards-administration',
 'energy-department',
 'energy-efficiency-and-renewable-energy-office',
 'energy-information-administration',
 'energy-policy-and-new-uses-office',
 'energy-research-office',
 'engineers-corps',
 'engraving-and-printing-bureau',
 'environment-office-energy-department',
 'environmental-protection-agency',
 'equal-employment-opportunity-commission',
 'executive-council-on-integrity-and-efficiency',
 'executive-office-for-immigration-review',
 'executive-office-of-the-president',
 'export-administration-bureau',
 'export-import-bank',
 'family-assistance-office',
 'farm-credit-administration',
 'farm-credit-system-insurance-corporation',
 'farm-service-agency',
 'federal-accounting-standards-advisory-board',
 'federal-acquisition-regulation-system',
 'federal-aviation-administration',
 'federal-bureau-of-investigation',
 'federal-communications-commission',
 'federal-contract-compliance-programs-office',
 'federal-council-on-the-arts-and-the-humanities',
 'federal-crop-insurance-corporation',
 'federal-deposit-insurance-corporation',
 'federal-election-commission',
 'federal-emergency-management-agency',
 'federal-energy-regulatory-commission',
 'federal-financial-institutions-examination-council',
 'federal-highway-administration',
 'federal-housing-enterprise-oversight-office',
 'federal-housing-finance-agency',
 'federal-housing-finance-board',
 'federal-labor-relations-authority',
 'federal-law-enforcement-training-center',
 'federal-maritime-commission',
 'federal-mediation-and-conciliation-service',
 'federal-mine-safety-and-health-review-commission',
 'federal-motor-carrier-safety-administration',
 'federal-pay-advisory-committee',
 'federal-permitting-improvement-steering-council',
 'federal-prison-industries',
 'federal-procurement-policy-office',
 'federal-railroad-administration',
 'federal-register-office',
 'federal-register-administrative-committee',
 'federal-reserve-system',
 'federal-retirement-thrift-investment-board',
 'federal-service-impasses-panel',
 'federal-trade-commission',
 'federal-transit-administration',
 'financial-crimes-enforcement-network',
 'financial-crisis-inquiry-commission',
 'financial-research-office',
 'financial-stability-oversight-council',
 'first-responder-network-authority',
 'fiscal-service',
 'fish-and-wildlife-service',
 'food-and-consumer-service',
 'food-and-drug-administration',
 'food-and-nutrition-service',
 'food-safety-and-inspection-service',
 'foreign-agricultural-service',
 'foreign-assets-control-office',
 'foreign-claims-settlement-commission',
 'foreign-service-grievance-board',
 'foreign-service-impasse-disputes-panel',
 'foreign-service-labor-relations-board',
 'foreign-trade-zones-board',
 'forest-service',
 'general-services-administration',
 'geographic-names-board',
 'geological-survey',
 'government-accountability-office',
 'government-ethics-office',
 'government-national-mortgage-association',
 'government-publishing-office',
 'grain-inspection-packers-and-stockyards-administration',
 'great-lakes-st-lawrence-seaway-development-corporation',
 'gulf-coast-ecosystem-restoration-council',
 'harry-s-truman-scholarship-foundation',
 'health-and-human-services-department',
 'health-care-finance-administration',
 'health-resources-and-services-administration',
 'hearings-and-appeals-office-energy-department',
 'hearings-and-appeals-office-interior-department',
 'homeland-security-department',
 'housing-and-urban-development-department',
 'immigration-and-naturalization-service',
 'indian-affairs-bureau',
 'indian-arts-and-crafts-board',
 'indian-health-service',
 'indian-trust-transition-office',
 'industry-and-security-bureau',
 'information-security-oversight-office',
 'inspector-general-office-agriculture-department',
 'inspector-general-office-health-and-human-services-department',
 'institute-of-american-indian-and-alaska-native-culture-and-arts-development',
 'institute-of-museum-and-library-services',
 'inter-american-foundation',
 'interagency-floodplain-management-review-committee',
 'intergovernmental-relations-advisory-commission',
 'interior-department',
 'internal-revenue-service',
 'international-boundary-and-water-commission-united-states-and-mexico',
 'international-broadcasting-board',
 'international-development-cooperation-agency',
 'u-s-international-development-finance-corporation',
 'international-investment-office',
 'international-joint-commission-united-states-and-canada',
 'international-organizations-employees-loyalty-board',
 'international-trade-administration',
 'international-trade-commission',
 'interstate-commerce-commission',
 'investment-security-office',
 'james-madison-memorial-fellowship-foundation',
 'japan-united-states-friendship-commission',
 'joint-board-for-enrollment-of-actuaries',
 'judicial-conference-of-the-united-states',
 'judicial-review-commission-on-foreign-asset-control',
 'justice-department',
 'justice-programs-office',
 'juvenile-justice-and-delinquency-prevention-office',
 'labor-department',
 'labor-statistics-bureau',
 'labor-management-standards-office',
 'land-management-bureau',
 'legal-services-corporation',
 'library-of-congress',
 'local-television-loan-guarantee-board',
 'management-and-budget-office',
 'marine-mammal-commission',
 'maritime-administration',
 'medicare-payment-advisory-commission',
 'merit-systems-protection-board',
 'military-compensation-and-retirement-modernization-commission',
 'millennium-challenge-corporation',
 'mine-safety-and-health-administration',
 'minerals-management-service',
 'mines-bureau',
 'minority-business-development-agency',
 'minority-economic-impact-office',
 'mississippi-river-commission',
 'monetary-offices',
 'morris-k-udall-and-stewart-l-udall-foundation',
 'national-aeronautics-and-space-administration',
 'national-agricultural-library',
 'national-agricultural-statistics-service',
 'national-archives-and-records-administration',
 'national-assessment-governing-board',
 'national-bankruptcy-review-commission',
 'national-biological-service',
 'national-bipartisan-commission-on-future-of-medicare',
 'national-capital-planning-commission',
 'national-civilian-community-corps',
 'national-commission-on-fiscal-responsibility-and-reform',
 'national-commission-on-intermodal-transportation',
 'national-commission-on-libraries-and-information-science',
 'national-commission-on-manufactured-housing',
 'national-commission-on-military-national-and-public-service',
 'national-commission-on-terrorist-attacks-upon-the-united-states',
 'national-commission-on-the-cost-of-higher-education',
 'national-communications-system',
 'national-consumer-cooperative-bank',
 'national-council-on-disability',
 'national-counterintelligence-center',
 'national-credit-union-administration',
 'national-crime-prevention-and-privacy-compact-council',
 'national-economic-council',
 'national-education-goals-panel',
 'national-endowment-for-the-arts',
 'national-endowment-for-the-humanities',
 'national-foundation-on-the-arts-and-the-humanities',
 'national-gambling-impact-study-commission',
 'national-geospatial-intelligence-agency',
 'national-highway-traffic-safety-administration',
 'national-historical-publications-and-records-commission',
 'national-indian-gaming-commission',
 'national-institute-for-literacy',
 'national-institute-of-corrections',
 'national-institute-of-food-and-agriculture',
 'national-institute-of-justice',
 'national-institute-of-standards-and-technology',
 'national-institutes-of-health',
 'national-intelligence-office-of-the-national-director',
 'national-labor-relations-board',
 'national-library-of-medicine',
 'national-mediation-board',
 'national-nanotechnology-coordination-office',
 'national-nuclear-security-administration',
 'national-oceanic-and-atmospheric-administration',
 'national-park-service',
 'national-partnership-for-reinventing-government',
 'national-prison-rape-elimination-commission',
 'national-railroad-passenger-corporation',
 'national-science-foundation',
 'national-security-agency-central-security-service',
 'national-security-commission-on-artificial-intelligence',
 'national-security-council',
 'national-shipping-authority',
 'national-skill-standards-board',
 'national-technical-information-service',
 'national-telecommunications-and-information-administration',
 'national-transportation-safety-board',
 'national-women-s-business-council',
 'natural-resources-conservation-service',
 'natural-resources-revenue-office',
 'navajo-and-hopi-indian-relocation-office',
 'navy-department',
 'neighborhood-reinvestment-corporation',
 'northeast-dairy-compact-commission',
 'northeast-interstate-low-level-radioactive-waste-commission',
 'nuclear-energy-office',
 'nuclear-regulatory-commission',
 'nuclear-waste-technical-review-board',
 'occupational-safety-and-health-administration',
 'occupational-safety-and-health-review-commission',
 'ocean-energy-management-bureau',
 'ocean-energy-management-regulation-and-enforcement-bureau',
 'ocean-policy-commission',
 'office-of-government-information-services',
 'office-of-motor-carrier-safety',
 'office-of-national-drug-control-policy',
 'office-of-policy-development',
 'office-of-the-chief-financial-officer-agriculture-department',
 'oklahoma-city-national-memorial-trust',
 'operations-office',
 'ounce-of-prevention-council',
 'overseas-private-investment-corporation',
 'pacific-northwest-electric-power-and-conservation-planning-council',
 'panama-canal-commission',
 'parole-commission',
 'partnerships-and-public-engagement-office',
 'patent-and-trademark-office',
 'peace-corps',
 'pension-and-welfare-benefits-administration',
 'pension-benefit-guaranty-corporation',
 'personnel-management-office',
 'physician-payment-review-commission',
 'pipeline-and-hazardous-materials-safety-administration',
 'postal-rate-commission',
 'postal-regulatory-commission',
 'postal-service',
 'president-s-council-on-integrity-and-efficiency',
 'president-s-council-on-sustainable-development',
 'president-s-critical-infrastructure-protection-board',
 'president-s-economic-policy-advisory-board',
 'presidential-advisory-committee-on-gulf-war-veterans-illnesses',
 'presidential-commission-on-assignment-of-women-in-the-armed-forces',
 'presidential-documents',
 'presidio-trust',
 'prisons-bureau',
 'privacy-and-civil-liberties-oversight-board',
 'procurement-and-property-management-office-of',
 'program-support-center',
 'prospective-payment-assessment-commission',
 'public-buildings-reform-board',
 'public-debt-bureau',
 'public-health-service',
 'railroad-retirement-board',
 'reagan-udall-foundation-for-the-food-and-drug-administration',
 'reclamation-bureau',
 'recovery-accountability-and-transparency-board',
 'refugee-resettlement-office',
 'regulatory-information-service-center',
 'research-and-innovative-technology-administration',
 'research-and-special-programs-administration',
 'resolution-trust-corporation',
 'risk-management-agency',
 'rural-business-cooperative-service',
 'rural-housing-and-community-development-service',
 'rural-housing-service',
 'rural-telephone-bank',
 'rural-utilities-service',
 'safety-and-environmental-enforcement-bureau',
 'saint-lawrence-seaway-development-corporation',
 'science-and-technology-policy-office',
 'secret-service',
 'securities-and-exchange-commission',
 'selective-service-system',
 'small-business-administration',
 'smithsonian-institution',
 'social-security-administration',
 'southeastern-power-administration',
 'southwestern-power-administration',
 'special-counsel-office',
 'special-inspector-general-for-afghanistan-reconstruction',
 'special-inspector-general-for-iraq-reconstruction',
 'special-trustee-for-american-indians-office',
 'state-department',
 'state-justice-institute',
 'substance-abuse-and-mental-health-services-administration',
 'surface-mining-reclamation-and-enforcement-office',
 'surface-transportation-board',
 'susquehanna-river-basin-commission',
 'technology-administration',
 'tennessee-valley-authority',
 'the-white-house-office',
 'thrift-depositor-protection-oversight-board',
 'thrift-supervision-office',
 'trade-and-development-agency',
 'trade-representative-office-of-united-states',
 'transportation-department',
 'transportation-office',
 'transportation-security-administration',
 'transportation-statistics-bureau',
 'travel-and-tourism-administration',
 'treasury-department',
 'twenty-first-century-workforce-commission',
 'u-s-citizenship-and-immigration-services',
 'us-codex-office',
 'u-s-customs-and-border-protection',
 'u-s-house-of-representatives',
 'u-s-immigration-and-customs-enforcement',
 'u-s-trade-deficit-review-commission',
 'u-s-china-economic-and-security-review-commission',
 'under-secretary-for-economic-affairs',
 'unified-carrier-registration-plan',
 'uniformed-services-university-of-the-health-sciences',
 'african-development-foundation',
 'united-states-agency-for-global-media',
 'united-states-enrichment-corporation',
 'united-states-information-agency',
 'united-states-institute-of-peace',
 'united-states-marshals-service',
 'united-states-mint',
 'united-states-olympic-and-paralympic-committee',
 'united-states-sentencing-commission',
 'utah-reclamation-mitigation-and-conservation-commission',
 'valles-caldera-trust',
 'veterans-affairs-department',
 'veterans-employment-and-training-service',
 'victims-of-crime-office',
 'wage-and-hour-division',
 'western-area-power-administration',
 'women-s-business-enterprise-interagency-committee',
 'women-s-progress-commemoration-commission',
 'women-s-suffrage-centennial-commission',
 'workers-compensation-programs-office']


# function that converts single column from a dataframe into date format
def FR_clean_agencies(df_input, column = 'agencies', schema = defaultAgencySchema):
    ''' Convert dataframe columns in str format to datetime.date format.
    
    Dependencies
    ------------
    pandas
    requests
    
    Parameters
    ----------
    df : Dataframe
    
    column : str, optional
        String of column name to clean. The default is 'agencies'.
    
    schema : list, optional
        List of agency slugs representing the array of agencies that can appear in the data. 
        The default is the Agency schema from the Federal Register API documentation.
    
    Returns
    -------
    Dataframe
        Copy of input dataframe with clean 'agencies' data added as new columns.
        New columns are located right after the original 'agencies' column.
    '''
    
    # 1) retrieve agencies metadata from FR API
    # endpoint: /agencies
    # define endpoint url; no parameters needed
    agencies_URL = 'https://www.federalregister.gov/api/v1/agencies.json'
    
    # request documents; raise error if it fails
    agencies_response = requests.get(agencies_URL)
    if agencies_response.status_code!=200:
        raise Exception("API request failed. Status code: "+str(agencies_response.status_code))
    
    # set variables
    agencies_metadata = agencies_response.json()
    
    # 2) extract data from 'agencies' column
    
    # create deep copy of input datafram
    df = df_input.copy(deep=True)
    
    # create list of agencies data
    agencies_list = df[column].tolist()
    
    # empty lists for results
    slug_list = []
    id_list = []
    parent_list = []
    
    # loop over documents and extract agencies data
    for rule in agencies_list:
        
        # len==7 is the normal length of an agencies dict with complete information
        slugs = [x['slug'] if len(x)==7 else x['raw_name'].lower().replace(" ","-") for x in rule]
        ids = [x['id'] for x in rule if len(x)==7]
        parents = [x['parent_id'] for x in rule if len(x)==7]
        
        slug_list.append(slugs)
        id_list.append(ids)
        parent_list.append(parents)
    
    # clean slug list to only include agencies in the schema
    # there are some bad metadata -- e.g., 'interim-rule', 'formal-comments-that-were-received-in-response-to-the-nprm-regarding'
    # also ensure no duplicate agencies in each document's list by using set()
    slug_list_clean = [list(set([i for i in slug if i in schema])) for slug in slug_list]
    
    # check if data was extracted correctly; raise error if not
    if not len(agencies_list)==len(slug_list_clean)==len(id_list)==len(parent_list):
        raise Exception("Error extracting data from 'agencies' column.")
    
    # create new columns with restructured data
    df.loc[:,'agency_slugs'] = slug_list_clean
    df.loc[:,'agency_ids'] = id_list
    df.loc[:,'agency_parents'] = parent_list
    
    # 3) generate two columns with unique top-level agency metadata:
    # a. list of unique top-level ids (i.e., parent_id for sub-agencies and agency_id for agencies without a parent)
    # b. list of slugs that correspond to the top-level ids
    
    # create empty lists for results
    top_level_ids = []
    agencies_clean = []
    
    # iterate over rows in dataframe
    for index, row in df.iterrows():
        
        # counter to grab agency_id when parent_id==None
        counter = 0
        
        # 1) create new column: list of unique top-level ids
        # iterate over parent_ids for each document
        # return parent_id for sub-agencies and agency_id for agencies with no parent
        r_list = []
        for r in row['agency_parents']:
            
            if r!=None:
                r_list.append(r)
                
            else:
                r_list.append(row['agency_ids'][counter])
            
            counter = counter + 1
        
        ''' this doesn't work yet; 
        trying to fix instances of agencies with parent agencies who also have parent agencies
        currently just returns the intermediate parent
        #r_list_clean = [] 
        #for r in r_list:
            
        #    r_list_clean.extend([a['id'] if (a['id']==r & a['parent_id']==None) else a['parent_id'] for a in agencies_metadata])
        '''
        
        # use set() to keep only unique ids
        r_ids = list(set(r_list))
        
        # append to results list #1
        top_level_ids.append(r_ids)
        
        # 2) create new column: list of unique top-level slugs
        # iterate over each document's top_level_ids
        # return slug for corresponding id from FR API's agencies endpoint
        r_slugs = []
        for r in r_ids:
            
            # locate slug for each input id from agencies endpoint metadata
            # add id by extending list for each document
            r_slugs.extend([a['slug'] for a in agencies_metadata if a['id']==r])
        
        # append to results list #2
        agencies_clean.append(r_slugs)
    
    # check if results make sense; raise error if not
    if not len(agencies_clean)==len(top_level_ids):
        raise Exception("Error extracting unique data from 'agencies' column.")
        
    # create new columns with extracted data
    df.loc[:,'agencies_id_uq'] = top_level_ids
    df.loc[:,'agencies_slug_uq'] = agencies_clean
    
    # 4) reorder columns
    
    # list of new columns added to dataframe
    new_cols = ['agency_slugs', 'agency_ids', 'agency_parents', 
                'agencies_id_uq', 'agencies_slug_uq']
    
    # list of original columns from input dataframe
    col_list = df_input.columns.tolist()
    
    # location of "agencies" column plus one
    index_loc = col_list.index('agencies') + 1
    
    # create new column list with new columns inserted after 'agencies'
    new_col_list = col_list[0:index_loc] + new_cols + col_list[index_loc:]
    
    df = df.reindex(columns = new_col_list)
    
    return df

