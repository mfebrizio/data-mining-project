# Proposal

Building a Classifier to Identify the Type of Federal Register Documents

## Background to the Problem

The [Federal Register](https://www.federalregister.gov/) is the daily journal of the U.S. government, with a new issue published each business day. Each issue is split into 4 sections containing 4 corresponding document types: Notices, Proposed Rules, Rules, and Presidential Documents.

Generally, documents are automatically categorized into the correct type. This is helpful for [searching](https://www.federalregister.gov/documents/search#advanced) the Federal Register for documents of a particular nature or relevance. Further, advanced users can call the [Federal Register REST API](https://www.federalregister.gov/reader-aids/developer-resources/rest-api) to retrieve large quantities of metadata on documents for additional analysis. Given that federal agencies use the Federal Register to officially announce the development and finalization of regulations (also called rules or rulemakings), as well as solicit public input on those regulations, these data are valuable when conducting research on the regulatory process, agency performance, and specific policy areas (e.g., environment, immigration, energy, housing).

One major challenge to research on agency rulemaking is that documents, particularly from the 1990s, remain uncategorized. In the available data from the 1990s (1994-1999), out of XX published documents, *XX* are categorized and *XX* are uncategorized. For some years, such as 1994, this leads to a severe under-counting of the number of rules published.



