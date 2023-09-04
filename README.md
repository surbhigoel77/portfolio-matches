### Scraping the Portfolio Page

1. Obtaining the Portfolio Page HTML
The first step is to make a request to the portfolio page URL and parse the HTML content using BeautifulSoup. This gives us a BeautifulSoup object that allows us to analyze the page structure and extract data.

2. Identifying Portfolio Company Cards
The portfolio page contains "portfolio-card" divs, one for each portfolio company. We use BeautifulSoup to find all divs with this class name. This gives us a list of cards to extract data from.

3. Extracting Data from Cards
For each card, we first look for a hover detail div, containing additional data like sector, headquarters, etc. We loop through the items and extract the keys and values. We also extract the company name from the logo alt text, and sector from another element. The profile link url is used to scrape additional details from that page. The company profile pages contain further details in sections like "Introduction", "Description", etc. We locate these elements and extract the text contents to add to the company data dictionary.


### Scraping the HN Job Thread

We start with the HN item url containing the job thread. The page HTML is fetched and parsed.

1. Extracting Comments
The page contains comment rows marked by CSS classes. We loop through these rows and extract key data like user, comment url, and text content into a Job tuple. We also extract the "morelink" element containing the URL for the next page if present.

This gives us structured job data from the entire thread extracted from HN comments.

### Identifying the relevant matches

We first do a preprocessing for the HN data. We use SpaCy named entity recognition to extract company headquarters locations from the job post text.
We check these locations against a list of European country names to tag each job as "European" or not. So this ensures that only European companies are considered from hn dataset when finding matches.

The location extraction using SpaCy NER allows the code to automatically detect European companies instead of needing a manual flag. This preprocessing step filters out non-European companies, letting the similarity matching focus only on relevant companies.

To calculate the text embeddings for similarity matching, we use the `text-embedding-ada-002` model from OpenAI. It generates 768-dimensional vector representations capturing semantic meaning and relationships between texts.

We use two strategies to find the most relevant companies from the Hacker News dataset (hn_df) based on their similarity to companies in the existing portfolio (portfolio_df):

#### Top Match

- It calculates a cosine similarity matrix between all the embeddings in hn_df and portfolio_df
- For each row in hn_df, it finds the row in portfolio_df that has the maximum similarity score
- It sorts the hn_df rows by their maximum similarity score and prints the top 10 matches

So for each potential company, it finds the single most similar company in the portfolio and uses that as the relevance match.

#### Top Match Aggregate 

- It calculates a cosine similarity matrix between all embeddings like before 

- But here instead of taking the max, it sums up the similarity scores for each hn_df row to all the portfolio_df rows

- This gives a total aggregate similarity score indicating if the hn_df company is similar to the portfolio overall

- It sorts hn_df rows by this aggregate score to find most relevant matches

- For each of the top matches, it also prints the top 3 most similar portfolio companies contributing to its score

So this strategy looks at aggregate similarity to the whole portfolio instead of just the single most similar company. It gives a more comprehensive view of relevance while still showing the mostinfluential matches.

In summary, the top match strategy gives a precise 1:1 mapping while the aggregate approach looks at overall similarity to the portfolio based on multiple companies. Both provide useful perspectives to find the most promising companies from the HN data.


### Future work

- The current project focuses on Hacker News and the portfolio site. Additional data sources like AngelList, Crunchbase, LinkedIn, etc could be incorporated to find a wider set of features regarding geography, funding stage and job openings.

- Beyond text, additional signals like funding data, news trends, job postings, etc could improve matching. For example, a company with recent funding activity and job openings could be more relevant than one that hasn't raised money in years.

- Sector wise clustering could be performed to understand the growin sectors. This could help identify promising companies in those sectors. For example, if the portfolio has a lot of companies in the AI space, it could be useful to find AI companies in the HN data.