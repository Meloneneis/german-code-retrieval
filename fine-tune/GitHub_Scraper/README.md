## GitHub Scraper

This script scrapes Github repositories from the specified language domain given some keywords.

### Dependencies

    pip install -r requirements.txt
    
### Example Usage

    python scraper.py \
       --lang "java" \
       --queries "Klasse,zurück,Variable,Parameter" \
       --auth_token "AUTH_TOKEN" \
       --output_dir "../../data/github/german_repository_names.json \
       --spoken_language "de" \
       --output_dir "output_dir"

### Scraping Process
1. The queries are combined into pairs of query size. This ensures that we retrieve more unique repositories due to the limitations of GitHub CodeSearch API.
2. First all repository from a query are retrieved.
3. A filtering is applied to filter the repositories by their specified starcount (defaults to 2)
4. Another filtering is applied to filter the repositories by their spoken language. This is done via the library langdetect and a counter to count the amount of times the library thinks that a documentation is in the specified language. If it is we assign +1 to the counter if it isn't we assign -1. If it hits a certain treshold like 5 or -5 we discard or add the repository.
5. The repositories are saved in the output_dir-
		    

### Note:
- Even with the specified spoken language and queries, the repository could be from another language domain than specified, albeit unlikely.

