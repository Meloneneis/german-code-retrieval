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
4. Another filtering step involves using the langdetect library to filter repositories based on their spoken language. This process utilizes a counter that starts at 0 and is incremented by 1 each time the library detects the documentation language as the specified language. Conversely, it is decremented by 1 if the detected language does not match. Repositories are discarded or added based on a predetermined threshold, such as reaching a count of 5 or -5.
5. The repositories are saved in the output_dir-
		    

### Note:
- Even with the specified spoken language and queries, the repository could be from another language domain than specified, albeit unlikely.

