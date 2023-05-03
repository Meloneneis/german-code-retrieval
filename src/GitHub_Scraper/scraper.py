import argparse
import itertools
import requests
import time
import json
import jsonpickle
from tqdm import tqdm
import base64
import urlfetch
from tree_sitter import Language, Parser
import langdetect
import re
from io import StringIO
import tokenize

"""
This script retrieves Github Repositories from specified Keywords (e.g. for German: "Klasse", "Parameter", "zurück" etc)
"""

Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',

    # Include one or more languages
    [
        'tree-sitter-java'
    ]
)
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
langparser = Parser()
langparser.set_language(JAVA_LANGUAGE)

method_query = JAVA_LANGUAGE.query("""
(
  (method_declaration) @method
)
""")


# function taken from https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/parser/utils.py#L4-L61
def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)


# gets all possible combinations of queries in the given query size
# this makes sure to retrieve as much unique repositories as possible from a GitHub API call
def query_combinator(args):
    queries = args.queries.split(',')
    return [' '.join(subset) for subset in itertools.combinations(queries, args.query_size)]


def main():
    argparser = argparse.ArgumentParser(description="Combine two datasets to produce a merge file")

    argparser.add_argument("--lang", default="java", type=str)
    argparser.add_argument("--queries", default="Klasse,zurück,Variable,Parameter", type=str)
    argparser.add_argument("--query_size", type=int, default=3)
    argparser.add_argument("--starcount", type=int, default=2)
    argparser.add_argument("--output_dir", default="../../data/github/german_repository_names.json")
    argparser.add_argument("--spoken_language", default="de")
    argparser.add_argument("--auth_token", default="<github token here>", type=str)
    args = argparser.parse_args()

    queries = query_combinator(args)
    headers = {
        'User-Agent': 'Github Scraper',
        'Authorization': f'Token {args.auth_token}'
    }
    repo_names = set()
    page_index = 0

    # get all possible repos from queries
    for query in tqdm(queries, desc="Queries processed"):
        while True:
            url = f"https://api.github.com/search/code?q={query}+in:file +language:{args.lang}&per_page=100&page=" \
                  f"{page_index}"
            response = requests.get(url, headers=headers)
            # wait 1 minute to prevent calling github api too much (rate limit)
            while response.status_code == 403:
                print("\nWait 60 secs...")
                time.sleep(60)
                response = requests.get(url, headers=headers)
            if response.status_code != 200:
                break
            for repository in response.json()["items"]:
                repo_names.add(repository["repository"]["full_name"])
            time.sleep(5)
            page_index += 1

    # filter repos by starcount and by spoken language
    remove_set = set()
    for repo in tqdm(repo_names, desc="Filterprogress of repos"):
        german_test = 0
        noDocCounter = 0
        url = f"https://api.github.com/repos/{repo}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            # broken repos.. can't be accessed anymore
            print(f"\n{response.status_code}:{repo} broken.. skip this")
            continue
        if response.json()["stargazers_count"] < args.starcount:
            # remove repos with starcount lower than the specified starcount
            remove_set.add(repo)
        else:
            print(f"\n{repo} has potential")
            # check if repo is spoken language
            repo_request = requests.get(f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1",
                                        headers=headers)
            while repo_request.status_code == 403:
                print("\n403 error.. wait a little bit..")
                time.sleep(90)
                repo_request = requests.get(f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1",
                                            headers=headers)
            if repo_request.status_code == 404:
                repo_request = requests.get(f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1",
                                            headers=headers)
            repo_json = repo_request.json() if repo_request.status_code == 200 else print(repo_request.status_code)
            if repo_request.status_code != 200:
                continue
            java_files = [file for file in repo_json["tree"] if file["path"].endswith(".java")]
            for file in (file for file in java_files if noDocCounter <= 500):
                try:
                    result = urlfetch.fetch(file["url"], headers=headers)
                    data = json.loads(result.content) if result.status_code == 200 else print(
                        result.status_code)
                    decoded_content = base64.b64decode(data["content"])
                    tree = langparser.parse(bytes(decoded_content))
                    method_captures = method_query.captures(tree.root_node)
                    for capture in (capture for capture in method_captures if (4 >= german_test >= -4)):
                        if capture[0].prev_sibling.type != "doc_comment" or capture[0].next_sibling.type != \
                                "doc_comment":
                            noDocCounter += 1
                            continue
                        docstring = capture[0].prev_sibling.text.decode("ISO-8859-1") if \
                            capture[0].prev_sibling.type == "doc_comment" else \
                            capture[0].next_sibling.text.decode("ISO-8859-1")
                        if langdetect.detect(docstring) != args.spoken_language:
                            german_test -= 1
                            print(german_test, end=" ")
                            if german_test == -5:
                                print(f"\n{repo} removed, because not {args.spoken_language}")
                                remove_set.add(repo)
                                print("check next repo")
                                break
                        else:
                            german_test += 1
                            print(german_test, end=" ")
                            noDocCounter = 0
                        if german_test == 5:
                            print(f"\n{repo} added")
                            print("check next repo")
                            break

                except:
                    pass
                if german_test >= 5 or german_test <= -5 or noDocCounter >= 500:
                    break
    for repo in remove_set:
        repo_names.remove(repo)
    json_set = jsonpickle.encode(repo_names)
    with open(f'{args.output_dir}', 'w') as f:
        json.dump(json_set, f)


if __name__ == "__main__":
    main()
