import os
import argparse
import jsonpickle
import json
from function_parser.language_data import LANGUAGE_METADATA
from function_parser.process import DataProcessor
from tree_sitter import Language
import langdetect
import re
from io import StringIO
import tokenize
import tqdm


# function taken from https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/parser/utils.py#L4-L61
def remove_comments_and_docstrings(source, lang):
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
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def main():
    parser = argparse.ArgumentParser(description="Create docstring/function pairs from json file")

    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--path_to_repos", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--spoken_language", type=str, required=True)
    args = parser.parse_args()

    language = args.lang
    DataProcessor.PARSER.set_language(Language(os.path.join(os.getcwd(), "tree-sitter-languages.so"), language))
    processor = DataProcessor(
        language=language, language_parser=LANGUAGE_METADATA[language]["language_parser"]
    )
    name_set = jsonpickle.decode(json.load(open(args.path_to_repos)))
    doc_func_list = []
    for dependee in tqdm.tqdm(name_set, desc="Repos processing"):
        try:
            dep = processor.process_dee(dependee, ext=LANGUAGE_METADATA[language]["ext"])
        except:
            continue
        for element in tqdm.tqdm(dep):
            if len(element["docstring_tokens"]) > 0:
                try:
                    if langdetect.detect(element["docstring_summary"]) == args.spoken_language:
                        element["function"] = remove_comments_and_docstrings(element["function"], lang="java")
                        doc_func_list.append(element)
                except:
                    pass
        print(f"Total number of samples: {len(doc_func_list)}")

    output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir)
    with open(output_path, 'w') as outfile:
        for entry in doc_func_list:
            json.dump(entry, outfile)
            outfile.write('\n')


if __name__ == "__main__":
    main()
