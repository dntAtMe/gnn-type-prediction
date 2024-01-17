import ast
import fnmatch
import os

from utils.graph_extractor import GraphExtractor


def _parse_gitignore(directory_path):
    gitignore_file = os.path.join(directory_path, '.gitignore')
    ignore_patterns = []

    if os.path.isfile(gitignore_file):
        with open(gitignore_file, 'r') as file:
            ignore_patterns = file.read().splitlines()

    return ignore_patterns


def _should_ignore_path(path, ignore_patterns):
    return any(fnmatch.fnmatch(path, pattern) for pattern in ignore_patterns)


def parse_python_files_in_directory(directory_path):
    results = {}

    # Parse .gitignore file
    ignore_patterns = _parse_gitignore(directory_path)

    for dirpath, dirnames, filenames in os.walk(directory_path, topdown=True):
        # Filter out directories to ignore
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and not _should_ignore_path(d, ignore_patterns)]

        for filename in filenames:
            if filename.endswith(".py") and not _should_ignore_path(filename, ignore_patterns):
                file_path = os.path.join(dirpath, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()

                    parsed_ast = ast.parse(file_content)

                    visitor = GraphExtractor()
                    visitor.visit(parsed_ast)
                    nodes, edges, labels = visitor.nodes, visitor.edges, visitor.labels

                    source_nodes, target_nodes = zip(*edges) if edges else ([], [])

                    results[file_path] = {
                        'nodes': nodes,
                        'source_nodes': source_nodes,
                        'target_nodes': target_nodes,
                        'labels': labels,
                        'names': visitor.names
                    }
                except UnicodeDecodeError:
                    print(f"Unicode decode error encountered in file {file_path}")
                except Exception as e:
                    pass
                #     print(f"Error processing file {file_path}: {e}")

    return results


def parse_python_files_in_directories(directory_paths):
    results = {}

    for directory_path in directory_paths:
        directory_results = parse_python_files_in_directory(directory_path)

        results = {**results, **directory_results}

    print(f"Number of files parsed: {len(results)}")
    print(f"Total number of nodes: {sum(len(result['nodes']) for result in results.values())}")
    print(f"Total number of labeled nodes: {sum(1 for result in results.values() for label in result['labels'] if label)}")

    return results
