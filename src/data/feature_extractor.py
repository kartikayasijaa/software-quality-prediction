"""
Feature extraction module for software quality prediction.
Extracts code metrics from repositories to be used as features for ML models.
"""

import os
import re
import ast
import numpy as np
import pandas as pd
import subprocess
from git import Repo
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
import lizard
from pydriller import Repository

class CodeFeatureExtractor:
    """Extract features from code repositories for quality prediction."""
    
    def __init__(self, repo_path):
        """Initialize with path to code repository."""
        self.repo_path = repo_path
        self.features = {}
        self.file_metrics = {}
        self.supported_languages = ['python', 'java', 'javascript', 'cpp', 'csharp']
        
    def extract_all_features(self):
        """Extract all features from the repository."""
        self._extract_git_metrics()
        self._extract_code_metrics()
        self._extract_architecture_metrics()
        self._extract_test_metrics()
        self._extract_dependency_metrics()
        return self.features
    
    def _get_file_language(self, file_path):
        """Determine the programming language of a file."""
        file_ext = os.path.splitext(file_path)[1].lower()
        ext_to_lang = {
            '.py': 'python',
            '.java': 'java',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp'
        }
        return ext_to_lang.get(file_ext, 'unknown')
    
    def _extract_git_metrics(self):
        """Extract metrics from git history."""
        try:
            repo = Repo(self.repo_path)
            
            # Commit frequency and distribution
            commits = list(repo.iter_commits())
            self.features['commit_count'] = len(commits)
            
            if len(commits) > 0:
                # Time between commits
                commit_dates = [commit.committed_datetime for commit in commits]
                commit_dates.sort()
                if len(commit_dates) > 1:
                    time_diffs = [(commit_dates[i] - commit_dates[i+1]).total_seconds() 
                                 for i in range(len(commit_dates)-1)]
                    self.features['avg_time_between_commits'] = np.mean(time_diffs) / 3600  # in hours
                    self.features['commit_frequency_variance'] = np.var(time_diffs) / 3600  # in hours
                
                # Author diversity
                authors = set(commit.author.name for commit in commits)
                self.features['author_count'] = len(authors)
                
                # File churn
                file_changes = {}
                for commit in commits:
                    for file in commit.stats.files:
                        if file not in file_changes:
                            file_changes[file] = 0
                        file_changes[file] += commit.stats.files[file]['insertions'] + commit.stats.files[file]['deletions']
                
                if file_changes:
                    self.features['avg_file_churn'] = np.mean(list(file_changes.values()))
                    self.features['max_file_churn'] = max(file_changes.values())
            
        except Exception as e:
            print(f"Error extracting git metrics: {e}")
            # Set default values if git metrics extraction fails
            self.features['commit_count'] = 0
            self.features['author_count'] = 0
    
    def _extract_code_metrics(self):
        """Extract metrics from code files."""
        total_loc = 0
        total_complexity = 0
        file_count = 0
        complexity_scores = []
        
        for root, _, files in os.walk(self.repo_path):
            # Skip hidden directories and virtual environments
            if any(part.startswith('.') for part in root.split(os.sep)) or 'venv' in root.split(os.sep) or 'node_modules' in root.split(os.sep):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                language = self._get_file_language(file_path)
                
                if language in self.supported_languages:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Basic metrics
                        loc = len(content.split('\n'))
                        total_loc += loc
                        
                        # Language-specific metrics
                        if language == 'python':
                            metrics = self._get_python_metrics(file_path, content)
                        else:
                            # Use lizard for other languages
                            analysis = lizard.analyze_file(file_path)
                            metrics = {
                                'complexity': sum(func.cyclomatic_complexity for func in analysis.function_list),
                                'function_count': len(analysis.function_list),
                                'avg_function_complexity': np.mean([func.cyclomatic_complexity for func in analysis.function_list]) if analysis.function_list else 0,
                                'loc': loc
                            }
                            
                        self.file_metrics[file_path] = metrics
                        total_complexity += metrics.get('complexity', 0)
                        complexity_scores.append(metrics.get('complexity', 0))
                        file_count += 1
                        
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
        
        # Aggregate metrics
        if file_count > 0:
            self.features['total_loc'] = total_loc
            self.features['avg_loc_per_file'] = total_loc / file_count
            self.features['avg_complexity'] = total_complexity / file_count
            self.features['complexity_variance'] = np.var(complexity_scores) if complexity_scores else 0
            self.features['file_count'] = file_count
        else:
            # Set default values if no files were processed
            self.features['total_loc'] = 0
            self.features['avg_loc_per_file'] = 0
            self.features['avg_complexity'] = 0
            self.features['complexity_variance'] = 0
            self.features['file_count'] = 0
    
    def _get_python_metrics(self, file_path, content):
        """Get detailed metrics for Python files."""
        metrics = {}
        
        # Cyclomatic complexity
        try:
            complexity_results = cc_visit(content)
            total_complexity = sum(item.complexity for item in complexity_results)
            metrics['complexity'] = total_complexity
            metrics['complexity_per_function'] = total_complexity / len(complexity_results) if complexity_results else 0
        except:
            metrics['complexity'] = 0
            metrics['complexity_per_function'] = 0
        
        # Maintainability index
        try:
            mi_result = mi_visit(content, multi=True)
            metrics['maintainability_index'] = mi_result
        except:
            metrics['maintainability_index'] = 0
        
        # Halstead metrics
        try:
            h_result = h_visit(content)
            metrics['halstead_volume'] = h_result.volume
            metrics['halstead_difficulty'] = h_result.difficulty
            metrics['halstead_effort'] = h_result.effort
        except:
            metrics['halstead_volume'] = 0
            metrics['halstead_difficulty'] = 0
            metrics['halstead_effort'] = 0
        
        # Raw metrics
        try:
            raw_metrics = analyze(content)
            metrics['loc'] = raw_metrics.lloc
            metrics['comments'] = raw_metrics.comments
            metrics['comment_ratio'] = raw_metrics.comments / raw_metrics.lloc if raw_metrics.lloc > 0 else 0
        except:
            metrics['loc'] = len(content.split('\n'))
            metrics['comments'] = 0
            metrics['comment_ratio'] = 0
        
        # Code smells and linting
        try:
            # Run pylint as a subprocess instead of using the API
            result = subprocess.run(['pylint', file_path], capture_output=True, text=True)
            pylint_output = result.stdout + result.stderr
            error_count = len([line for line in pylint_output.split('\n') if any(code in line for code in ['C:', 'W:', 'E:', 'F:'])])
            metrics['lint_errors'] = error_count
        except Exception as e:
            print(f"Error running pylint on {file_path}: {e}")
            metrics['lint_errors'] = 0
        
        # AST-based metrics
        try:
            tree = ast.parse(content)
            class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            
            # Method count calculation without using parent_node
            method_count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_count += 1
            
            metrics['class_count'] = class_count
            metrics['function_count'] = function_count
            metrics['method_count'] = method_count
            
            # Inheritance depth
            if class_count > 0:
                inheritance_depths = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        depth = len(node.bases)
                        inheritance_depths.append(depth)
                metrics['avg_inheritance_depth'] = np.mean(inheritance_depths)
            else:
                metrics['avg_inheritance_depth'] = 0
                
        except:
            metrics['class_count'] = 0
            metrics['function_count'] = 0
            metrics['method_count'] = 0
            metrics['avg_inheritance_depth'] = 0
        
        return metrics
    
    def _extract_architecture_metrics(self):
        """Extract architecture-related metrics."""
        # Detect architectural patterns
        file_imports = {}
        dependency_graph = {}
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                language = self._get_file_language(file_path)
                
                if language == 'python':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        tree = ast.parse(content)
                        imports = []
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for name in node.names:
                                    imports.append(name.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    imports.append(node.module)
                        
                        file_imports[file_path] = imports
                        
                        # Build dependency graph
                        rel_path = os.path.relpath(file_path, self.repo_path)
                        dependency_graph[rel_path] = []
                        
                        for imp in imports:
                            # Convert import to potential file path
                            imp_parts = imp.split('.')
                            for i in range(len(imp_parts)):
                                potential_path = os.path.join(self.repo_path, *imp_parts[:i+1])
                                if os.path.exists(potential_path + '.py'):
                                    rel_imp_path = os.path.relpath(potential_path + '.py', self.repo_path)
                                    dependency_graph[rel_path].append(rel_imp_path)
                                    break
                                elif os.path.exists(os.path.join(potential_path, '__init__.py')):
                                    rel_imp_path = os.path.relpath(os.path.join(potential_path, '__init__.py'), self.repo_path)
                                    dependency_graph[rel_path].append(rel_imp_path)
                                    break
                    except Exception as e:
                        print(f"Error analyzing imports in {file_path}: {e}")
        
        # Calculate dependency metrics
        if dependency_graph:
            # Average number of dependencies
            dependency_counts = [len(deps) for deps in dependency_graph.values()]
            self.features['avg_dependencies'] = np.mean(dependency_counts)
            
            # Dependency fan-in and fan-out
            fan_in = {}
            for file, deps in dependency_graph.items():
                for dep in deps:
                    if dep not in fan_in:
                        fan_in[dep] = 0
                    fan_in[dep] += 1
            
            fan_in_values = list(fan_in.values()) if fan_in else [0]
            fan_out_values = dependency_counts
            
            self.features['max_fan_in'] = max(fan_in_values) if fan_in_values else 0
            self.features['avg_fan_in'] = np.mean(fan_in_values)
            self.features['max_fan_out'] = max(fan_out_values) if fan_out_values else 0
            self.features['avg_fan_out'] = np.mean(fan_out_values)
            
            # Circular dependencies
            def has_circular_dependency(graph, start, current, visited):
                if current in visited:
                    return True
                visited.add(current)
                for dep in graph.get(current, []):
                    if dep == start or has_circular_dependency(graph, start, dep, visited.copy()):
                        return True
                return False
            
            circular_deps = 0
            for file in dependency_graph:
                if has_circular_dependency(dependency_graph, file, file, set()):
                    circular_deps += 1
            
            self.features['circular_dependencies'] = circular_deps
        else:
            self.features['avg_dependencies'] = 0
            self.features['max_fan_in'] = 0
            self.features['avg_fan_in'] = 0
            self.features['max_fan_out'] = 0
            self.features['avg_fan_out'] = 0
            self.features['circular_dependencies'] = 0
    
    def _extract_test_metrics(self):
        """Extract testing-related metrics."""
        test_files = []
        test_loc = 0
        
        # Find test files
        test_patterns = [
            r'test_.*\.py$',
            r'.*_test\.py$',
            r'.*Test\.java$',
            r'.*\.test\.js$',
            r'.*\.spec\.js$'
        ]
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                is_test = any(re.match(pattern, file) for pattern in test_patterns) or 'test' in root.lower()
                
                if is_test:
                    test_files.append(file_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        test_loc += len(content.split('\n'))
                    except:
                        pass
        
        # Calculate test metrics
        self.features['test_file_count'] = len(test_files)
        self.features['test_loc'] = test_loc
        
        if self.features.get('total_loc', 0) > 0:
            self.features['test_to_code_ratio'] = test_loc / self.features['total_loc']
        else:
            self.features['test_to_code_ratio'] = 0
            
        # Detect test frameworks
        test_frameworks = set()
        for file_path in test_files:
            language = self._get_file_language(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if language == 'python':
                    if 'unittest' in content:
                        test_frameworks.add('unittest')
                    if 'pytest' in content:
                        test_frameworks.add('pytest')
                elif language == 'javascript':
                    if 'jest' in content:
                        test_frameworks.add('jest')
                    if 'mocha' in content:
                        test_frameworks.add('mocha')
                    if 'jasmine' in content:
                        test_frameworks.add('jasmine')
                elif language == 'java':
                    if 'junit' in content.lower():
                        test_frameworks.add('junit')
                    if 'testng' in content.lower():
                        test_frameworks.add('testng')
            except:
                pass
        
        self.features['test_frameworks'] = list(test_frameworks)
        self.features['test_frameworks_count'] = len(test_frameworks)
    
    def _extract_dependency_metrics(self):
        """Extract dependency-related metrics."""
        dependency_files = [
            'requirements.txt',
            'package.json',
            'pom.xml',
            'build.gradle',
            'Gemfile',
            'Cargo.toml'
        ]
        
        dependencies = []
        
        for dep_file in dependency_files:
            file_path = os.path.join(self.repo_path, dep_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if dep_file == 'requirements.txt':
                        # Parse Python requirements
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                dependencies.append(line)
                    
                    elif dep_file == 'package.json':
                        # Parse npm package.json
                        import json
                        pkg_data = json.loads(content)
                        deps = pkg_data.get('dependencies', {})
                        dev_deps = pkg_data.get('devDependencies', {})
                        dependencies.extend(list(deps.keys()))
                        dependencies.extend(list(dev_deps.keys()))
                except Exception as e:
                    print(f"Error parsing dependency file {dep_file}: {e}")
        
        self.features['dependency_count'] = len(dependencies)
        self.features['has_dependency_management'] = len(dependencies) > 0

# Example usage:
# extractor = CodeFeatureExtractor('/path/to/repo')
# features = extractor.extract_all_features()
# print(features)