import ast


class GraphExtractor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.labels = []
        self.names = []
        self.parent_node = {}
        self.current_node = -1

    def visit(self, node):
        node_id = len(self.nodes)
        node_type = type(node).__name__

        if isinstance(node, ast.AnnAssign) or isinstance(node, ast.Assign):
            node_type = 'Assign'

        self.nodes.append(node_type)

        label = self.determine_label(node)
        self.labels.append(label)

        name = self.get_name(node)
        self.names.append(name)

        if self.current_node != -1:
            self.edges.append([self.current_node, node_id])
            self.parent_node[node_id] = self.current_node

        parent_node = self.current_node
        self.current_node = node_id

        if isinstance(node, ast.Name):
            self.visit_Name(node)
        self.generic_visit(node)
        self.current_node = parent_node

    def visit_Name(self, node):
        """Handle Name nodes."""
        parent_id = self.parent_node.get(self.current_node)
        if parent_id is not None and self.nodes[parent_id] == 'Assign':
            # Transfer the label from the Assign node to this Name node
            self.labels[self.current_node] = self.labels[parent_id]

        self.generic_visit(node)

    def determine_label(self, node):
        if isinstance(node, ast.arg) and node.annotation is not None:
            return self.node_to_string(node.annotation)
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            return self.node_to_string(node.annotation)
        elif isinstance(node, ast.FunctionDef) and node.returns is not None:
            return self.node_to_string(node.returns)
        return None

    def get_name(self, node):
        """Get the name of the variable or function if applicable."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.FunctionDef):
            return node.name
        return None

    def node_to_string(self, node):
        """Convert AST node to string representation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Constant):
            return node.value
        elif hasattr(ast, 'unparse'):
            return ast.unparse(node)
        else:
            # Fallback for other node types
            return self.generic_node_to_string(node)

    def generic_node_to_string(self, node):
        """Fallback method to convert an AST node to a string."""
        if isinstance(node, ast.Attribute):
            value = self.node_to_string(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self.node_to_string(node.value)
            slice = self.node_to_string(node.slice)
            return f"{value}[{slice}]"
        elif isinstance(node, ast.Index):
            return self.node_to_string(node.value)
        elif isinstance(node, ast.Tuple):
            elements = [self.node_to_string(e) for e in node.elts]
            return f"({', '.join(elements)})"
        elif isinstance(node, ast.List):
            elements = [self.node_to_string(e) for e in node.elts]
            return f"[{', '.join(elements)}]"
        else:
            return type(node).__name__
