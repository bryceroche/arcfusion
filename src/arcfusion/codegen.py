"""
Code Generator - Generate PyTorch code from dreamed architectures.

Takes a list of components and produces a runnable nn.Module.
"""

import re
import textwrap
from dataclasses import dataclass
from typing import Optional
from .db import Component
from .composer import get_component_category


def sanitize_class_name(name: str) -> str:
    """Convert component name to valid Python class name."""
    # Remove parenthetical content
    name = re.sub(r'\([^)]*\)', '', name)
    # Remove special characters, keep alphanumeric
    name = re.sub(r'[^a-zA-Z0-9]', '', name)
    # Ensure starts with letter
    if name and name[0].isdigit():
        name = 'M' + name
    return name or 'Module'


def sanitize_var_name(name: str) -> str:
    """Convert component name to valid Python variable name."""
    # Remove parenthetical content
    name = re.sub(r'\([^)]*\)', '', name)
    # Convert to snake_case
    name = re.sub(r'([A-Z])', r'_\1', name).lower()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name or 'module'


def extract_hyperparams(component: Component) -> dict:
    """Extract hyperparameters with defaults."""
    params = component.hyperparameters or {}

    # Common defaults if not specified
    defaults = {
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.1,
        'vocab_size': 32000,
        'max_len': 5000,
    }

    # Merge with component-specific params
    result = defaults.copy()
    result.update(params)
    return result


def extract_forward_body(code: str) -> str:
    """Extract just the forward method body from a code sketch."""
    if not code:
        return "return x"

    lines = code.strip().split('\n')

    # Find forward method and extract its body
    in_forward = False
    forward_lines = []
    base_indent = 0

    for line in lines:
        if 'def forward' in line:
            in_forward = True
            # Get the indentation of the def line
            base_indent = len(line) - len(line.lstrip())
            continue

        if in_forward:
            # Check if we've exited the method (new def or class at same/lower indent)
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped) if stripped else 999

            if stripped and (stripped.startswith('def ') or stripped.startswith('class ')):
                if current_indent <= base_indent:
                    break

            # Add line if it's part of forward (or empty line)
            if line.strip() or forward_lines:
                # Remove the base indentation
                if current_indent > base_indent:
                    forward_lines.append(line[base_indent + 4:] if len(line) > base_indent + 4 else line.strip())
                elif not line.strip():
                    forward_lines.append('')

    if forward_lines:
        return '\n'.join(forward_lines).strip()

    # No forward found, return simple passthrough
    return "return x"


def generate_component_class(component: Component, index: int) -> str:
    """Generate a PyTorch nn.Module class for a single component."""
    class_name = sanitize_class_name(component.name)
    category = get_component_category(component)

    # Extract hyperparameters
    params = extract_hyperparams(component)
    d_model = params.get('d_model', 512)

    # Determine what layers to initialize and forward logic based on category
    init_layers = ""
    forward_body = ""

    if category == 'attention':
        init_layers = """
        self.n_heads = kwargs.get('n_heads', 8)
        self.head_dim = d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))"""
        forward_body = """
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = kwargs.get('mask')
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)"""

    elif category == 'layer':
        if 'norm' in component.name.lower():
            init_layers = """
        self.norm = nn.LayerNorm(d_model)"""
            forward_body = """
        return self.norm(x)"""
        elif 'feed' in component.name.lower() or 'ffn' in component.name.lower():
            init_layers = """
        self.d_ff = kwargs.get('d_ff', d_model * 4)
        self.fc1 = nn.Linear(d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))"""
            forward_body = """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x"""
        else:
            init_layers = """
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))"""
            forward_body = """
        return self.dropout(x)"""

    elif category == 'position':
        init_layers = """
        self.max_len = kwargs.get('max_len', 5000)
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))"""
        forward_body = """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]"""

    elif category == 'output':
        init_layers = """
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.proj = nn.Linear(d_model, self.vocab_size)"""
        forward_body = """
        return self.proj(x)"""

    elif category == 'structure':
        init_layers = """
        self.n_layers = kwargs.get('n_layers', 6)
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(d_model)"""
        forward_body = """
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.norm(x)"""

    else:
        # Default passthrough
        forward_body = """
        return x"""

    # Build the class
    class_code = f'''
class {class_name}(nn.Module):
    """
    {component.name}

    {(component.description or 'Generated component.')[:200]}

    Interface:
        Input:  {component.interface_in.get('shape', 'variable') if component.interface_in else 'variable'}
        Output: {component.interface_out.get('shape', 'variable') if component.interface_out else 'variable'}
    """

    def __init__(self, d_model={d_model}, **kwargs):
        super().__init__()
        self.d_model = d_model{init_layers}

    def forward(self, x, **kwargs):{forward_body}
'''

    return class_code


def generate_architecture_class(
    components: list[Component],
    name: str = "DreamedArchitecture",
    description: str = ""
) -> str:
    """Generate a complete PyTorch architecture from components."""

    # Generate individual component classes
    component_classes = []
    for i, comp in enumerate(components):
        component_classes.append(generate_component_class(comp, i))

    # Generate the main architecture class
    var_names = []
    init_lines = []
    forward_lines = []

    for i, comp in enumerate(components):
        class_name = sanitize_class_name(comp.name)
        var_name = f"self.{sanitize_var_name(comp.name)}"

        # Handle duplicate names
        base_var = sanitize_var_name(comp.name)
        if base_var in [v.split('.')[-1] for v in var_names]:
            var_name = f"self.{base_var}_{i}"

        var_names.append(var_name)
        init_lines.append(f"        {var_name} = {class_name}(d_model=d_model)")

        # Forward pass - chain components
        category = get_component_category(comp)
        if category == 'position':
            forward_lines.append(f"        x = x + {var_name}(x)")
        elif category == 'output':
            forward_lines.append(f"        x = {var_name}(x)")
        else:
            forward_lines.append(f"        x = {var_name}(x)")

    # Build main class
    main_class = f'''
class {name}(nn.Module):
    """
    {description or 'Dreamed architecture generated by ArcFusion.'}

    Components:
{chr(10).join(f'        - {c.name}' for c in components)}
    """

    def __init__(self, d_model=512, vocab_size=32000, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Component modules
{chr(10).join(init_lines)}

    def forward(self, x, **kwargs):
        """
        Forward pass through all components.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model] or [batch, seq_len]
            **kwargs: Additional arguments (mask, memory, etc.)

        Returns:
            Output tensor
        """
{chr(10).join(forward_lines)}
        return x
'''

    return main_class


def generate_full_module(
    components: list[Component],
    name: str = "DreamedArchitecture",
    description: str = "",
    include_imports: bool = True,
    include_example: bool = True
) -> str:
    """
    Generate a complete, runnable Python module.

    Args:
        components: List of components from a dream
        name: Name for the architecture class
        description: Description for the module docstring
        include_imports: Include import statements
        include_example: Include example usage at bottom

    Returns:
        Complete Python module as a string
    """
    parts = []

    # Module docstring
    parts.append(f'''"""
{name} - Generated by ArcFusion

{description or 'This architecture was dreamed up by combining compatible components.'}

Components ({len(components)}):
{chr(10).join(f'  - {c.name}' for c in components)}

Generated from ArcFusion dream composition.
"""
''')

    # Imports
    if include_imports:
        parts.append('''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
''')

    # Component classes
    for i, comp in enumerate(components):
        parts.append(generate_component_class(comp, i))

    # Main architecture class
    parts.append(generate_architecture_class(components, name, description))

    # Example usage
    if include_example:
        parts.append(f'''

# Example usage
if __name__ == "__main__":
    # Create model
    model = {name}(d_model=512, vocab_size=32000)
    print(f"Model: {{model.__class__.__name__}}")
    print(f"Parameters: {{sum(p.numel() for p in model.parameters()):,}}")

    # Test forward pass
    batch_size, seq_len, d_model = 2, 128, 512
    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape:  {{x.shape}}")
    print(f"Output shape: {{output.shape}}")
''')

    return '\n'.join(parts)


@dataclass
class GeneratedCode:
    """Result of code generation."""
    code: str
    name: str
    num_components: int
    component_names: list[str]

    def save(self, path: str, validate: bool = True) -> bool:
        """
        Save generated code to file.

        Args:
            path: File path to save to
            validate: If True, validate syntax before saving (default True)

        Returns:
            True if saved successfully, False if validation failed

        Raises:
            ValueError: If validation is enabled and code has syntax errors
        """
        if validate:
            valid, error = self.validate_syntax()
            if not valid:
                raise ValueError(f"Cannot save invalid Python code: {error}")

        with open(path, 'w') as f:
            f.write(self.code)
        return True

    def validate_syntax(self) -> tuple[bool, Optional[str]]:
        """Check if generated code has valid Python syntax."""
        try:
            compile(self.code, '<generated>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, str(e)


class CodeGenerator:
    """Generate PyTorch code from dreamed architectures."""

    def __init__(self, db=None):
        self.db = db

    def generate(
        self,
        components: list[Component],
        name: str = "DreamedArchitecture",
        description: str = ""
    ) -> GeneratedCode:
        """
        Generate code from a list of components.

        Args:
            components: List of Component objects
            name: Name for the generated class
            description: Optional description

        Returns:
            GeneratedCode object with the generated module
        """
        code = generate_full_module(
            components=components,
            name=name,
            description=description,
            include_imports=True,
            include_example=True
        )

        return GeneratedCode(
            code=code,
            name=name,
            num_components=len(components),
            component_names=[c.name for c in components]
        )

    def generate_from_dream(
        self,
        strategy: str = "greedy",
        name: str = "DreamedArchitecture",
        **kwargs
    ) -> GeneratedCode:
        """
        Dream an architecture and generate code for it.

        Args:
            strategy: Dream strategy (greedy, random, crossover, mutate)
            name: Name for the generated class
            **kwargs: Arguments for the dream strategy

        Returns:
            GeneratedCode object
        """
        from .composer import EngineComposer

        if not self.db:
            raise ValueError("Database required for generate_from_dream")

        composer = EngineComposer(self.db)
        components, score = composer.dream(strategy, **kwargs)

        description = f"Architecture dreamed using '{strategy}' strategy (score: {score:.2f})"

        return self.generate(components, name, description)
