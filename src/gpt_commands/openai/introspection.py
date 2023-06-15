from typing import List, Optional
from docstring_parser import parse
from dataclasses import dataclass

@dataclass
class Parameter:
    name: str
    type: str
    optional: bool
    description: Optional[str]

    def json_schema_type(self):
        if self.type == "str":
            return "string"
        elif self.type == "int":
            return "integer"
        elif self.type == "float":
            return "number"
        elif self.type == "bool":
            return "boolean"
        else:
            return "object"
    
    def json_schema(self):
        return {
            "type": self.json_schema_type(),
            "description": self.description
        }

@dataclass
class Function:
    name: str
    description: Optional[str]
    parameters: List[Parameter]

    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    parameter.name: parameter.json_schema()
                    for parameter in self.parameters
                },
                "required": [parameter.name for parameter in self.parameters if not parameter.optional]
            }
        }

def get_functions(object: object) -> List[Function]:
    functions = []
    for name, function in object.__class__.__dict__.items():
        if not name.startswith("_"):
            docstring = parse(function.__doc__)
            parameters = []
            for parameter in docstring.params:
                annotation = function.__annotations__[parameter.arg_name]

                parameters.append(Parameter(
                    name=parameter.arg_name,
                    type=annotation.__name__,
                    optional=parameter.is_optional or False,
                    description=parameter.description
                ))
            functions.append(Function(
                name=name,
                description=docstring.short_description,
                parameters=parameters
            ))
    return functions

