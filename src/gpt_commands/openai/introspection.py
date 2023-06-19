import inspect
import json
import typing
from dataclasses import dataclass
from typing import Any, Optional, Union

from dataclasses_jsonschema import JsonSchemaMixin
from docstring_parser import parse
from inspect import signature, getmembers, ismethod


class UnsupportedTypeException(Exception):
    def __init__(self, type: type):
        super().__init__(
            f"Unsupported type: {type.__name__}. Only primitive types, lists, dictionaries, Optional and JsonSchemaMixin dataclasses are supported."
        )


class UnsupportedDictionaryKeyTypeException(Exception):
    def __init__(self, key_type: type):
        super().__init__(
            f"Unsupported dictionary key type: {key_type.__name__}. Only string keys are supported."
        )


def decode_json(json_value: str, hint_type: type) -> object:
    if typing.get_origin(hint_type) == Union:
        if type(None) in typing.get_args(hint_type):
            if json_value is None or json_value == "null":
                return None
            else:
                non_none_type = [
                    t for t in typing.get_args(hint_type) if t != type(None)
                ]
                if len(non_none_type) != 1:
                    raise UnsupportedTypeException(hint_type)

                return decode_json(json_value, typing.get_args(hint_type)[0])

        raise UnsupportedTypeException(hint_type)
    elif issubclass(hint_type, JsonSchemaMixin):
        return hint_type.from_json(json_value)
    elif typing.get_origin(hint_type) == list:
        item_type = typing.get_args(hint_type)[0]
        return [
            decode_json(
                item_value if isinstance(item_value, str) else json.dumps(item_value),
                item_type,
            )
            for item_value in json.loads(json_value)
        ]
    elif typing.get_origin(hint_type) == dict:
        key_type, value_type = typing.get_args(hint_type)
        if key_type != str:
            raise UnsupportedDictionaryKeyTypeException(key_type.__name__)

        dictionary: dict = json.loads(json_value)
        return {
            key: decode_json(
                value if isinstance(value, str) else json.dumps(value), value_type
            )
            for key, value in dictionary.items()
        }
    elif hint_type == str:
        return json.loads(json_value) if json_value.startswith('"') else json_value
    elif hint_type == int:
        return int(json_value)
    elif hint_type == float:
        return float(json_value)
    elif hint_type == bool:
        return json_value == "true"
    else:
        raise UnsupportedTypeException(hint_type)


def type_to_json_schema(hint_type: type) -> dict:
    if typing.get_origin(hint_type) == typing.Union:
        if hint_type.__name__ != "Optional":
            raise UnsupportedTypeException(hint_type)
        
        actual_type = typing.get_args(hint_type)[0]

        return type_to_json_schema(actual_type)
    if hint_type == str:
        return {"type": "string"}
    elif hint_type == int:
        return {"type": "integer"}
    elif hint_type == float:
        return {"type": "number"}
    elif hint_type == bool:
        return {"type": "boolean"}
    elif typing.get_origin(hint_type) == list:
        item = typing.get_args(hint_type)[0]
        item_type = type_to_json_schema(item)
        return {"type": "array", "items": item_type}
    elif typing.get_origin(hint_type) == dict:
        key, value = typing.get_args(hint_type)
        key_type = type_to_json_schema(key)["type"]

        if key_type != "string":
            raise UnsupportedDictionaryKeyTypeException(key)

        value_type = type_to_json_schema(value)
        return {"type": "object", "additionalProperties": value_type}
    elif issubclass(hint_type, JsonSchemaMixin):
        schema = hint_type.json_schema()
        schema.pop("$schema")
        return schema
    else:
        raise UnsupportedTypeException(hint_type)


@dataclass
class Parameter:
    name: str
    type: type
    optional: bool
    default_value: Any
    description: Optional[str]

    def json_schema(self):
        schema = type_to_json_schema(self.type)
        schema["description"] = self.description
        return schema

    def serialize(self, value: object) -> str:
        return json.dumps(value, default=lambda o: o.to_dict())

    def deserialize(self, value: str) -> object:
        return decode_json(value, self.type)


@dataclass
class Function:
    name: str
    description: Optional[str]
    parameters: dict[str, Parameter]
    has_return: bool

    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: parameter.json_schema()
                    for name, parameter in self.parameters.items()
                },
                "required": [
                    name
                    for name, parameter in self.parameters.items()
                    if not parameter.optional
                ],
            },
        }


@dataclass
class Manager:
    object: object
    functions: dict[str, Function]

    def execute(self, function_name: str, arguments: dict[str, Any]) -> Optional[str]:
        function = getattr(self.object, function_name, None)
        function_definition = self.functions[function_name]

        if function_definition is None:
            raise Exception(f"Function not found: {function_name}")

        if function is None:
            raise Exception(f"Function not found: {function_name}")

        try:
            for parameter in function_definition.parameters.values():
                if parameter.optional:
                    if parameter.name not in arguments:
                        arguments[parameter.name] = parameter.default_value
                        continue

                if parameter.name not in arguments:
                    raise Exception(
                        f"Missing argument in {function_name}: {parameter.name}"
                    )
                
                arguments[parameter.name] = parameter.deserialize(
                    arguments[parameter.name]
                )
            data = function(**arguments)
        except Exception as e:
            raise Exception("Failed to execute function") from e

        if data is None:
            return None

        json_data = json.dumps(data, default=lambda o: o.to_dict())
        return json_data

    def get_function(self, function_name: str) -> Optional[Function]:
        return self.functions.get(function_name, None)

def create_manager(object: object) -> Manager:
    functions: dict[str, Function] = {}

    for name, method in getmembers(object, predicate=ismethod):
        if not name.startswith("_"):
            if method.__doc__ is None:
                raise Exception(f"Missing docstring for function {name}")

            docstring = parse(method.__doc__)
            parameters: dict[str, Parameter] = {}
            docstring_params = {p.arg_name: p for p in docstring.params} or {}
            sign = signature(method)

            for parameter_name in sign.parameters:
                parameter = sign.parameters[parameter_name]

                if parameter.annotation == inspect._empty:
                    raise Exception(
                        f"Missing type hint for parameter {parameter_name} in function {name}"
                    )

                if parameter_name not in docstring_params:
                    raise Exception(
                        f"Missing docstring for parameter {parameter_name} in function {name}"
                    )

                docstring_param = docstring_params[parameter_name]

                parameters[parameter_name] = Parameter(
                    name=parameter_name,
                    type=parameter.annotation,
                    optional=parameter.default != inspect.Parameter.empty,
                    description=docstring_param.description,
                    default_value=None
                    if parameter.default == inspect.Parameter.empty
                    else parameter.default,
                )

            functions[name] = Function(
                name=name,
                description=docstring.short_description,
                parameters=parameters,
                has_return=sign.return_annotation != inspect.Signature.empty
                and sign.return_annotation != None,
            )

    return Manager(object, functions)
